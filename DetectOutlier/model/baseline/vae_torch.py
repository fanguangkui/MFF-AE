# -*- coding: utf-8 -*-
"""Using AutoEncoder with Outlier Detection (PyTorch)
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn

from .base import BaseDetector
from DetectOutlier.utils.stat_models import pairwise_distances_no_broadcast
from DetectOutlier.utils.torch_utility import get_activation_by_name


class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None, mean=None, std=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
            # assert_almost_equal (0, sample.mean(), decimal=1)

        return torch.from_numpy(sample), idx


class VariationalAutoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_neurons=(128, 64),
                 latent_dim=2,
                 dropout_rate=0.2,
                 batch_norm=True,
                 hidden_activation='relu'):

        # initialize the super class
        super(VariationalAutoencoder, self).__init__()

        # save the default values
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.hidden_activation = hidden_activation

        # create the dimensions for the input and hidden layers
        self.layers_neurons_encoder_ = [self.n_features, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_encoder_[::-1]

        # get the object for the activations functions
        self.activation = get_activation_by_name(hidden_activation)

        # initialize encoder and decoder as a sequential
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0


        # fill the encoder sequential with hidden layers
        for idx, layer in enumerate(self.layers_neurons_encoder_[:-1]):

            # create a linear layer of neurons
            self.encoder.add_module(
                "linear" + str(idx),
                torch.nn.Linear(layer,self.layers_neurons_encoder_[idx + 1]))

            # add a batch norm per layer if wanted (leave out first layer)
            if batch_norm:
                self.encoder.add_module("batch_norm" + str(idx),
                                        nn.BatchNorm1d(
                                            self.layers_neurons_encoder_[
                                                idx + 1]))

            # create the activation
            self.encoder.add_module(self.hidden_activation + str(idx),
                                    self.activation)

            # create a dropout layer
            self.encoder.add_module("dropout" + str(idx),
                                    torch.nn.Dropout(dropout_rate))

        self.mu_layer = torch.nn.Linear(
            self.layers_neurons_encoder_[-1],
            latent_dim
        )
        self.sigma_layer = torch.nn.Linear(
            self.layers_neurons_encoder_[-1],
            latent_dim
        )

        # fill the decoder layer
        self.layers_neurons_decoder_ = [latent_dim]+self.layers_neurons_decoder_
        for idx, layer in enumerate(self.layers_neurons_decoder_[:-1]):

            # create a linear layer of neurons
            self.decoder.add_module(
                "linear" + str(idx),
                torch.nn.Linear(layer,self.layers_neurons_decoder_[idx + 1]))

            # create a batch norm per layer if wanted (only if it is not the
            # last layer)
            if batch_norm and idx < len(self.layers_neurons_decoder_[:-1]) - 1:
                self.decoder.add_module("batch_norm" + str(idx),
                                        nn.BatchNorm1d(
                                            self.layers_neurons_decoder_[
                                                idx + 1]))

            # create the activation
            self.decoder.add_module(self.hidden_activation + str(idx),
                                    self.activation)

            # create a dropout layer (only if it is not the last layer)
            if idx < len(self.layers_neurons_decoder_[:-1]) - 1:
                self.decoder.add_module("dropout" + str(idx),
                                        torch.nn.Dropout(dropout_rate))



    def forward(self, x):
        # we could return the latent representation here after the encoder
        # as the latent representation
        x = self.encoder(x)
        mu = self.mu_layer(x)
        std = torch.exp(0.5*self.sigma_layer(x))
        eps = torch.randn_like(std).to(std.device)
        z = mu + eps * std
        x = self.decoder(z)
        return {
            "laten_vec": z,
            "score": x,
            "mu": mu,
            "sigma": std,
        }


class VAE_Pytorch(BaseDetector):
    def __init__(self,
                 hidden_neurons=None,
                 hidden_activation='relu',
                 batch_norm=True,
                 learning_rate=1e-3,
                 epochs=100,
                 batch_size=32,
                 dropout_rate=0.2,
                 weight_decay=1e-5,
                 # validation_size=0.1,
                 preprocessing=True,
                 loss_fn=None,
                 # verbose=1,
                 # random_state=None,
                 contamination=0.1,
                 device=None):
        super(VAE_Pytorch, self).__init__(contamination=contamination)

        # save the initialization values
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing
        self.loss_fn = loss_fn
        # self.verbose = verbose
        self.device = device

        # create default loss functions
        if self.loss_fn is None:
            self.loss_fn = torch.nn.MSELoss()

        # create default calculation device (support GPU if available)
        if self.device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        # default values for the amount of hidden neurons
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32]

    def _train_autoencoder(self, train_loader):
        """Internal function to train the autoencoder

        Parameters
        ----------
        train_loader : torch dataloader
            Train data.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay)

        self.best_loss = float('inf')
        self.best_model_dict = None

        for epoch in range(self.epochs):
            overall_loss = []
            for data, data_idx in train_loader:
                data = data.to(self.device).float()
                model_res = self.model(data)
                # reconstruction loss
                recon_loss = self.loss_fn(data, model_res['score'])
                # kl loss
                kl_loss = (model_res['sigma'] ** 2 + model_res['mu'] ** 2 - torch.log(model_res['sigma']) - 1 / 2).sum()

                loss = kl_loss+recon_loss

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())
            print('epoch {epoch}: training loss {train_loss} '.format(
                epoch=epoch, train_loss=np.mean(overall_loss)))

            # track the best model so far
            if np.mean(overall_loss) <= self.best_loss:
                # print("epoch {ep} is the current best; loss={loss}".format(ep=epoch, loss=np.mean(overall_loss)))
                self.best_loss = np.mean(overall_loss)
                self.best_model_dict = self.model.state_dict()

    # noinspection PyUnresolvedReferences
    def fit(self, X, y=None, **kwargs):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        n_samples, n_features = X.shape[0], X.shape[1]

        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std)

        else:
            train_set = PyODDataset(X=X)

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        # initialize the model
        self.model = VariationalAutoencoder(
            n_features=n_features,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation)

        # move to device and print model information
        self.model = self.model.to(self.device)
        print(self.model)

        # train the autoencoder to find the best one
        self._train_autoencoder(train_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)

        self._process_decision_scores()
        return self



    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model', 'best_model_dict'])
        X = check_array(X)

        # note the shuffle may be true but should be False
        if self.preprocessing:
            dataset = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            dataset = PyODDataset(X=X)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)
        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])
        feat_dist = np.zeros(X.shape)
        with torch.no_grad():
            for data, data_idx in dataloader:
                data_cuda = data.to(self.device).float()
                model_res = self.model(data_cuda)
                feat_dist[data_idx] = model_res['score'].cpu().numpy()
                # this is the outlier score
                outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                    data, model_res['score'].cpu().numpy())

        return {
            'score': outlier_scores,
            'feat_dist': feat_dist,
        }
    def _process_decision_scores(self, sample_name=None):
        """Internal function to calculate key attributes:

        - threshold_: used to decide the binary label
        - labels_: binary labels of training data

        Returns
        -------
        self
        """
        # np.where(sample_name.values=='iBAQ 38')[0]
        self.threshold_ = np.percentile(self.decision_scores_['score'],
                                     100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_['score'] > self.threshold_).astype(
            'int').ravel()

        # calculate for predict_proba()

        self._mu = np.mean(self.decision_scores_['score'])
        self._sigma = np.std(self.decision_scores_['score'])

        return self

    def predict(self, X, return_confidence=False):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        return_confidence : boolean, optional(default=False)
            If True, also return the confidence of prediction.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        confidence : numpy array of shape (n_samples,).
            Only if return_confidence is set to True.
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        pred_score = self.decision_function(X)
        prediction = (pred_score['score'] > self.threshold_).astype('int').ravel()

        if return_confidence:
            confidence = self.predict_confidence(X)
            return prediction, confidence

        return prediction