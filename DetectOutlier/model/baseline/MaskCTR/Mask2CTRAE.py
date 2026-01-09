# -*- coding: utf-8 -*-
"""Using AutoEncoder with Outlier Detection (PyTorch)
important mask 乘在了GT值上
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

from DetectOutlier.model.baseline.base import BaseDetector
from DetectOutlier.utils.stat_models import pairwise_distances_no_broadcast
from DetectOutlier.utils.torch_utility import get_activation_by_name
from torch.distributions.distribution import Distribution
from torch.distributions import MultivariateNormal, Normal
import random
import matplotlib.pyplot as plt
from torch.autograd import Function
from DetectOutlier.utils.utility import unique as unique2
from DetectOutlier.model.loss.InfoNCE import InfoNCE
from DetectOutlier.model.loss.ClusterLoss import ClusterLoss

class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None, mean=None, std=None, y_train_label=None
                 ):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std
        if y_train_label is not None:
            self.y_train_label = np.expand_dims(y_train_label, axis=1)
        else:
            self.y_train_label = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if self.mean is not None and self.std is not None:
            if np.all(self.std > 0):
                sample = (sample - self.mean) / self.std
            else:
                assert "std is 0"
            # assert_almost_equal (0, sample.mean(), decimal=1)

        if self.y_train_label is not None:
            batch_y_train_label = self.y_train_label[idx, :]
            return torch.from_numpy(sample), idx, batch_y_train_label
        else:
            return torch.from_numpy(sample), idx


# InfoNCE: https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py

class AdaptiveModule(nn.Module):
    def __init__(self,
                 adaptive_avg_feature_num=5000,
                 adaptive_max_feature_num=5000
                 ):
        super(AdaptiveModule, self).__init__()
        # set the adaptive feature
        # self.ada_avg_feat_layer = nn.AdaptiveAvgPool1d(adaptive_avg_feature_num)
        # self.ada_max_feat_layer = nn.AdaptiveMaxPool1d(adaptive_max_feature_num)
    def forward(self, x):
        # avg_feat = self.ada_avg_feat_layer(x)
        # max_feat = self.ada_max_feat_layer(x)
        # x_feat = torch.cat([max_feat, avg_feat], dim=1)
        # return x_feat
        return x

class ADSDAutoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 n_samples,
                 hidden_neurons=(128, 64),
                 dropout_rate=0.2,
                 batch_norm=True,
                 hidden_activation='relu',
                 ):

        # initialize the super class
        super(ADSDAutoencoder, self).__init__()

        # save the default values
        self.n_features = n_features
        self.n_samples = n_samples

        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.hidden_activation = hidden_activation

        # set the importance mask
        self.important_mask = torch.nn.Parameter(torch.FloatTensor(1, self.n_features))
        # init the important mask with
        nn.init.constant_(self.important_mask, 1)



        # set the adaptive feature
        self.adaptive_format = AdaptiveModule()

        # create the dimensions for the input and hidden layers
        self.layers_neurons_encoder_ = [self.n_features, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_encoder_[::-1]

        # get the object for the activations functions
        self.activation = get_activation_by_name(hidden_activation)

        # initialize encoder and decoder as a sequential
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

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

        # fill the decoder layer
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

        # fill the cluster projection
        self.cluster_projector = nn.Sequential(
            nn.Linear(hidden_neurons[-1], hidden_neurons[-1]),
            nn.ReLU(),
            nn.Linear(hidden_neurons[-1], 2),
            nn.Sigmoid()
        )
    #version 1:
    def forward(self, x):
        x_feat = self.adaptive_format(x)
        # we could return the latent representation here after the encoder
        # as the latent representation
        h = self.encoder(x_feat)
        X_hat = self.decoder(h)

        gt_encode = self.cluster_projector(h)
        return {
            # "important_feature": important_feature,
            "gt_encode": gt_encode,
            "x_feat": X_hat,
            "latent_vec": h,
        }

    # # version 2:
    # def forward(self, x):
    #     x_feat = self.adaptive_format(x)
    #     # we could return the latent representation here after the encoder
    #     # as the latent representation
    #     h = self.encoder(x_feat)
    #     X_hat = self.decoder(h)
    #
    #     # important_feature = torch.softmax(self.important_mask, dim=1).repeat(x.size(0), 1) * X_hat
    #     # important_feature = self.important_mask.repeat(x.size(0), 1) * X_hat
    #     return {
    #         # "important_feature": important_feature,
    #         "x_feat": X_hat,
    #     }

class Mask2CTRAE(BaseDetector):
    def __init__(self,
                 hidden_neurons=None,
                 hidden_activation='relu',
                 batch_norm=True,
                 learning_rate=1e-3,
                 epochs=100,
                 batch_size=32,
                 dropout_rate=0.2,
                 weight_decay=1e-5,
                 preprocessing=True,
                 loss_fn_name="MSE",
                 contamination=0.1,
                 topk_value=1,
                 device=None):
        super(Mask2CTRAE, self).__init__(contamination=contamination)

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
        self.loss_fn_name = loss_fn_name
        # self.verbose = verbose
        self.topk_value = topk_value
        self.device = device

        # # create default loss functions
        if "MSE" in self.loss_fn_name:
            self.loss_fn = torch.nn.MSELoss()

        if "mask_loss" in self.loss_fn_name:
            self.mask_loss = InfoNCE(temperature=0.01)

        if "cluster_loss" in self.loss_fn_name:
            self.cluster_loss = ClusterLoss(temperature=0.2, device=self.device)

        # create default calculation device (support GPU if available)
        if self.device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        # default values for the amount of hidden neurons
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32]

    # noinspection PyUnresolvedReferences
    def fit(self, X, y=None, **kwargs):
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        if "y_train_label" in kwargs:
            y_train_label = kwargs['y_train_label']
        else:
            y_train_label = None

        n_samples, n_features = X.shape[0], X.shape[1]

        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            # self.max, self.min = np.max(np.max(X)), np.min(np.min(X))
            train_set = PyODDataset(
                X=X, mean=self.mean, std=self.std,
                y_train_label=y_train_label
            )
        else:
            train_set = PyODDataset(X=X, y_train_label=y_train_label)

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        # initialize the model
        self.model = ADSDAutoencoder(
            n_features=n_features,
            n_samples=n_samples,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation,
        )

        # move to device and print model information
        self.model = self.model.to(self.device)
        print(self.model)

        # train the autoencoder to find the best one
        self._train_autoencoder(train_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)

        self._process_decision_scores()
        return self

    def _train_autoencoder(self, train_loader):
        """Internal function to train the autoencoder

        Parameters
        ----------
        train_loader : torch dataloader
            Train data.
        """
        # important_mask 与 AE设置为不同的学习
        AE_para = []
        mask_para = []
        for iname, ipara in self.model.named_parameters():
            if "important_mask" in iname:
                mask_para.append(ipara)
            else:
                AE_para.append(ipara)
        param_groups = [
            {'params': mask_para, 'lr_mult': 100},
            {'params': AE_para, 'lr_mult': 1}]

        optimizer = torch.optim.Adam(
            param_groups, lr=self.learning_rate,
            weight_decay=self.weight_decay)

        def adjust_lr():
            for g in optimizer.param_groups:
                g['lr'] = self.learning_rate * g.get('lr_mult', 1)

        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.best_model_dict = None

        for epoch in range(self.epochs):
            adjust_lr()
            overall_loss = []
            contrastive_acc = []
            pseudo_acc = []
            for data, data_idx, batch_y_label in train_loader:
                data = data.to(self.device).float()
                batch_y_label = batch_y_label.to(self.device)
                model_outputs = self.model(data)

                format_data = self.model.adaptive_format(data)
                # reconstruction loss
                if "MSE" in self.loss_fn_name:
                    loss = self.loss_fn(format_data, model_outputs['x_feat'])

                if "mask_loss" in self.loss_fn_name:
                    # todo: make sure the self.important_mask is working
                    # print(torch.max(self.model.important_mask), torch.min(self.model.important_mask))
                    # version 1:
                    mask_loss_res = self.mask_loss(
                        model_outputs['x_feat'],
                        torch.sigmoid(self.model.important_mask) * format_data
                    )
                    loss += mask_loss_res['loss_v']
                    # t1 = batch_y_label[mask_loss_res['recog_mask']]
                    # print(f"unable to recognize outlier/inlier: {torch.where(t1==1)[0].size(0)}/{t1.size(0)}")
                    contrastive_acc.append(mask_loss_res['accuracy'].item())

                if (epoch>=30) and ("cluster_loss" in self.loss_fn_name):
                    cluster_loss_res = self.cluster_loss(
                        model_outputs['gt_encode'],
                        self.model.cluster_projector(
                            self.model.encoder(torch.sigmoid(self.model.important_mask) * format_data)
                        )
                    )
                    loss += cluster_loss_res['loss_v']
                    # pseudo_acc.append(cluster_loss_res['accuracy'].item())

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())

            print('epoch {epoch}: training loss {train_loss}, contrastive accuracy {ctr_accuracy}'.format(
                epoch=epoch, train_loss=np.mean(overall_loss), ctr_accuracy=np.mean(contrastive_acc)))

            # track the best model so far
            if np.mean(contrastive_acc) >= self.best_acc:
                if np.mean(overall_loss) <= self.best_loss:
                    # print("epoch {ep} is the current best; loss={loss}".format(ep=epoch, loss=np.mean(overall_loss)))
                    self.best_acc = np.mean(contrastive_acc)
                    self.best_loss = np.mean(overall_loss)
                    self.best_model_dict = self.model.state_dict()

    def decision_function(self, X, sample_name=None):
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
            # dataset = PyODDataset(X=X, mean=self.mean, std=self.std)
            dataset = PyODDataset(
                X=X, mean=self.mean, std=self.std
            )
        else:
            dataset = PyODDataset(X=X)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)
        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])
        with torch.no_grad():
            for data, data_idx in dataloader:
                data_cuda = data.to(self.device).float()
                infer_res = self.model(data_cuda)
                # this is the outlier score
                outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                    self.model.adaptive_format(data).cpu().numpy(),
                    # data,
                    infer_res['x_feat'].cpu().numpy()
                )

        return {
            'score': outlier_scores,
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

    def predict(self, X, return_confidence=False, sample_name=None, rerank=False):
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

        # check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        prediction = self.labels_

        if return_confidence:
            confidence = self.predict_confidence(X)
            return prediction, confidence

        return prediction