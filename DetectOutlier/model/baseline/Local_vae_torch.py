# -*- coding: utf-8 -*-
"""Using AutoEncoder with Outlier Detection (PyTorch)
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import collections
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn
from copy import deepcopy
from .base import BaseDetector
import torch.nn.functional as F

from DetectOutlier.utils.utility import normalization
from DetectOutlier.model.baseline.knn import KNN
from DetectOutlier.model.baseline.lof import LOF
from DetectOutlier.utils.stat_models import pairwise_distances_no_broadcast
from DetectOutlier.utils.torch_utility import get_activation_by_name
from DetectOutlier.model.loss.MemoryLoss import ClusterMemory

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length

        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None, mean=None, std=None, pseudo_y=None, feature_label=None, fill_mask=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std
        self.feature_label = feature_label
        self.pseudo_y = pseudo_y
        if fill_mask is not None:
            self.fill_mask = fill_mask
        else:
            self.fill_mask = np.ones_like(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]
        sample_fill_mask = self.fill_mask[idx, :]

        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
            # assert_almost_equal (0, sample.mean(), decimal=1)
        if self.pseudo_y is not None:
            pseudo_label = self.pseudo_y[idx, :]
        else:
            pseudo_label = -100

        if self.feature_label is not None:
            pseudo_feature = self.feature_label[idx, :]
        else:
            pseudo_feature = -100

        return torch.from_numpy(sample), idx, pseudo_label, pseudo_feature, sample_fill_mask

class SingleNet(nn.Module):
    def __init__(self, x_dim, h_dim, num_layers):
        super(SingleNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers-1):
            net.append(nn.Linear(input_dim,h_dim,bias=False))
            net.append(nn.ReLU())
            input_dim= h_dim
        net.append(nn.Linear(input_dim,x_dim,bias=False))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        mask = torch.sigmoid(out)

        return mask * x

class MSELossFunction(nn.Module):
    def __init__(self):
        super(MSELossFunction, self).__init__()

    def forward(self, x_input, x_pred):
        # mse loss
        sub_result = x_pred - x_input
        mse_score = torch.norm(sub_result, p=2, dim=1)
        mse = torch.mean(mse_score)
        return mse_score, mse


class VariationalAutoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 n_samples,
                 hidden_neurons=(128, 64),
                 dropout_rate=0.2,
                 latent_dim=2,
                 batch_norm=True,
                 mask_num=15,
                 hidden_activation='relu'):

        # initialize the super class
        super(VariationalAutoencoder, self).__init__()

        # save the default values
        self.n_features = n_features
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.mask_num = mask_num
        self.hidden_activation = hidden_activation



        self.maskmodel = SingleNet(self.n_features, self.n_features, 2)


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

        # set memory module:
        self.memory = None
    def forward(self, x):
        # we could return the latent representation here after the encoder
        # x_mask = self.maskmodel(x)

        # as the latent representation
        mid_x_feat = self.encoder(x)
        x_feat = self.decoder(mid_x_feat)


        # mid_x_feat = mid_x_feat.reshape(x.shape[0], self.mask_num, -1)
        # x_feat = x_feat.reshape(x.shape[0], self.mask_num, x.shape[-1])
        return {
            "laten_vec":mid_x_feat,
            "X_hat": x_feat,
        }


class LocalVAE(BaseDetector):
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
                 loss_fn_name="MSE",
                 start_pesudo_epoch=80,
                 mask_num=15,
                 lambda_v=5,
                 # verbose=1,
                 # random_state=None,
                 contamination=0.1,
                 mem_temp=0.02,
                 use_hard=False,
                 mem_momentum=0.05,
                 mem_loss_weight=0.05,
                 kl_loss_weight=0.05,
                 device=None):
        super(LocalVAE, self).__init__(contamination=contamination)

        self.method_type = "DeepLearning"

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
        self.start_pesudo_epoch = start_pesudo_epoch
        # self.verbose = verbose
        self.device = device
        self.mem_temp = mem_temp
        self.use_hard = use_hard
        self.mem_momentum = mem_momentum
        self.mem_loss_weight = mem_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.mask_num = mask_num
        self.lambda_v = lambda_v

        # # create default loss functions
        if "MSE" in self.loss_fn_name:
            self.loss_fn = MSELossFunction()
            # self.loss_fn = torch.nn.MSELoss()

        # create default calculation device (support GPU if available)
        if self.device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        # default values for the amount of hidden neurons
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32]

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

        self.n_samples, self.n_features = X.shape[0], X.shape[1]

        # initialize the model
        self.model = VariationalAutoencoder(
            n_features=self.n_features,
            n_samples=self.n_samples,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            mask_num=self.mask_num,
            hidden_activation=self.hidden_activation)

        # move to device and print model information
        self.model = self.model.to(self.device)
        print(self.model)

        # train the autoencoder to find the best one
        self._train_autoencoder(X, **kwargs)

        print(f"use best model weights on {self.best_epoch}")
        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)

        self._process_decision_scores()
        return self

    @torch.no_grad()
    def generate_cluster_features(self, labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).squeeze()
        return centers



    def _train_autoencoder(self, X, **kwargs):
        """Internal function to train the autoencoder

        Parameters
        ----------
        train_loader : torch dataloader
            Train data.
        """
        # conduct standardization if needed
        fill_mask = kwargs['fill_mask']
        self.fill_mask = fill_mask

        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std, pseudo_y=None, fill_mask=fill_mask)
            test_set = PyODDataset(X=X, mean=self.mean, std=self.std, pseudo_y=None, fill_mask=fill_mask)

        else:
            train_set = PyODDataset(X=X, pseudo_y=None, fill_mask=fill_mask)
            test_set = PyODDataset(X=X, pseudo_y=None, fill_mask=fill_mask)

        train_loader = IterLoader(torch.utils.data.DataLoader(train_set,
                                                              batch_size=self.batch_size,
                                                              shuffle=True, drop_last=True))

        # shuffle must be false, assure correct sample index of each epoch
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay)

        self.best_loss = float('inf')
        self.best_ctr_acc = 0.0
        self.best_acc = 0.0
        self.best_model_dict = None
        history_outlier_info = dict()
        history_outlier_label = np.zeros([self.n_samples, ])

        for epoch in range(self.epochs):

            overall_loss = []
            kl_loss_list = []
            mse_loss_list = []
            div_loss_list = []
            mem_loss_list = []

            train_loader.new_epoch()
            self.model.train()
            # for data, data_idx, batch_y_pseudo_label in train_loader:
            for i in range(len(train_loader)):
                data, data_idx, batch_y_pseudo_label, batch_feature_label, data_fill_mask = train_loader.next()
                data = data.to(self.device).float()

                model_output = self.model(data)
                if "MSE" in self.loss_fn_name:
                    # MSE loss for reconstruction
                    _, mean_mse = self.loss_fn(data, model_output['X_hat'])
                    mse_loss_list.append(mean_mse.item())
                    loss = mean_mse

                if ("mem_loss" in self.loss_fn_name) and (epoch > self.start_pesudo_epoch):
                    batch_y_pseudo_label = batch_y_pseudo_label.to(self.device)
                    memory_res = self.model.memory(
                        model_output['laten_vec'],
                        batch_y_pseudo_label.squeeze().long()
                    )
                    loss += self.mem_loss_weight * memory_res['CE_loss']
                    mem_loss_list.append(memory_res['CE_loss'].item())

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())

            print(f'epoch {epoch}:'
                  f' training loss {np.mean(overall_loss)}; '
                  f' mse loss {np.mean(mse_loss_list)}; '
                  f' kl loss {np.mean(kl_loss_list)}; '
                  f' div loss {np.mean(div_loss_list)}; '
                  f' mem loss {np.mean(mem_loss_list)}; '
                  )

            if "mem_loss" in self.loss_fn_name:
                # collect history outlier data
                print("===========> collect pseudo label <==================")

                self.model.eval()
                with torch.no_grad():
                    features = []
                    # cls_list = []
                    idx_list = []
                    outlier_scores = np.zeros([self.n_samples, ])
                    for data, data_idx, _, _, data_fill_mask in test_loader:
                        data_cuda = data.to(self.device).float()
                        data_gpu_idx = data_idx.to(self.device).long()
                        infer_res = self.model(data_cuda)
                        # cls_list.append(infer_res['class_res'])
                        idx_list.append(data_gpu_idx)
                        features += infer_res['laten_vec'].split(1)
                        outlier_scores[data_idx] = self.loss_fn(
                            data_cuda,
                            infer_res['X_hat']
                        ).cpu().numpy()
                    pseudo_threshold = np.percentile(
                        outlier_scores,
                        100 * (1 - self.contamination)
                    )
                    pseudo_labels = (outlier_scores > pseudo_threshold).astype('int').ravel()
                    history_outlier_label += pseudo_labels

                    # pseudo label via AE
                    for data_idx, data_score in enumerate(outlier_scores.ravel()):
                        if data_score > pseudo_threshold:
                            if data_idx not in history_outlier_info:
                                history_outlier_info[data_idx] = {
                                    "num": 0,
                                    "score_list": [],
                                }
                            history_outlier_info[data_idx]["num"] += 1
                            history_outlier_info[data_idx]["score_list"].append(data_score)
                    # pseudo label via KNN
                    # np_feat = torch.cat(features, dim=0).cpu().numpy()
                    # knn_model = KNN(contamination=self.contamination).fit(np_feat)
                    # knn_score = knn_model.decision_scores_
                    # knn_label = knn_model.labels_
                    # # pseudo label via LOF
                    # lof_model = LOF(contamination=self.contamination).fit(np_feat)
                    # lof_score = lof_model.decision_scores_
                    # lof_label = lof_model.labels_

                    if epoch == self.start_pesudo_epoch:
                        # create memory cluster
                        pseudo_inliner_list = []
                        pseudo_inliner_features = []
                        for _, data_idx, _, _, _ in test_loader:
                            for idx in data_idx.cpu().numpy().tolist():
                                if idx not in history_outlier_info:
                                    pseudo_inliner_list.append(0)
                                else:
                                    pseudo_inliner_list.append(1)
                                pseudo_inliner_features.append(features[idx])
                        # generate new dataset and calculate cluster centers
                        cluster_features = self.generate_cluster_features(pseudo_inliner_list,
                                                                          pseudo_inliner_features)

                        # Create hybrid memory
                        self.model.memory = ClusterMemory(
                            self.hidden_neurons[-1], num_cluster=2,
                            temp=self.mem_temp, momentum=self.mem_momentum, use_hard=self.use_hard
                        ).cuda()
                        self.model.memory.features = torch.nn.functional.normalize(cluster_features, dim=1).cuda()

                        self.best_loss = float('inf')
                        self.best_ctr_acc = 0.0
                        self.best_acc = 0.0
                    if epoch >= self.start_pesudo_epoch:
                        # collect new dataset
                        history_outlier_label[history_outlier_label > 0] = 1
                        pseudo_labeled_dataset = PyODDataset(
                            X=X, mean=self.mean, std=self.std,
                            pseudo_y=np.expand_dims((history_outlier_label), axis=1)
                        )
                        train_loader = IterLoader(torch.utils.data.DataLoader(pseudo_labeled_dataset,
                                                                              batch_size=self.batch_size,
                                                                              shuffle=True))

            # track the best model so far
            if np.mean(overall_loss) <= self.best_loss:
                # print("epoch {ep} is the current best; loss={loss}".format(ep=epoch, loss=np.mean(overall_loss)))
                self.best_loss = np.mean(overall_loss)
                self.best_model_dict = deepcopy(self.model.state_dict())
                self.best_epoch = epoch

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
            dataset = PyODDataset(X=X, mean=self.mean, std=self.std, fill_mask=self.fill_mask)
        else:
            dataset = PyODDataset(X=X)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False, drop_last=False)
        # enable the evaluation mode
        self.model.eval()
        print(self.model.training)
        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])
        outlier_mem_scores = np.zeros([X.shape[0], 2])
        # cls_list= []
        with torch.no_grad():
            for data, data_idx, _, _, data_mask_fill in dataloader:
                data_cuda = data.to(self.device).float()
                model_res = self.model(data_cuda)

                outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                    data, model_res['X_hat'].cpu().numpy())

                # outlier_scores[data_idx] = batch_dist
                # this is the outlier score
                if ("mem_loss" in self.loss_fn_name):
                    outlier_mem_scores[data_idx, :] = self.model.memory(
                        model_res['laten_vec']
                    )['logits'].cpu().numpy()

        return {
            'score': outlier_scores,
            'memo_score': outlier_mem_scores,
            # 'feat_dist': feat_dist,
            # 'outlier_mask': outlier_mask,
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

    def predict(self, X, return_confidence=False, **kwargs):
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
        prediction_AE = (pred_score['score'] > self.threshold_).astype('int').ravel()
        prediction_memo = pred_score['memo_score'].argmax(axis=1)

        if return_confidence:
            confidence = self.predict_confidence(X)
            return prediction_memo, confidence

        return prediction_AE+prediction_memo