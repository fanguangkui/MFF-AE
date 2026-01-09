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

from .base import BaseDetector
from copy import deepcopy
import torch.nn.functional as F
from DetectOutlier.model.loss.ClusterLoss import KmeansFeature


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

    def __init__(self, X, y=None, mean=None, std=None, pseudo_y=None, feature_label=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std
        self.feature_label = feature_label
        self.pseudo_y = pseudo_y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

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

        return torch.from_numpy(sample), idx, pseudo_label, pseudo_feature


class Conv_Block(nn.Module):
    def __init__(self, Input, Output, kernel_size=9, outputpadding=1, use_block='standard'):
        super(Conv_Block, self).__init__()
        if use_block == 'standard':
            self.basic_net = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=Input, out_channels=Output, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2),
                torch.nn.BatchNorm1d(Output),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv1d(in_channels=Output, out_channels=Output, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2),
                torch.nn.BatchNorm1d(Output),
                torch.nn.ReLU(inplace=False)
            )
        elif use_block == 'simple':
            self.basic_net = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=Input, out_channels=Output, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2),
                torch.nn.BatchNorm1d(Output),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv1d(in_channels=Output, out_channels=Output, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2),
            )
        elif use_block == 'upsample':
            self.basic_net = torch.nn.Sequential(
                torch.nn.ConvTranspose1d(in_channels=Input, out_channels=Output, kernel_size=9, stride=2, padding=4,
                                         output_padding=outputpadding),
                torch.nn.BatchNorm1d(Output),
                torch.nn.ReLU(inplace=False)
            )
        elif use_block == 'gen':
            self.basic_net = torch.nn.Sequential(
                torch.nn.ConvTranspose1d(in_channels=Input, out_channels=Input, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm1d(Input),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv1d(in_channels=Input, out_channels=Input, kernel_size=9, stride=1, padding=4),
                torch.nn.BatchNorm1d(Input),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv1d(in_channels=Input, out_channels=Output, kernel_size=9, stride=1, padding=4),
                torch.nn.Tanh()
            )

    def forward(self, x):
        out = self.basic_net(x)
        return out

class CNNAutoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_neurons=(128, 64),
                 dropout_rate=0.2,
                 latent_dim=2,
                 batch_norm=True,
                 num_parts=2,
                 part_dim=512,
                 learning_mask=False,
                 hidden_activation='relu'):
        # initialize the super class
        super(CNNAutoencoder, self).__init__()

        # save the default values
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.hidden_activation = hidden_activation
        self.num_parts = num_parts
        self.part_dim = part_dim
        self.learning_mask = learning_mask

        # ==================== encoder ====================== #
        self.Encoder1 = Conv_Block(1, 64)
        self.EncoderPool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.Encoder2 = Conv_Block(64, 128)
        self.EncoderPool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.Encoder3 = Conv_Block(128, 256)
        self.EncoderPool3 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.EncoderPool4 = Conv_Block(256, 512, use_block='simple')

        # ==================== decoder ====================== #
        self.Decoder1 = Conv_Block(512, 256)
        self.DecoderUpsample1 = Conv_Block(256, 256, 1, use_block='upsample')

        self.Decoder2 = Conv_Block(256, 128)
        self.DecoderUpsample2 = Conv_Block(128, 128, 1, use_block='upsample')

        self.Decoder3 = Conv_Block(128, 64)
        self.DecoderUpsample3 = Conv_Block(64, 64, 1, use_block='upsample')

        self.Decoder4 = Conv_Block(64, 1, 64, use_block='gen')

        # ==================== feature mask learning ====================== #
        self.MaskUpsample1 = Conv_Block(512, 512, 1, use_block='upsample')
        self.MaskUpsample2 = Conv_Block(512, 512, 1, use_block='upsample')
        self.MaskUpsample3 = Conv_Block(512, 512, 1, use_block='upsample')


    def forward(self, x):
        # we could return the latent representation here after the encoder
        # as the latent representation
        x = x.unsqueeze(1)
        out = self.Encoder1(x)
        out = self.EncoderPool1(out)

        out = self.Encoder2(out)
        out = self.EncoderPool2(out)

        out = self.Encoder3(out)
        out = self.EncoderPool3(out)

        middle_featuer = self.EncoderPool4(out)
        features = F.adaptive_avg_pool1d(out, self.part_dim).squeeze() # for sample clustering

        tensorConv = self.Decoder1(middle_featuer)
        tensorUpsample4 = self.DecoderUpsample1(tensorConv)

        tensorDeconv3 = self.Decoder2(tensorUpsample4)
        tensorUpsample3 = self.DecoderUpsample2(tensorDeconv3)

        tensorDeconv2 = self.Decoder3(tensorUpsample3)
        tensorUpsample2 = self.DecoderUpsample3(tensorDeconv2)

        output = self.Decoder4(tensorUpsample2)

        # ==================== feature cluster ====================== #
        if self.learning_mask:
            featuer_point = self.MaskUpsample1(middle_featuer)
            featuer_point = self.MaskUpsample2(featuer_point)
            featuer_point = self.MaskUpsample3(featuer_point)

            return {
                "laten_vec": middle_featuer,
                'prototype_features': features,
                "X_hat": output.squeeze(1),
                "points_feat": featuer_point,
            }
        else:
            return {
                "laten_vec": middle_featuer,
                'prototype_features': features,
                "X_hat": output.squeeze(1)
            }

class CNNAE(BaseDetector):
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
                 num_parts=64,
                 preprocessing=True,
                 loss_fn_name="MSE",
                 start_pesudo_epoch=80,
                 # verbose=1,
                 # random_state=None,
                 contamination=0.1,
                 mem_temp=0.02,
                 use_hard=False,
                 mem_momentum=0.05,
                 mem_loss_weight=0.05,
                 kl_loss_weight=0.05,
                 device=None):
        super(CNNAE, self).__init__(contamination=contamination)

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
        self.num_parts = num_parts

        # self.verbose = verbose
        self.device = device
        self.mem_temp = mem_temp
        self.use_hard = use_hard
        self.mem_momentum = mem_momentum
        self.mem_loss_weight = mem_loss_weight

        self.learning_mask = False
        # # create default loss functions
        if "MSE" in self.loss_fn_name:
            self.loss_fn = torch.nn.MSELoss()
        if 'cluster_loss' in self.loss_fn_name:
            self.learning_mask = True
            # todo: 定义一个分类损失
            self.crossentropy = nn.CrossEntropyLoss()

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
        self.model = CNNAutoencoder(
            n_features=self.n_features,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation,
            learning_mask=self.learning_mask
        )

        # move to device and print model information
        self.model = self.model.to(self.device)
        print(self.model)

        # train the autoencoder to find the best one
        self._train_autoencoder(X, **kwargs)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)

        self._process_decision_scores()
        return self

    def cluster_for_each_sample(self, feats, n_part, n_featuers):
        N, C, WH = feats.shape
        cluster_feats = feats.reshape(N * WH, C)
        deepcluster = KmeansFeature(k=2, norm=False)

        normal_noise_feats = np.linalg.norm(feats, axis=2)
        normal_noise_feats = normal_noise_feats / np.max(normal_noise_feats, axis=1).reshape(N,1)
        normal_noise_feats = normal_noise_feats.reshape(N * WH, 1)
        clustering_loss = deepcluster.cluster(
            normal_noise_feats, device=0, min_num=int(n_featuers // 100),
            max_num=n_featuers, verbose=False
        )
        # Determining which category is the noise feature
        mean_len = []
        for i in range(deepcluster.k):
            length = []
            for feat_point in deepcluster.featuers_lists[i]:
                length.append(normal_noise_feats[feat_point])
            mean_len.append(np.mean(length))

        # 异常的feature还是少数，大多数还是正常的
        if mean_len[0] < mean_len[1]:
            normal_list = deepcluster.featuers_lists[0]
            normal_list = list(filter(lambda x: normal_noise_feats[x] != 0, normal_list))
            normal_feats = cluster_feats[normal_list]
        else:
            normal_list = deepcluster.featuers_lists[1]
            normal_list = list(filter(lambda x: normal_noise_feats[x] != 0, normal_list))
            normal_feats = cluster_feats[normal_list]
        return clustering_loss

    def _train_autoencoder(self, X, **kwargs):
        """Internal function to train the autoencoder

        Parameters
        ----------
        train_loader : torch dataloader
            Train data.
        """
        # conduct standardization if needed
        pseudo_y = np.expand_dims(np.array([-100] * X.shape[0]), axis=1)
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std, pseudo_y=pseudo_y, **kwargs)
            test_set = PyODDataset(X=X, mean=self.mean, std=self.std, pseudo_y=pseudo_y, **kwargs)

        else:
            train_set = PyODDataset(X=X, pseudo_y=pseudo_y, **kwargs)
            test_set = PyODDataset(X=X, pseudo_y=pseudo_y, **kwargs)

        train_loader = IterLoader(torch.utils.data.DataLoader(train_set,
                                                              batch_size=self.batch_size,
                                                              shuffle=True))

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

        for epoch in range(self.epochs):
            overall_loss = []
            overall_recon_loss = []

            train_loader.new_epoch()
            self.model.train()
            # for data, data_idx, batch_y_pseudo_label in train_loader:
            for i in range(len(train_loader)):
                data, data_idx, batch_y_pseudo_label = train_loader.next()

                data = data.to(self.device).float()
                model_output = self.model(data)

                recon_loss = self.loss_fn(data, model_output['X_hat'])
                overall_recon_loss.append(recon_loss.item())

                loss = recon_loss

                if "cluster_loss" in self.loss_fn_name:
                    # todo: 定义featuer_point的分类损失
                    pass

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())

            print(f'epoch {epoch}: training loss {np.mean(overall_loss)}, '
                  f'recon_loss {np.mean(overall_recon_loss)} '
                  )

            # track the best model so far
            if np.mean(overall_loss) <= self.best_loss:
                print("epoch {ep} is the current best; loss={loss}".format(ep=epoch, loss=np.mean(overall_loss)))
                self.best_loss = np.mean(overall_loss)
                self.best_model_dict = deepcopy(self.model.state_dict())
                self.best_epoch = epoch

            print(f"starting clustering the feature in high dimensional space at epoch {epoch}")
            if "cluster_loss" in self.loss_fn_name:
                # collect history outlier data
                self.model.eval()
                with torch.no_grad():
                    features = []
                    for data, data_idx, _ in test_loader:
                        data_cuda = data.to(self.device).float()
                        infer_res = self.model(data_cuda)
                        features.append(infer_res['points_feat'])
                    all_cluster_featuer = torch.cat(features, dim=0)
                    self.cluster_for_each_sample(
                        all_cluster_featuer.cpu().numpy(), n_part=2, n_featuers=self.n_features
                    )

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
                                                 batch_size=1,
                                                 shuffle=False)
        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])
        outlier_mem_scores = np.zeros([X.shape[0], 2])

        feat_dist = np.zeros([X.shape[0], X.shape[1]])

        self.model.eval()
        with torch.no_grad():
            for data, data_idx in dataloader:
                data_cuda = data.to(self.device).float()
                model_res = self.model(data_cuda)
                batch_dist = np.square(
                    (data_cuda).cpu().numpy() - (model_res['X_hat']).cpu().numpy()
                )
                # this is the outlier score
                score0 = np.sqrt(np.sum(batch_dist, axis=1)).ravel()
                outlier_scores[data_idx] = score0
                feat_dist[data_idx, :] = batch_dist

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

        if return_confidence:
            confidence = self.predict_confidence(X)
            return prediction_AE, confidence

        return prediction_AE
