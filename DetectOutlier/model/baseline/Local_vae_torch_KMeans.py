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
from visualization.visulize import plot_cluster_centroid
from sklearn.manifold import TSNE
from torch import nn

from .base import BaseDetector
from copy import deepcopy
from DetectOutlier.utils.stat_models import pairwise_distances_no_broadcast
from DetectOutlier.utils.torch_utility import get_activation_by_name
from torch_kmeans import KMeans


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

class AE_Block(nn.Module):
    def __init__(self, in_channels, out_channels, activation, dropout_rate=None, batch_norm=False):
        super(AE_Block, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.activation = activation

        if batch_norm is not None:
            self.bn_layer = torch.nn.BatchNorm1d(out_channels)
        else:
            self.bn_layer = None

        if dropout_rate is not None:
            self.dropout = torch.nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.linear(x)

        if self.bn_layer is not None:
            out = self.bn_layer(out)

        out = self.activation(out)

        if self.dropout is not None:
            out = self.dropout(out)
        return out


class Conv_Block(nn.Module):
    def __init__(self, Input, Output, kernel_size=9, use_block='standard'):
        super(Conv_Block, self).__init__()
        if use_block == 'standard':
            self.basic_net = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=Input, out_channels=Output, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2),
                torch.nn.BatchNorm1d(Output),
                torch.nn.ReLU(inplace=False),
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
                torch.nn.Conv1d(in_channels=Input, out_channels=Output, kernel_size=1, stride=1),
                torch.nn.BatchNorm1d(Output),
                torch.nn.LeakyReLU(inplace=False),
            )

    def forward(self, x):
        out = self.basic_net(x)
        return out


class ResidualLearner(nn.Module):
    def __init__(self, feat_dim=256, hidden_layer=128):
        super(ResidualLearner, self).__init__()
        self.hidden_layer = hidden_layer
        ######################################## feature extracting ########################################
        self.moduleConv1 = Conv_Block(1, 64)
        self.moduleConv2 = Conv_Block(64, 128)
        self.moduleConv3 = Conv_Block(128, feat_dim)

        self.feat_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.ReLU(inplace=True),
        )
        self.class_header = nn.Linear(hidden_layer, 2)
        self.regression_header = nn.Linear(hidden_layer, 1)


    def forward(self, x):
        batch_size, n_features = x.shape
        x = x.unsqueeze(1)
        out = self.moduleConv1(x)
        out = self.moduleConv2(out)
        out = self.moduleConv3(out)
        sample_feat = self.feat_net(out.view(batch_size*n_features, -1))
        feat_classify = self.class_header(sample_feat)
        error_reg = self.regression_header(sample_feat)
        feat_classify = feat_classify.view(batch_size, n_features, -1)
        error_reg = error_reg.view(batch_size, n_features)
        return out, feat_classify, error_reg


class Simi_Loss(nn.Module):
    def __init__(self, margin=0.1):
        super(Simi_Loss, self).__init__()
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

    def forward(self, points_feat, centroid_feat):
        '''
        :param points_feat: bs, dim, feature_nums
        :param centroid_feat: feature_nums, 2, dim
        :return:
        '''

        points_feat = points_feat.permute(0, 2, 1)

        pos_ = centroid_feat[:, 0, :].unsqueeze(0)
        neg_ = centroid_feat[:, 1, :].unsqueeze(0)

        points_feat_sq = torch.sqrt(torch.sum(torch.pow(points_feat, 2), 2))
        pos_sq = torch.sqrt(torch.sum(torch.pow(pos_, 2), 2))
        neg_sq = torch.sqrt(torch.sum(torch.pow(neg_, 2), 2))

        pos_cos_loss = (points_feat * pos_).sum(dim=-1) / (points_feat_sq * pos_sq)
        neg_cos_loss = (points_feat * neg_).sum(dim=-1) / (points_feat_sq * neg_sq)

        return {
            'pos_simi_loss' : (1 - pos_cos_loss).mean(),
            "neg_simi_loss" : neg_cos_loss.mean()
        }

class VariationalAutoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_neurons=(128, 64),
                 dropout_rate=0.2,
                 latent_dim=2,
                 batch_norm=True,
                 num_parts=2,
                 hidden_activation='relu'):
        # initialize the super class
        super(VariationalAutoencoder, self).__init__()

        # save the default values
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.hidden_activation = hidden_activation
        self.num_parts = num_parts

        # create the dimensions for the input and hidden layers
        self.layers_neurons_encoder_ = [self.n_features, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_encoder_[::-1]

        # get the object for the activations functions
        self.activation = get_activation_by_name(hidden_activation)

        ######################################## global feature learning ########################################
        # fill the encoder sequential with hidden layers
        self.encoder0 = AE_Block(
            self.layers_neurons_encoder_[0], self.layers_neurons_encoder_[1],
            activation=self.activation, dropout_rate=self.dropout_rate, batch_norm=True
        )
        self.encoder1 = AE_Block(
            self.layers_neurons_encoder_[1], self.layers_neurons_encoder_[2],
            activation=self.activation, dropout_rate=self.dropout_rate, batch_norm=True
        )

        self.mu_layer = torch.nn.Linear(
            self.layers_neurons_encoder_[-1],
            latent_dim
        )
        self.sigma_layer = torch.nn.Linear(
            self.layers_neurons_encoder_[-1],
            latent_dim
        )

        # fill the decoder layer
        self.layers_neurons_decoder_ = [latent_dim] + self.layers_neurons_decoder_

        self.decoder0 = AE_Block(
            self.layers_neurons_decoder_[0], self.layers_neurons_decoder_[1],
            activation=self.activation, dropout_rate=self.dropout_rate, batch_norm=True
        )
        self.decoder1 = AE_Block(
            self.layers_neurons_decoder_[1], self.layers_neurons_decoder_[2],
            activation=self.activation, dropout_rate=self.dropout_rate, batch_norm=True
        )
        self.decoder2 = AE_Block(
            self.layers_neurons_decoder_[2], self.layers_neurons_decoder_[3],
            activation=self.activation
        )
        # ######################################## local feature learning ########################################
        self.local_learner = ResidualLearner()

    def forward(self, x):
        # we could return the latent representation here after the encoder
        # as the latent representation
        residual = x
        x = self.encoder0(x)
        mid_x_feat = self.encoder1(x)

        mu = self.mu_layer(mid_x_feat)
        std = torch.exp(0.5 * self.sigma_layer(mid_x_feat))
        eps = torch.randn_like(std).to(std.device)
        z = mu + eps * std

        x = self.decoder0(z)
        x = self.decoder1(x)
        x_feat = self.decoder2(x)

        points_feat, feat_classify, error_reg = self.local_learner(residual)

        return {
            "laten_vec": mid_x_feat,
            "points_feat": points_feat,
            "feat_classify": feat_classify,
            "error_reg": error_reg,
            "X_hat": x_feat,
            "mu": mu,
            "sigma": std,
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
                 num_parts=64,
                 preprocessing=True,
                 loss_fn_name="MSE",
                 start_pesudo_epoch=80,
                 start_cluster_epoch=50,
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
        self.start_cluster_epoch = start_cluster_epoch
        self.num_parts = num_parts

        # self.verbose = verbose
        self.device = device
        self.mem_temp = mem_temp
        self.use_hard = use_hard
        self.mem_momentum = mem_momentum
        self.mem_loss_weight = mem_loss_weight
        self.kl_loss_weight = kl_loss_weight

        # # create default loss functions
        if "MSE" in self.loss_fn_name:
            self.loss_fn = torch.nn.MSELoss()

        if "cluster_loss" in self.loss_fn_name:
            self.simi_loss = Simi_Loss()
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

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
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation)

        # move to device and print model information
        self.model = self.model.to(self.device)
        print(self.model)

        # train the autoencoder to find the best one
        self._train_autoencoder(X, **kwargs)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)

        self._process_decision_scores()
        return self

    def cluster_for_each_sample(self, feats, n_part, init_centroid, batch_size=512):
        N, dim, n_features = feats.shape

        cluster_feats = feats.permute(2, 0, 1) # n_features, N, dim

        feat_pred_label = torch.zeros((n_features, N), device=self.device)
        feature_centroid = torch.zeros((n_features, n_part, dim), device=self.device)

        for ifeat_no in range(n_features // batch_size + 1):
            if ifeat_no == n_features // batch_size:
                start_pos = ifeat_no * batch_size
                end_pos = n_features
            else:
                start_pos = ifeat_no * batch_size
                end_pos = (ifeat_no + 1) * batch_size

            if init_centroid is None:
                model = KMeans(n_clusters=2, verbose=False, normalize=True)
                res = model(cluster_feats[start_pos: end_pos, :, :])
            else:
                model = KMeans(n_clusters=2, verbose=False, normalize=True)
                res = model(cluster_feats[start_pos: end_pos, :, :])
            # cluster_label[start_pos: end_pos, :] = res.labels
            # cluster_centroid[start_pos: end_pos, :, :] = res.centers

            #### 假设 异常的 feature 值更大

            ### 假设 异常的feature 数量少， 1标记为异常，0标记为正常
            num_zeros = (res.labels == 0).sum(dim=1)
            num_ones = (res.labels == 1).sum(dim=1)
            compare_ind = num_zeros > num_ones
            for ind in range(start_pos, end_pos):
                if compare_ind[ind % batch_size]:
                    feat_pred_label[ind, :] = torch.where(
                        res.labels[ind % batch_size] == 0, torch.zeros(N, device=self.device),
                        torch.ones(N, device=self.device)
                    )
                    feature_centroid[ind, 0, :] = res.centers[ind % batch_size, 0, :]
                    feature_centroid[ind, 1, :] = res.centers[ind % batch_size, 1, :]
                else:
                    feat_pred_label[ind, :] = torch.where(
                        res.labels[ind % batch_size] == 1, torch.zeros(N, device=self.device),
                        torch.ones(N, device=self.device)
                    )
                    feature_centroid[ind, 0, :] = res.centers[ind % batch_size, 1, :]
                    feature_centroid[ind, 1, :] = res.centers[ind % batch_size, 0, :]

        return feat_pred_label.permute(1, 0), feature_centroid

    def _train_autoencoder(self, X, **kwargs):
        """Internal function to train the autoencoder

        Parameters
        ----------
        train_loader : torch dataloader
            Train data.
        """
        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std)
            test_set = PyODDataset(X=X, mean=self.mean, std=self.std)

        else:
            train_set = PyODDataset(X=X)
            test_set = PyODDataset(X=X)

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
            if epoch == self.start_cluster_epoch:
                print(f"set the centroid of cluster via random value")
                feature_centroid = None
                print(f"reset the criterion of model saving via resi_loss")
                self.best_resi_loss = float('inf')
                self.best_clus_loss = float('inf')
                self.best_loss = float('inf')

            overall_loss = []
            overall_kl_loss = []
            overall_recon_loss = []
            overall_cluster_loss = []
            overall_resi_loss = []

            train_loader.new_epoch()
            self.model.train()
            # for data, data_idx, batch_y_pseudo_label in train_loader:
            for i in range(len(train_loader)):
                data, data_idx, batch_y_pseudo_label, batch_feature_label = train_loader.next()

                data = data.to(self.device).float()
                model_output = self.model(data)

                # reconstruction loss
                recon_loss = self.loss_fn(data, model_output['X_hat'])
                overall_recon_loss.append(recon_loss.item())

                # kl loss
                kl_loss = (model_output['sigma'] ** 2 + model_output['mu'] ** 2 - torch.log(
                    model_output['sigma']) - 1 / 2).sum()
                overall_kl_loss.append(kl_loss.item())

                # residual value loss
                resi_loss = self.loss_fn(data - model_output['X_hat'], model_output['error_reg'])
                overall_resi_loss.append(resi_loss.item())

                loss = self.kl_loss_weight * kl_loss + recon_loss + resi_loss
                # todo: cluster loss
                if "cluster_loss" in self.loss_fn_name and epoch > self.start_cluster_epoch:
                    clu_loss = self.cross_entropy_loss(
                        model_output['feat_classify'].reshape(-1, 2),
                        batch_feature_label.reshape(-1, 1).type(torch.int64).squeeze()
                    )
                    simi_loss = self.simi_loss(
                        model_output['points_feat'],
                        feature_centroid
                    )
                    overall_cluster_loss.append(clu_loss.item()+ simi_loss['pos_simi_loss'].item())
                    loss = loss + clu_loss + simi_loss['pos_simi_loss']

                self.model.zero_grad()
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.5)
                optimizer.step()
                overall_loss.append(loss.item())

            print(f'epoch {epoch}: training loss {np.mean(overall_loss)}, '
                  f'recon_loss {np.mean(overall_recon_loss)} '
                  f'kl_loss {np.mean(overall_kl_loss)}, '
                  f'cluster_loss {np.mean(overall_cluster_loss)}, '
                  f'resi_loss {np.mean(overall_resi_loss)}, '
                  )

            # track the best model so far
            if "cluster_loss" in self.loss_fn_name and epoch >= self.start_cluster_epoch and np.mean(overall_loss) <= self.best_loss:
                if (np.mean(overall_resi_loss) <= self.best_resi_loss) and (np.mean(overall_cluster_loss) <= self.best_clus_loss):
                    print(f"epoch {epoch} is the current best; resi_loss={np.mean(overall_resi_loss)}; clus_loss={np.mean(overall_cluster_loss)}")
                    self.best_resi_loss = np.mean(overall_resi_loss)
                    self.best_clus_loss = np.mean(overall_cluster_loss)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    self.best_epoch = epoch
            else:
                if np.mean(overall_loss) <= self.best_loss:
                    print("epoch {ep} is the current best; loss={loss}".format(ep=epoch, loss=np.mean(overall_loss)))
                    self.best_loss = np.mean(overall_loss)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    self.best_epoch = epoch

            if "cluster_loss" in self.loss_fn_name and epoch >= self.start_cluster_epoch:
                print(f"starting clustering the feature in high dimensional space at epoch {epoch}")
                # collect history outlier data
                torch.cuda.empty_cache()
                self.model.eval()
                with torch.no_grad():
                    all_cluster_feature = []
                    feat_classify = []
                    for data, data_idx, _, _ in test_loader:
                        data_cuda = data.to(self.device).float()
                        infer_res = self.model(data_cuda)
                        all_cluster_feature.append(infer_res['points_feat'])
                        feat_classify.append(infer_res['feat_classify'])
                    all_cluster_feature = torch.cat(all_cluster_feature, dim=0)
                    all_prob_feature = torch.cat(feat_classify, dim=0)

                    # cluster via kmeans
                    feat_cluster_label, feature_centroid = self.cluster_for_each_sample(
                        all_cluster_feature, n_part=2, init_centroid=feature_centroid
                    )
                    # reset the train loader
                    train_set = PyODDataset(X=X, mean=self.mean, std=self.std, feature_label=feat_cluster_label)
                    train_loader = IterLoader(
                        torch.utils.data.DataLoader(
                            train_set,
                            batch_size=self.batch_size,
                            shuffle=True)
                    )

                    # evaluate the pseudo feature
                    if "feature_label" in kwargs:
                        feature_label = kwargs['feature_label']
                        print(f"evaluate the predicted label at epoch {epoch}")
                        noisy_label_index = np.where(feature_label.values == 1)[0]
                        np_feat_cluster_label = feat_cluster_label.cpu().numpy()
                        np_feat_classify_label = all_prob_feature.sigmoid().argmax(dim=2).cpu().numpy()
                        cluster_mean_shot = []
                        cluster_search_range = []

                        classify_mean_shot = []
                        classify_search_range = []
                        for i in range(self.n_samples):
                            np_feat_cluster_label_index = np.where(np_feat_cluster_label[i] == 1)[0]
                            cluster_matched_feature = np.intersect1d(noisy_label_index, np_feat_cluster_label_index)
                            cluster_mean_shot.append(len(cluster_matched_feature)/len(noisy_label_index))
                            cluster_search_range.append(len(np_feat_cluster_label_index)/self.n_features)

                            np_feat_classify_label_index = np.where(np_feat_classify_label[i] == 1)[0]
                            classify_matched_feature = np.intersect1d(noisy_label_index, np_feat_classify_label_index)
                            classify_mean_shot.append(len(classify_matched_feature) / len(noisy_label_index))
                            classify_search_range.append(len(np_feat_classify_label_index) / self.n_features)
                        print(f"cluster_mean_shot {np.mean(cluster_mean_shot)} and cluster_search_range {np.mean(cluster_search_range)}")
                        print(f"classify_mean_shot {np.mean(classify_mean_shot)} and classify_search_range {np.mean(classify_search_range)}")

                        # if "y_true" in kwargs:
                        #     # reuse model for mapping feature_centroid into 2 dim
                        #     pos_classify_center = self.model.local_learner.class_header(self.model.local_learner.feat_net(feature_centroid[:, 0, :]))
                        #     neg_classify_center = self.model.local_learner.class_header(self.model.local_learner.feat_net(feature_centroid[:, 1, :]))
                        #     plot_cluster_centroid(
                        #         pos_classify_center[noisy_label_index[0]:noisy_label_index[8], :].cpu().numpy(),
                        #         neg_classify_center[noisy_label_index[0]:noisy_label_index[8], :].cpu().numpy(),
                        #         all_prob_feature[:, noisy_label_index[0]:noisy_label_index[8] ,:].permute(1,0,2).cpu().numpy(),
                        #         kwargs['y_true'],
                        #         title=f"E-{epoch}, shot-{np.around(np.mean(classify_mean_shot), 3)} on"
                        #               f" {np.around(np.mean(classify_search_range), 3)}"
                        #     )

                        print("===="*10)
                torch.cuda.empty_cache()

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

        feat_dist = np.zeros([X.shape[0], X.shape[1]])

        self.model.eval()
        with torch.no_grad():
            for data, data_idx, _, _ in dataloader:
                data_cuda = data.to(self.device).float()
                model_res = self.model(data_cuda)

                mask = model_res['feat_classify'].sigmoid().argmax(dim=2)

                batch_dist = np.square(
                    (data_cuda[mask.bool()]).cpu().numpy() - (model_res['X_hat'][mask.bool()]).cpu().numpy()
                )
                # this is the outlier score
                score0 = np.sqrt(np.sum(batch_dist)).ravel()
                outlier_scores[data_idx] = score0
                feat_dist[data_idx, np.argwhere(mask[0].cpu().numpy()==1).flatten()] = batch_dist.squeeze()

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
