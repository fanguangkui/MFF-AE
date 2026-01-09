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
from sklearn.cluster import DBSCAN, AffinityPropagation
from collections import Counter
from torch import nn

from .base import BaseDetector
from copy import deepcopy
from DetectOutlier.utils.torch_utility import get_activation_by_name
from DetectOutlier.utils.utility import get_dist
from torch_kmeans import SoftKMeans, KMeans


def initialize_weights(m):
    # Conv1d init
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    # Linear init
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    # BatchNorm1d init
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def forward(self, x):
    out = self.basic_net(x)
    return out

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
        self.basic_net.apply(initialize_weights)

    def forward(self, x):
        out = self.basic_net(x)
        return out


class ResidualLearner(nn.Module):
    def __init__(self, middle_dim=256, hidden_layer=256):
        super(ResidualLearner, self).__init__()
        self.hidden_layer = hidden_layer
        ######################################## feature extracting ########################################
        self.moduleConv1 = Conv_Block(1, 64)
        self.moduleConv2 = Conv_Block(64, 128)
        self.moduleConv3 = Conv_Block(128, hidden_layer)

        self.feat_net = nn.Sequential(
            nn.Linear(hidden_layer, middle_dim),
            # nn.BatchNorm1d(middle_dim),
            nn.Tanh(),
        )


    def forward(self, x_hat):
        batch_size, n_features = x_hat.shape
        x_hat = x_hat.unsqueeze(1)
        out = self.moduleConv1(x_hat)
        out = self.moduleConv2(out)
        out = self.moduleConv3(out)
        feat_embedding = self.feat_net(out.view(batch_size * n_features, -1))

        return feat_embedding


class VariationalAutoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_neurons=(128, 64),
                 dropout_rate=0.2,
                 latent_dim=2,
                 batch_norm=True,
                 num_parts=4,
                 embedding_dim=128,
                 hidden_activation='relu'):
        # initialize the super class
        super(VariationalAutoencoder, self).__init__()

        # save the default values
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.hidden_activation = hidden_activation
        self.num_parts = num_parts
        self.embedding_dim = embedding_dim

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
        self.feat_embedding_learner = ResidualLearner(middle_dim=self.embedding_dim)
        self.classifier_header = nn.Linear(self.embedding_dim, 1)
        self.regression_header = nn.Linear(self.embedding_dim, 1)

    def forward(self, x, data_fill_mask=None):
        # we could return the latent representation here after the encoder
        # as the latent representation
        batch_size, n_features = x.shape

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

        # if self.training:
        unmask_feat_embedding = self.feat_embedding_learner(x_feat)
        feat_classify = self.classifier_header(unmask_feat_embedding)

        feat_embedding = self.feat_embedding_learner(x_feat*data_fill_mask)
        error_reg = self.regression_header(feat_embedding)

        return {
            "laten_vec": mid_x_feat,
            "feat_embedding": feat_embedding.view(batch_size, n_features, -1),
            "feat_classify": feat_classify.view(batch_size, n_features),
            "error_reg": error_reg.view(batch_size, n_features),
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
                 num_parts=4,
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
                 embedding_dim=128,
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
        self.embedding_dim = embedding_dim

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
            self.l1loss_fn = torch.nn.L1Loss()

        if "cluster_loss" in self.loss_fn_name:
            self.classify_loss = torch.nn.BCEWithLogitsLoss()

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
            hidden_activation=self.hidden_activation,
            num_parts=self.num_parts,
        )

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

    def cluster_for_each_feat(self, feats, old_center, data_fill_mask_list, n_part=2, batch_size=256, method='kmeans', threshold=0.7):
        N = feats.size(0)
        cluster_feats = feats.permute(1, 0, 2)  # n_features, N, dim
        # 选择未被填充的feature
        # calculate the similarity between samples of each feature: feat_num, n_samples, n_samples

        # todo 使用异常检测算法，给出异常分数，作为这个feature在后续AE重建过程中的soft weights
        # todo 使用聚类算法，计算高质量的中心表示，作为这个feature在后续AE重建过程中的soft weights
        sample_similarity = 1 - get_dist(cluster_feats, cluster_feats, metric='cos')

        centroid_feat = torch.zeros((self.n_features, self.embedding_dim), device=self.device)
        # 预先计算共享的三角掩码（避免在循环中重复创建）
        mask = torch.triu(torch.ones(N, N, device=self.device), diagonal=1).bool()

        for ifeat_no in range(self.n_features):
            # 向量化条件判断
            sim_matrix = sample_similarity[ifeat_no]
            cond_matrix = (sim_matrix > threshold) & mask

            # 直接获取满足条件的行列索引
            rows, cols = torch.where(cond_matrix)

            if rows.numel() == 0:
                continue  # 无满足条件的元素时跳过

            # 合并行列索引并统计出现频次（使用GPU加速）
            all_indices = torch.cat([rows, cols])
            unique_indices, counts = torch.unique(all_indices, return_counts=True)

            # 向量化权重计算和特征聚合
            weights = counts.float()
            normalized_weights = weights / weights.sum()
            centroid_feat[ifeat_no] = torch.einsum('n,nd->d',
                                                   normalized_weights,
                                                   cluster_feats[ifeat_no, unique_indices])

        # for ifeat_no in range(n_features):
        #     mask = torch.triu(torch.ones_like(sample_similarity[ifeat_no]), diagonal=1).bool()
        #     indices = torch.where(mask & (sample_similarity[ifeat_no] > threshold))
        #     rows = indices[0].tolist()
        #     cols = indices[1].tolist()
        #     total_counts = Counter(rows + cols)
        #     if len(total_counts) > 0:
        #         weights = torch.from_numpy(np.array(list(total_counts.values()))).to(self.device)
        #         normalized_weights = weights / weights.sum()
        #         centroid_feat[ifeat_no, :] = torch.einsum('n,nm->m', normalized_weights, cluster_feats[ifeat_no, list(total_counts.keys()), :])
        if old_center is None:
            return centroid_feat
        else:
            offset_dist = get_dist(centroid_feat.unsqueeze(0), old_center.unsqueeze(0))
            print(f"average offset of centroid {offset_dist.sqrt().mean().item()}")
            min_values, min_indices = offset_dist.squeeze(0).mean(dim=1).min(dim=0)
            max_values, max_indices = offset_dist.squeeze(0).mean(dim=1).max(dim=0)
            print(f"max offset of {max_indices} is {max_values}")
            print(f"min offset of {min_indices} is {min_values}")
            centroid_feat = (centroid_feat + old_center) / 2
            return centroid_feat
        # feat_level_cluster_feats = sample_similarity.mean(dim=2)  # # n_features, N

        ### refine pseudo label
        # feat_sampel_pred_label = np.zeros((n_features, N))
        # for ifeat_no in range(n_features // batch_size + 1):
        #     if ifeat_no == n_features // batch_size:
        #         start_pos = ifeat_no * batch_size
        #         end_pos = n_features
        #     else:
        #         start_pos = ifeat_no * batch_size
        #         end_pos = (ifeat_no + 1) * batch_size
        #     feat_bin_model = KMeans(n_clusters=2, verbose=False, normalize=True)
        #     sample_level_cluster_feats = sample_similarity[start_pos: end_pos, :, :]
        #     res = feat_bin_model(sample_level_cluster_feats)
        #     cluster_cor0 = (feat_level_cluster_feats[start_pos: end_pos, :]*(res.labels == 0)).sum(dim=1).cpu().numpy()
        #     cluster_cor1 = (feat_level_cluster_feats[start_pos: end_pos, :]*(res.labels == 1)).sum(dim=1).cpu().numpy()
        #     compare_ind = cluster_cor0 > cluster_cor1
        #     for ind in range(start_pos, end_pos):
        #         if compare_ind[ind % batch_size]:
        #             feat_sampel_pred_label[ind, :] = torch.where(
        #                 res.labels[ind % batch_size] == 0, torch.zeros(N, device=self.device),
        #                 torch.ones(N, device=self.device)
        #             )
        #         else:
        #             feat_sampel_pred_label[ind, :] = torch.where(
        #                 res.labels[ind % batch_size] == 1, torch.zeros(N, device=self.device),
        #                 torch.ones(N, device=self.device)
        #             )

        ### 假设 异常的feature相关度更小， 1标记为异常，0标记为正常
        # if method == 'kmeans':
        #     model = KMeans(n_clusters=2, verbose=False, normalize="unit")
        #     res = model(feat_level_cluster_feats.unsqueeze(0))
        #     cluster_label = res.labels[0].cpu().numpy()
        #
        #     cluster_cor0 = (feat_level_cluster_feats[res.labels[0] == 0, :]).mean().cpu().numpy()
        #     cluster_cor1 = (feat_level_cluster_feats[res.labels[0] == 1, :]).mean().cpu().numpy()
        #
        #     feat_pred_label = np.zeros(n_features)
        #     # feat_index = np.arange(n_features)
        #     ### 假设 异常的feature相关度更小， 1标记为异常
        #     if cluster_cor0 > cluster_cor1:
        #         feat_pred_label[cluster_label == 1] = 1
        #         feat_pred_label[cluster_label == 0] = 0
        #         pseudo_noisy_feat = feat_level_cluster_feats[cluster_label == 1, :]
        #         # pseudo_noisy_indx = feat_index[cluster_label == 1]
        #     else:
        #         feat_pred_label[cluster_label == 0] = 1
        #         feat_pred_label[cluster_label == 1] = 0
        #         pseudo_noisy_feat = feat_level_cluster_feats[cluster_label == 0, :]
        #         # pseudo_noisy_indx = feat_index[cluster_label == 0]
        #
        #     print(pseudo_noisy_feat.shape)
        #     # ### 假设 异常的feature相关度更小， 1标记为异常最严重
        #     # model = KMeans(n_clusters=n_part-1, verbose=False, normalize="unit")
        #     # res = model(pseudo_noisy_feat.unsqueeze(0))
        #     # cluster_label = res.labels[0].cpu().numpy()
        #     # cluster_cor = []
        #     # for i in range(n_part-1):
        #     #     cluster_cor.append(
        #     #         (pseudo_noisy_feat[cluster_label == i, :]).mean().cpu().numpy()
        #     #     )
        #     # cluster2label = np.argsort(cluster_cor)
        #     # for i in range(n_part-1):
        #     #     pseudo_index = pseudo_noisy_indx[cluster_label == cluster2label[i]]
        #     #     feat_pred_label[pseudo_index] = i + 1
        #
        #     # get centroid of each feature
        #     centroid_feat = torch.zeros((2, n_features, dim), device=self.device)
        #     pos_sample_stat = (sample_similarity > sample_similarity.mean(dim=2, keepdim=True)).sum(dim=2)
        #     pos_sample = torch.topk(pos_sample_stat, k=N//5)
        #     expanded_indices = pos_sample[1].unsqueeze(-1).expand(-1, -1, dim)
        #     centroid_feat[0, :, :] = torch.gather(cluster_feats, dim=1, index=expanded_indices).mean(dim=1)
        #
        #     neg_sample_stat = (sample_similarity < sample_similarity.mean(dim=2, keepdim=True)).sum(dim=2)
        #     neg_sample = torch.topk(neg_sample_stat, k=N//5)
        #     expanded_indices = neg_sample[1].unsqueeze(-1).expand(-1, -1, dim)
        #     centroid_feat[1, :, :] = torch.gather(cluster_feats, dim=1, index=expanded_indices).mean(dim=1)
        #
        #     return feat_pred_label, centroid_feat

    # def cluster_for_each_sample(self, feats, n_part, batch_size=512, method='kmeans'):
    #     N, n_features, dim = feats.shape
    #     # calculate the similarity between samples of each feature: feat_num, n_samples, n_samples
    #     cluster_feats = feats.permute(1, 0, 2)  # n_features, N, dim
    #     sample_similarity = 1 - get_dist(cluster_feats, cluster_feats, metric='cos')
    #

    #
    #         res = model(sample_similarity[start_pos: end_pos, :, :])

    # feat_v = feats[:, ind].t()
    # # print(ind)
    # if (feat_v[res.labels[ind% batch_size, :] == 0]).abs().mean() > (feat_v[res.labels[ind% batch_size, :] == 1]).abs().mean():
    #
    # else:
    #     feat_pred_label[ind, :] = torch.where(
    #         res.labels[ind% batch_size, :] == 1,
    #         torch.ones(N, device=self.device),
    #         torch.zeros(N, device=self.device)
    #     )
    #     feature_centroid[ind, 0] = res.centers[ind % batch_size, 0]
    #     feature_centroid[ind, 1] = res.centers[ind % batch_size, 1]
    #
    #
    #
    # for ifeat_no in range(n_features // batch_size + 1):
    #     if ifeat_no == n_features // batch_size:
    #         start_pos = ifeat_no * batch_size
    #         end_pos = n_features
    #     else:
    #         start_pos = ifeat_no * batch_size
    #         end_pos = (ifeat_no + 1) * batch_size
    #
    #     if method == 'kmeans':
    #         model = SoftKMeans(n_clusters=2, verbose=False, normalize=True)
    #         res = model(cluster_feats[start_pos: end_pos, :, :])
    #     # elif method == 'dbscan':
    #
    #     # cluster_label[start_pos: end_pos, :] = res.labels
    #     # cluster_centroid[start_pos: end_pos, :, :] = res.centers
    #
    #
    #
    #     ### 假设 异常的feature 数量少， 1标记为异常，0标记为正常
    #     num_zeros = (res.labels == 0).sum(dim=1)
    #     num_ones = (res.labels == 1).sum(dim=1)
    #     compare_ind = num_zeros > num_ones
    #     for ind in range(start_pos, end_pos):
    #         if compare_ind[ind % batch_size]:
    #             feat_pred_label[ind, :] = torch.where(
    #                 res.labels[ind % batch_size] == 0, torch.zeros(N, device=self.device),
    #                 torch.ones(N, device=self.device)
    #             )
    #             feature_centroid[ind, 0, :] = res.centers[ind % batch_size, 0, :]
    #             feature_centroid[ind, 1, :] = res.centers[ind % batch_size, 1, :]
    #         else:
    #             feat_pred_label[ind, :] = torch.where(
    #                 res.labels[ind % batch_size] == 1, torch.zeros(N, device=self.device),
    #                 torch.ones(N, device=self.device)
    #             )
    #             feature_centroid[ind, 0, :] = res.centers[ind % batch_size, 1, :]
    #             feature_centroid[ind, 1, :] = res.centers[ind % batch_size, 0, :]

    # return feat_pred_label.permute(1, 0), feature_centroid

    def _train_autoencoder(self, X, **kwargs):
        """Internal function to train the autoencoder

        Parameters
        ----------
        train_loader : torch dataloader
            Train data.
        """
        # conduct standardization if needed
        fill_mask = kwargs['fill_mask']
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std, fill_mask=fill_mask)
            test_set = PyODDataset(X=X, mean=self.mean, std=self.std, fill_mask=fill_mask)

        else:
            train_set = PyODDataset(X=X, fill_mask=fill_mask)
            test_set = PyODDataset(X=X, fill_mask=fill_mask)

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
        self.best_class_acc = 0.0
        self.best_model_dict = None
        centroid = None

        for epoch in range(self.epochs):

            overall_loss = []
            overall_kl_loss = []
            overall_recon_loss = []
            overall_error_loss = []
            overall_mask_loss = []
            overall_mask_acc = []

            train_loader.new_epoch()
            self.model.train()
            # for data, data_idx, batch_y_pseudo_label in train_loader:
            for i in range(len(train_loader)):
                data, data_idx, batch_y_pseudo_label, batch_feature_label, data_fill_mask = train_loader.next()
                data = data.to(self.device).float()
                data_fill_mask = data_fill_mask.to(self.device).float()

                model_output = self.model(data, data_fill_mask)

                # reconstruction loss
                recon_loss = self.loss_fn(data, model_output['X_hat'])
                overall_recon_loss.append(recon_loss.item())

                # kl loss
                kl_loss = (model_output['sigma'] ** 2 + model_output['mu'] ** 2 - torch.log(
                    model_output['sigma']) - 1 / 2).sum()
                overall_kl_loss.append(kl_loss.item())

                # residual value loss
                error_loss = self.loss_fn(
                    (data - model_output['X_hat']) * data_fill_mask,
                    model_output['error_reg']* data_fill_mask
                )
                overall_error_loss.append(error_loss.item())

                # classify value loss
                mask_loss = self.classify_loss(model_output['feat_classify'], data_fill_mask)
                overall_mask_loss.append(mask_loss.item())
                mask_acc = ((model_output['feat_classify'].sigmoid() > 0.5) == data_fill_mask).sum() / (
                            data_fill_mask.shape[0] * data_fill_mask.shape[1])
                overall_mask_acc.append(mask_acc.item())

                loss = self.kl_loss_weight * kl_loss + recon_loss + error_loss + mask_loss

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())

            print(
                f'epoch {epoch}: training loss {np.mean(overall_loss)}, '
                f'recon_loss {np.mean(overall_recon_loss)} '
                f'error_loss {np.mean(overall_error_loss)}, '
                f'kl_loss {np.mean(overall_kl_loss)}, '
                f'mask_loss {np.mean(overall_mask_loss)}, '
                f'mask_acc {np.mean(overall_mask_acc)}, '
            )

            # track the best model so far
            if np.mean(overall_mask_acc) >= self.best_class_acc and np.mean(overall_error_loss) < self.best_loss:
                print(
                    "epoch {ep} is the current best; loss={loss}; acc={acc}".format(
                        ep=epoch, loss=np.mean(overall_loss), acc=np.mean(overall_mask_acc)
                    ))
                self.best_loss = np.mean(overall_error_loss)
                self.best_class_acc = np.mean(overall_mask_acc)
                self.best_model_dict = deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                self.centroid = centroid

            # # cluster feature
            torch.cuda.empty_cache()
            self.model.eval()
            with torch.no_grad():
                feat_embedding_list = []
                data_fill_mask_list = []
                for data, data_idx, _, _, data_fill_mask in test_loader:
                    data_cuda = data.to(self.device).float()
                    data_fill_mask = data_fill_mask.to(self.device)
                    infer_res = self.model(data_cuda, data_fill_mask)

                    feat_embedding_list.append(infer_res['feat_embedding'])
                    data_fill_mask_list.append(data_fill_mask)

                feat_embedding_list = torch.cat(feat_embedding_list, dim=0)
                data_fill_mask_list = torch.cat(data_fill_mask_list, dim=0)
                centroid = self.cluster_for_each_feat(feat_embedding_list, centroid, data_fill_mask_list)
                    # mask_value = torch.sigmoid(infer_res['feat_classify'])
                    # feat_classify.append(mask_value)
                    # data_fill_mask_list.append(data_fill_mask)
                    # if self.centroid is not None:
                    #     cos_sim = torch.nn.functional.cosine_similarity(infer_res['feat_embedding'], self.centroid,
                    #                                                     dim=-1)
                    #     print(cos_sim.cpu().numpy()[0, 3788: 3863].mean())
                    #     print(cos_sim.cpu().numpy()[0, 3700: 3787].mean())
                    #     print(np.argsort(cos_sim.cpu().numpy()[0]))
                    #     print(cos_sim.cpu().numpy()[0, np.argsort(cos_sim.cpu().numpy()[0])])
                #
                # data_fill_mask_list = torch.cat(data_fill_mask_list, dim=0)
                # feat_classify_correct = ((all_feat_classify > 0.5) == data_fill_mask_list).sum() / (
                #             data_fill_mask_list.shape[0] * data_fill_mask_list.shape[1])
                # print(f"The final accuracy of all feature is {feat_classify_correct.item()}")
                # evaluate the pseudo feature
                # if "feature_label" in kwargs:
                #     feature_label = kwargs['feature_label']
                #     self.artificial_mask = fill_mask.sum(axis=0) >= 50
                #
                #     print(f"evaluate the predicted label at epoch {epoch}")
                #     noisy_label_index = np.where(feature_label.values == 1)[0]
                #     #     # normal_label_index = np.where(feature_label.values == 0)[0]
                #     #     self.feature_label = feature_label
                #     #
                #     #     cluster_mean_shot = []
                #     #     cluster_search_range = []
                #     #
                #     #     # classify_mean_shot = []
                #     #     # classify_search_range = []
                #     if len(self.artificial_mask.shape) == 1:
                #         np_feat_cluster_label_index = np.where(self.artificial_mask > 0)[0]
                #         cluster_matched_feature = np.intersect1d(noisy_label_index, np_feat_cluster_label_index)
                #
                #         print(
                #             f"cluster_mean_shot {len(cluster_matched_feature) / len(noisy_label_index)}"
                #             f" and cluster_search_range {len(np_feat_cluster_label_index) / self.n_features}")
                #     # else:
                #     #     for i in range(self.n_samples):
                #     #         np_feat_cluster_label_index = np.where(feat_cluster_label[i] == 0)
                #     #         cluster_matched_feature = np.intersect1d(noisy_label_index, np_feat_cluster_label_index)
                #     #         cluster_mean_shot.append(len(cluster_matched_feature) / len(noisy_label_index))
                #     #         cluster_search_range.append(len(np_feat_cluster_label_index) / self.n_features)
                #         # #
                #         # np_feat_classify_label_index = np.where(np_feat_classify_label[i] == 1)[0]
                #         # classify_matched_feature = np.intersect1d(noisy_label_index, np_feat_classify_label_index)
                #         # classify_mean_shot.append(len(classify_matched_feature) / len(noisy_label_index))
                #         # classify_search_range.append(len(np_feat_classify_label_index) / self.n_features)
                # print(
                #     f"classify_mean_shot {np.mean(classify_mean_shot)} and classify_search_range {np.mean(classify_search_range)}")
                #
                #     # if "y_true" in kwargs:
                #     #     plot_feat_num = 49
                #     #     # reuse model for mapping feature_centroid into 2 dim
                #     #     pos_classify_center = feature_centroid[0, :, :]
                #     #     neg_classify_center = feature_centroid[1, :, :]
                #     #     plot_cluster_centroid(
                #     #         pos_classify_center[noisy_label_index[0: plot_feat_num], :].cpu().numpy(),
                #     #         neg_classify_center[noisy_label_index[0: plot_feat_num], :].cpu().numpy(),
                #     #         all_cluster_feature[:, noisy_label_index[0:plot_feat_num], :].permute(1, 0,
                #     #                                                                               2).cpu().numpy(),
                #     #         kwargs['y_true'],
                #     #         title=f"E-{epoch}",
                #     #         # title=f"E-{epoch}, shot-{np.around(np.mean(classify_mean_shot), 3)} on"
                #     #         #       f" {np.around(np.mean(classify_search_range), 3)}",
                #     #         plot_feat_num=plot_feat_num
                #     #     )
                # torch.cuda.empty_cache()
            print("====" * 10)

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
        outlier_mask = np.zeros([X.shape[0], X.shape[1]])

        feat_dist = np.zeros([X.shape[0], X.shape[1]])

        self.model.eval()
        with torch.no_grad():
            for data, data_idx, _, _, data_mask_fill in dataloader:
                data_cuda = data.to(self.device).float()
                data_mask_fill = data_mask_fill.to(self.device)
                model_res = self.model(data_cuda)

                # mask = self.feature_label.values
                # batch_dist = np.square(
                #     ((data_cuda - model_res['X_hat']).cpu().numpy())[np.bool_(mask[np.newaxis, :])]
                # )
                # print(batch_dist.max())

                # mask_weights = model_res['feat_classify'].sigmoid()
                batch_dist = np.square(
                    # ((data_cuda - model_res['X_hat'])[mask.bool()]).cpu().numpy()
                    (((data_cuda - model_res['X_hat']) * data_mask_fill)).cpu().numpy()
                )

                # this is the outlier score
                score0 = np.sqrt(np.sum(batch_dist)).ravel()
                outlier_scores[data_idx] = score0
                feat_dist[data_idx] = batch_dist
                # outlier_mask[data_idx] = cos_sim[0].cpu().numpy()

        return {
            'score': outlier_scores,
            'feat_dist': feat_dist,
            'outlier_mask': outlier_mask,
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
