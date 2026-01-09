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
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation
from sklearn.metrics import confusion_matrix

from torch import nn
import torch.nn.functional as F

from DetectOutlier.model.baseline.base import BaseDetector
from DetectOutlier.utils.stat_models import pairwise_distances_no_broadcast
from DetectOutlier.utils.torch_utility import get_activation_by_name

from torch.distributions.distribution import Distribution
from torch.distributions import MultivariateNormal, Normal
import random
import matplotlib.pyplot as plt
from DetectOutlier.utils.utility import unique as unique2
from DetectOutlier.utils.utility import epsilon


from DetectOutlier.model.loss.InfoNCE import InfoNCE
from DetectOutlier.model.loss.MemoryLoss import ClusterMemory
import collections

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

    def __init__(self, X, batch_size=16, mean=None, std=None,
                 y_train_label=None, sample_index_id=None,
                 resampling=False, y_true_label=None
                 ):
        super(PyODDataset, self).__init__()

        self.mean = mean
        self.std = std
        if y_train_label is not None:
            y_train_label = np.expand_dims(y_train_label, axis=1)
        else:
            y_train_label = np.expand_dims(np.array([-100]*X.shape[0]), axis=1)

        if sample_index_id is not None:
            sample_index_id = np.expand_dims(sample_index_id, axis=1)
        else:
            sample_index_id = np.expand_dims(np.arange(0, X.shape[0]), axis=1)

        if y_true_label is not None:
            y_true_label = np.expand_dims(y_true_label, axis=1)
        else:
            y_true_label = np.expand_dims(np.array([-100]*X.shape[0]), axis=1)

        if resampling:
            self.X, self.y_train_label, self.sample_index_id, self.y_true_label = self.resampling(
                X, y_train_label, sample_index_id, batch_size, y_true_label
            )
        else:
            self.X = X
            self.y_train_label = y_train_label
            self.sample_index_id = sample_index_id
            self.y_true_label = y_true_label

    def resampling(self, X_train, y_train, sample_index_id, batch_size, y_true_label):
        index_u = np.where(y_train == 0)[0]
        index_a = np.where(y_train == 1)[0]

        n = 0
        while len(index_u) >= batch_size:
            # basic seed
            np.random.seed(n)
            random.seed(n)
            # pytorch seed
            torch.manual_seed(n)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            index_u_batch = np.random.choice(index_u, batch_size - 1, replace=False)
            index_u = np.setdiff1d(index_u, index_u_batch)

            index_a_batch = np.random.choice(index_a, 1, replace=True)

            # batch index
            index_batch = np.append(index_u_batch, index_a_batch)
            # shuffle
            np.random.shuffle(index_batch)

            if n == 0:
                X_train_new = X_train[index_batch]
                y_train_new = y_train[index_batch]
                sample_index_id_new = sample_index_id[index_batch]
                y_true_label_new = y_true_label[index_batch]
            else:
                X_train_new = np.append(X_train_new, X_train[index_batch], axis=0)
                y_train_new = np.append(y_train_new, y_train[index_batch])
                sample_index_id_new = np.append(sample_index_id_new, sample_index_id[index_batch])
                y_true_label_new = np.append(y_true_label_new, y_true_label[index_batch])
            n += 1

        return X_train_new, np.expand_dims(y_train_new, axis=1), \
               np.expand_dims(sample_index_id_new, axis=1), \
               np.expand_dims(y_true_label_new, axis=1)


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

        batch_y_true_label = self.y_true_label[idx, :]
        batch_y_train_label = self.y_train_label[idx, :]
        batch_sample_index_id = self.sample_index_id[idx, :]

        return torch.from_numpy(sample), idx, batch_y_train_label, batch_sample_index_id, batch_y_true_label


# calculate the GaussianKDE with Pytorch
class GaussianKDE(Distribution):  # 已经检验过与sklearn计算结果一致
    def __init__(self, X, bw, lam=1e-4, device=None):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims).to(device),
                                      covariance_matrix=torch.eye(self.dims).to(device))
        self.lam = lam

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.
        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.
        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X

        # 注意此处取log当值接近0时会产生正负无穷的数
        # 利用with autograd.detect_anomaly()检测出算法发散的原因在于torch.log变量值接近0,需要探究接近0的原因
        log_probs = torch.log(
            (self.bw ** (-self.dims) *
             torch.exp(self.mvn.log_prob(
                 (X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n + self.lam)

        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.
        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        X_chunks = self.X.split(1000)
        Y_chunks = Y.split(1000)

        log_prob = 0

        for x in X_chunks:
            for y in Y_chunks:
                log_prob += self.score_samples(y, x).sum(dim=0)

        return log_prob

    def __repr__(self):
        return "GaussianKDE"

def loss_overlap(score, assign_prob, seed, batch_y_label, bw_u=1.0, bw_a=1.0, x_num=1000,
                 plot=False, pro=True, ensemble=False, device=None):
    # set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # remove duplicates
    # s_u = torch.unique(s_u)
    # s_a = torch.unique(s_a)
    u_mask = torch.where(batch_y_label<1)
    a_mask = torch.where(batch_y_label==1)
    s_u = score[u_mask]

    s_a = score[a_mask]
    w_a = assign_prob[a_mask]

    if bw_u is None and bw_a is None:
        d = 1  # one-dimension data
        n_u = s_u.size(0)
        n_a = s_a.size(0)

        # Scott's Rule, which requires the data from the normal distribution.
        # This may be inappropriate when the neural network output can be arbitrary distribution
        # bw_u = n_u ** (-1. / (d + 4))
        # bw_a = n_a ** (-1. / (d + 4))

        # Silverman's Rule
        bw_u = (n_u * (d + 2) / 4.) ** (-1. / (d + 4))
        bw_a = (n_a * (d + 2) / 4.) ** (-1. / (d + 4))

    # reshape
    s_u = s_u.reshape(-1, 1)
    s_a = s_a.reshape(-1, 1)

    # the range of x axis
    xmin = torch.min(torch.min(s_u), torch.min(s_a))
    xmax = torch.max(torch.max(s_u), torch.max(s_a))

    dx = 0.2 * (xmax - xmin)
    xmin -= dx
    xmax += dx
    x = torch.linspace(xmin.detach(), xmax.detach(), x_num).to(device)

    # estimate the Gaussian KDE
    kde_u = GaussianKDE(X=s_u, bw=bw_u, device=device)
    kde_a = GaussianKDE(X=s_a, bw=bw_a, device=device)

    # estimated pdf
    kde_u_x = torch.exp(kde_u.score_samples(x.reshape(-1, 1)))
    kde_a_x = torch.exp(kde_a.score_samples(x.reshape(-1, 1)))

    if plot:
        plt.plot(x, kde_u_x.detach(), color='blue')
        plt.plot(x, kde_a_x.detach(), color='red')
        plt.show()

    if pro:
        # find the intersection point (could be multiple points)
        intersection_points_idx = torch.where(torch.diff(torch.sign(kde_a_x - kde_u_x)))[0].cpu()

        # 修改了求trapz的bug, 之前版本算出来可能会产生negative的area问题
        if intersection_points_idx.size(0) >= 1:
            if ensemble:
                area = 0
                for idx in intersection_points_idx:
                    c = x[idx]
                    idx_u = torch.where(x > c)[0]
                    idx_a = torch.where(x < c)[0]

                    area_u = torch.trapz(kde_u_x[idx_u], x[idx_u])
                    area_a = torch.trapz(kde_a_x[idx_a], x[idx_a])
                    area += (area_u + area_a)

                area = area / intersection_points_idx.size(0)

            else:
                c = x[np.random.choice(intersection_points_idx.numpy(), 1)]
                # print(f'Intersection point: {c}')

                idx_u = torch.where(x > c)[0]
                idx_a = torch.where(x < c)[0]

                area_u = torch.trapz(kde_u_x[idx_u], x[idx_u])
                area_a = torch.trapz(kde_a_x[idx_a], x[idx_a])
                area = area_u + area_a

        else: # no intersection point
            # area = torch.tensor(10.0, requires_grad=True)
            # raise NotImplementedError

            area_u = torch.trapz(kde_u_x, x)
            area_a = torch.trapz(kde_a_x, x)
            area = (area_u + area_a) / 2

            print(f'没有交点! 重叠面积: {area.item()}')

    else:
        inters_x = torch.min(kde_u_x, kde_a_x)
        area = torch.trapz(inters_x, x)

    return {
        "loss": w_a*area,
    }


# InfoNCE: https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py


class AdaptiveModule(nn.Module):
    def __init__(self,
                 n_features
                 ):
        super(AdaptiveModule, self).__init__()
        # set the importance mask
        self.important_mask = torch.nn.Parameter(torch.FloatTensor(1, n_features))
        # init the important mask with
        nn.init.constant_(self.important_mask, 1)
    def forward(self, x):
        important_mask = torch.sigmoid(self.important_mask)
        return important_mask*x

class ScoreMapModule(nn.Module):
    def __init__(self,
                 n_features,
                 latent_vec_neuros,
                 activation
                 ):
        super(ScoreMapModule, self).__init__()
        # self.reg_layer = {}
        self.reg_1 = nn.Sequential(
            nn.Linear(n_features+latent_vec_neuros+1, 256),
            activation
        )

        self.reg_2 = nn.Sequential(
            nn.Linear(256+1, 32),
            activation
        )

        self.reg_3 = nn.Sequential(
            nn.Linear(32+1, 1),
            nn.BatchNorm1d(num_features=1)
        )
    def forward(self, X_hat, x_feat, latent_feat):
        # reconstruction residual vector
        r = torch.sub(X_hat, x_feat)
        # reconstruction error
        e = r.norm(dim=1).reshape(-1, 1)
        # normalized reconstruction residual vector
        r = torch.div(r, e)  # div by broadcast

        hybird_feat = torch.cat((latent_feat, r, e), dim=1)

        # regression
        feature = self.reg_1(hybird_feat)
        feature = self.reg_2(torch.cat((feature, e), dim=1))
        score = self.reg_3(torch.cat((feature, e), dim=1))
        return score

# https://github.com/Minqi824/Overlap/blob/master/baseline/ADSD/model.py
class ADSDAutoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 n_samples,
                 hidden_neurons=(128, 64),
                 dropout_rate=0.0,
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

        # set the adaptive feature
        self.adaptive_format = AdaptiveModule(n_features)

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

        # # set the score distribution learning
        self.score_learner = ScoreMapModule(
            self.n_features,
            self.layers_neurons_encoder_[-1],
            activation=self.activation
        )

    def forward(self, x):
        # we could return the latent representation here after the encoder
        # as the latent representation
        x_feat = self.adaptive_format(x)
        mask_h = self.encoder(x_feat)

        h = self.encoder(x)
        X_hat = self.decoder(h)

        # version 1:
        # important_feature = self.important_mask.repeat(x.size(0), 1) * X_hat
        # version 2:
        return {
            "latent_mask_h": mask_h,
            "latent_h": h,
            "X_hat": X_hat,
        }


class MaskCTRMemoryCE(BaseDetector):
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
                 device=None,
                 cluster_methods='kmeans',
                 eps=0.6, start_pesudo_epoch=100
                 ):
        super(MaskCTRMemoryCE, self).__init__(contamination=contamination)

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
        self.eps = eps
        self.topk_value = topk_value
        self.cluster_methods = cluster_methods
        self.device = device

        # # create default loss functions
        if "MSE" in self.loss_fn_name:
            self.loss_fn = torch.nn.MSELoss()

        if "mask_loss" in self.loss_fn_name:
            self.mask_loss = InfoNCE(temperature=0.01)

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
            y_true_label = kwargs['y_train_label']
        else:
            y_train_label = None

        if "sample_index_id" in kwargs:
            sample_index_id = kwargs["sample_index_id"]
        else:
            sample_index_id = None

        if "sample_name" in kwargs:
            sample_name = kwargs["sample_name"]
        else:
            sample_name = None

        n_samples, n_features = X.shape[0], X.shape[1]

        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            # self.max, self.min = np.max(np.max(X)), np.min(np.min(X))
            test_set = PyODDataset(
                X=X, mean=self.mean, std=self.std,
                sample_index_id=sample_index_id,
                y_train_label=y_true_label,
                y_true_label=y_true_label,
            )
        else:
            test_set = PyODDataset(
                X=X, y_train_label=y_true_label,
                y_true_label=y_true_label,
                sample_index_id=sample_index_id
            )

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
        self._train_autoencoder(
            test_set, y_true_label,
            sample_index_id, X, sample_name
        )

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

        centers = torch.stack(centers, dim=0)
        return centers

    def _train_autoencoder(self, test_set, y_true_label, sample_index_id, X, sample_name):
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
            {'params': mask_para, 'lr_mult': 20},
            {'params': AE_para, 'lr_mult': 1}]

        optimizer = torch.optim.Adam(
            param_groups, lr=self.learning_rate,
            weight_decay=self.weight_decay)

        def adjust_lr():
            for g in optimizer.param_groups:
                g['lr'] = self.learning_rate * g.get('lr_mult', 1)

        self.best_loss = float('inf')
        self.best_ctr_acc = 0.0
        self.best_acc = 0.0
        self.best_model_dict = None
        self.memory = None
        cluster = None

        for epoch in range(self.epochs):
            adjust_lr()
            if epoch >= self.start_pesudo_epoch:
                cluster_loader = torch.utils.data.DataLoader(test_set,
                                                             batch_size=self.batch_size,
                                                             shuffle=False)
                self.model.load_state_dict(self.best_model_dict)

                with torch.no_grad():
                    print("==> Create pseudo labels for unlabeled data")
                    features = []
                    mask_features = []
                    outlier_scores = np.zeros([X.shape[0], ])
                    for data, data_idx, _, _, _ in cluster_loader:
                        data_cuda = data.to(self.device).float()
                        infer_res = self.model(data_cuda)
                        features += infer_res['latent_h'].split(1)
                        mask_features += infer_res['latent_mask_h'].split(1)
                        outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                            data.cpu().numpy(),
                            infer_res['X_hat'].cpu().numpy()
                        )
                    features = torch.cat(features, dim=0)
                    mask_features = torch.cat(mask_features, dim=0)

                np_mask_features = mask_features.cpu().numpy()
                np_features = features.cpu().numpy()

                if cluster is not None:
                    del cluster

                true_outlier_ind = np.where(y_true_label == True)

                # https://stackoverflow.com/questions/46409846/using-k-means-with-cosine-similarity-python
                if self.cluster_methods == 'kmeans':
                    num_cluster = 5
                    length = np.sqrt((np_features ** 2).sum(axis=1))[:, None]
                    np_features = np_features / length

                    cluster = KMeans(n_clusters=num_cluster, random_state=0)
                    # select & cluster images as training set of this epochs
                    pseudo_labels = cluster.fit_predict(np_features)

                    len_ = np.sqrt(np.square(cluster.cluster_centers_).sum(axis=1)[:, None])
                    centers = cluster.cluster_centers_ / len_
                    Point2Center_Dist = 1 - np.dot(centers, np_features.T)  # K x N matrix of cosine distances

                elif self.cluster_methods == 'DBSCAN':
                    # DBSCAN cluster
                    mask_features_norm = mask_features / torch.norm(mask_features, dim=1, keepdim=True)
                    features_norm = features / torch.norm(features, dim=1, keepdim=True)
                    rerank_dist = 1 - torch.mm(mask_features_norm, features_norm.t())
                    eps = float(1-rerank_dist.mean())
                    print('Clustering criterion: eps: {:.3f}'.format(eps))

                    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
                    pseudo_labels = cluster.fit_predict(rerank_dist.cpu().numpy())
                    pseudo_labels += 1

                    cluster_features = self.generate_cluster_features(pseudo_labels, features)
                    # sort keys in cluster_features, outlier feature is in the first line
                    del cluster_loader, features

                    num_cluster = len(set(pseudo_labels))
                    # Create hybrid memory
                    memory = ClusterMemory(self.hidden_neurons[-1], num_cluster=num_cluster, temp=0.05,
                                           momentum=0.05, use_hard=False).cuda()
                    memory.features = F.normalize(cluster_features, dim=1).cuda()

                elif self.cluster_methods == 'AffinityPropagation':
                    cluster = AffinityPropagation(n_clusters=num_cluster, random_state=0)
                    # select & cluster images as training set of this epochs
                    pseudo_labels = cluster.fit_predict(np_features)

                elif self.cluster_methods == 'SelfMaskDist':
                    mask_features_norm = mask_features / torch.norm(mask_features, dim=1, keepdim=True)
                    features_norm = features / torch.norm(features, dim=1, keepdim=True)
                    rerank_dist = 1 - torch.mm(mask_features_norm, features_norm.t())
                    count_num = []
                    for i in range(rerank_dist.size(0)):
                        icount = torch.sum(rerank_dist[i, :]<rerank_dist[i, i])
                        count_num.append(icount)

                    sorted_arr = sorted(enumerate(count_num), key=lambda x: x[1], reverse=True)
                    indices = [i[0] for i in sorted_arr]
                    print(sample_name[indices])

                elif self.cluster_methods == 'SelfDist':
                    pseudo_threshold = np.percentile(
                        outlier_scores,
                        100 * (1 - self.contamination)
                    )
                    pseudo_labels = (outlier_scores > pseudo_threshold).astype('int').ravel()

                    shot_mask = pseudo_labels[true_outlier_ind] == 1
                    unshot_mask = pseudo_labels[true_outlier_ind] == 0
                    print(f"pseudo label accuracy: {sum(shot_mask) / len(true_outlier_ind[0])}")
                    print(f"Epoch {epoch}: Unselect outlier sample {sample_name[true_outlier_ind][unshot_mask]}")

                    pseudo_outlier_ind = np.where(pseudo_labels == 1)
                    wrong_shot_ind = np.setdiff1d(pseudo_outlier_ind, true_outlier_ind)
                    print(f"select wrong outlier sample {sample_name[wrong_shot_ind]}")

                    if epoch == self.start_pesudo_epoch:
                        # generate new dataset and calculate cluster centers
                        cluster_features = self.generate_cluster_features(pseudo_labels, features)

                        # Create hybrid memory
                        self.memory = ClusterMemory(self.hidden_neurons[-1], num_cluster=2, temp=0.01,
                                               momentum=0.7, use_hard=False).cuda()
                        self.memory.features = F.normalize(cluster_features, dim=1).cuda()

                        self.best_loss = float('inf')
                        self.best_ctr_acc = 0.0
                        self.best_acc = 0.0

                    del cluster_loader, features

                # for iclust in range(num_cluster):
                #     cluster_mask = cluster.labels_[true_outlier_ind] == iclust
                #     print(f"For {iclust}:")
                #     print(f"number of cluster is {np.sum(cluster.labels_==iclust)}")
                #
                #     icluster_dist = Point2Center_Dist[:, true_outlier_ind].squeeze()[:, cluster_mask][iclust]
                #     print(f"max dist: {np.max(Point2Center_Dist[iclust])}")
                #     print(f"min dist: {np.min(Point2Center_Dist[iclust])}")
                #     print("Dist of each outlier to own centroids of each cluster:", icluster_dist)
                #
                #     print(f"Selected outlier name: {sample_name[true_outlier_ind][cluster_mask]}")



                # print("No selected outlier:", sample_name[true_outlier_mask][select_mask])
                #
                # pseudo_outlier_mask = pseudo_binary_labels == 1
                # select_mask = y_train_label[pseudo_outlier_mask] != pseudo_binary_labels[pseudo_outlier_mask]
                # print("Wrong selected outlier:", sample_name[pseudo_outlier_mask][select_mask])
                #
                # cluster_matrix = confusion_matrix(
                #     y_train_label, pseudo_binary_labels
                # )
                # print("confusion matrix:\n", cluster_matrix)


                # print("epoch: {} \n pseudo_labels confusion matrix: {}".format(epoch, cluster_matrix))


                pseudo_labeled_dataset = PyODDataset(
                    X=X, mean=self.mean, std=self.std,
                    sample_index_id=sample_index_id,
                    y_train_label=pseudo_labels,
                    resampling=False, batch_size=self.batch_size,
                    y_true_label=y_true_label
                )

                train_loader = IterLoader(torch.utils.data.DataLoader(pseudo_labeled_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=True))

            else:
                pseudo_labeled_dataset = PyODDataset(
                    X=X, mean=self.mean, std=self.std,
                    sample_index_id=sample_index_id,
                    y_true_label=y_true_label
                )
                train_loader = IterLoader(torch.utils.data.DataLoader(pseudo_labeled_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=True))

            train_loader.new_epoch()

            self.train_model(
                len(train_loader), train_loader,
                epoch, optimizer, sample_name
            )


    def train_model(self, train_iters, data_loader, epoch, optimizer, sample_name=None):
        overall_loss = []
        contrastive_acc = []
        correct_acc = []
        for i in range(train_iters):
        # for data, data_idx, batch_y_label, batch_sample_ind in train_loader:
            data, data_idx, batch_y_pseudo_label, batch_sample_ind, batch_y_true_label = data_loader.next()
            data = data.to(self.device).float()
            batch_y_pseudo_label = batch_y_pseudo_label.to(self.device)
            batch_y_true_label = batch_y_true_label.to(self.device)

            model_outputs = self.model(data)

            # reconstruction loss
            if "MSE" in self.loss_fn_name:
                loss = self.loss_fn(data, model_outputs['X_hat'])

            if "mask_loss" in self.loss_fn_name:
                # todo: make sure the self.important_mask is working
                mask_loss_res = self.mask_loss(
                    model_outputs['latent_mask_h'],
                    model_outputs['latent_h'],
                )
                loss += mask_loss_res['loss_v']
                # t1 = batch_y_label[mask_loss_res['recog_mask']]
                # print(f"unable to recognize outlier/inlier: {torch.where(t1==1)[0].size(0)}/{t1.size(0)}")
                contrastive_acc.append(mask_loss_res['accuracy'].item())

                # distribution area loss
            if (epoch >= self.start_pesudo_epoch) and ("score_distribution_loss" in self.loss_fn_name):
                # print(
                #     "for important feature mask:"
                #     f"max weights: {torch.sigmoid(torch.max(self.model.adaptive_format.important_mask))}",
                #     f"min weights: {torch.sigmoid(torch.min(self.model.adaptive_format.important_mask))}"
                # )
                memory_res = self.memory(model_outputs['latent_h'], batch_y_pseudo_label.squeeze().long())
                loss += memory_res['CE_loss']

                refine_label = torch.max(memory_res['logits'], dim=1)[1]

                pseudo_outlier_mask = torch.where(batch_y_pseudo_label.squeeze() == 1)
                if pseudo_outlier_mask[0].size(0) > 0:
                    print(memory_res['logits'][pseudo_outlier_mask[0]])
                    print(sample_name[data_idx.cpu().numpy()][pseudo_outlier_mask[0].cpu().numpy()])

                true_mask = refine_label == batch_y_true_label.squeeze()
                correct_acc.append(sum(true_mask.cpu().numpy())/self.batch_size)


                # 由memory_res 提供新的pseudo label以及对应的概率
                # score = self.model.score_learner(
                #     model_outputs['X_hat'],
                #     data,
                #     model_outputs['latent_h']
                # )

                # overlap_info = loss_overlap(
                #     score, memory_res['assign_weight'],
                #     seed=unique2(self.epochs, epoch),
                #     batch_y_label=batch_pseudo_y_label,
                #     device=self.device
                # )
                # loss += overlap_info['loss']

            self.model.zero_grad()
            loss.backward()
            optimizer.step()
            overall_loss.append(loss.item())

        print(
            'epoch {epoch}: training loss {train_loss}, '
            'contrastive accuracy {ctr_accuracy}, '
            'correct accuracy {corr_acc}'.format(
                epoch=epoch, train_loss=np.mean(overall_loss),
                ctr_accuracy=np.mean(contrastive_acc),
                corr_acc=np.mean(correct_acc)
            )
        )

        # track the best model so far
        # todo: find way to select the best weight
        if (epoch >= self.start_pesudo_epoch) and\
                (np.mean(contrastive_acc) >= self.best_ctr_acc) and\
                (np.mean(overall_loss) <= self.best_loss) and\
                (np.mean(correct_acc) >= self.best_acc):
            self.best_ctr_acc = np.mean(contrastive_acc)
            self.best_loss = np.mean(overall_loss)
            self.best_model_dict = self.model.state_dict()
            self.best_memory_dict = self.memory.state_dict()

        elif (epoch < self.start_pesudo_epoch) and\
                (np.mean(contrastive_acc) >= self.best_ctr_acc) and\
                (np.mean(overall_loss) <= self.best_loss):
            self.best_ctr_acc = np.mean(contrastive_acc)
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
        self.memory.eval()

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])
        memory_logits = np.zeros([X.shape[0], 2])
        with torch.no_grad():
            for data, data_idx, _, _, _ in dataloader:
                data_cuda = data.to(self.device).float()
                infer_res = self.model(data_cuda)
                # this is the outlier score
                outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                    data.cpu().numpy(),
                    # data,
                    infer_res['X_hat'].cpu().numpy()
                )
                memory_logits[data_idx] = self.memory(infer_res['latent_h'])["logits"].cpu().numpy()
                # distribution_score[data_idx] = infer_res['score'].cpu()

        return {
            'score': outlier_scores,
            'memory_logits': memory_logits,
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
        self.threshold_dist = self.decision_scores_['distribution_score']
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

        # check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        prediction = self.labels_

        if return_confidence:
            confidence = self.predict_confidence(X)
            return prediction, confidence

        return prediction