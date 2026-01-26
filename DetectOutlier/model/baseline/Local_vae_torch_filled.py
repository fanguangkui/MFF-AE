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
from sklearn.cluster import DBSCAN
import scipy
from torch import nn

from .base import BaseDetector
from copy import deepcopy
from DetectOutlier.utils.torch_utility import get_activation_by_name



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
    def __init__(self, Input, Output, kernel_size=1, use_block='standard'):
        super(Conv_Block, self).__init__()
        if use_block == 'standard':
            self.basic_net = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=Input, out_channels=Output, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2, bias=False),
                # torch.nn.BatchNorm1d(Output),
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
            nn.Linear(hidden_layer, middle_dim, bias=False),
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
        self.mask_feat_embedding_learner = ResidualLearner(middle_dim=self.embedding_dim)
        self.feat_embedding_learner = ResidualLearner(middle_dim=self.embedding_dim)
        self.classifier_header = nn.Linear(self.embedding_dim, 1)
        self.regression_header = nn.Linear(self.embedding_dim, 1, bias=False)

    def forward(self, x, data_fill_mask=None):
        # we could return the latent representation here after the encoder
        # as the latent representation
        batch_size, n_features = x.shape
        residual_x = x
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

        feat_embedding = self.mask_feat_embedding_learner(residual_x*data_fill_mask)
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

    def cluster_for_each_sample(self, data_fill_mask):
        db = DBSCAN(eps=0.2, min_samples=20, metric=scipy.spatial.distance.rogerstanimoto).fit(data_fill_mask)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.

        return labels


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

        # cluster by filled mask
        self.mask_labels = self.cluster_for_each_sample(fill_mask)

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
        # 创建参数组
        # params = []
        # local_params = []
        # 遍历子模块
        # for name, module in self.model.named_children():
        #     print(name)
        #     if name in ['feat_embedding_learner', 'classifier_header', 'regression_header']:  # 全连接层
        #         local_params += list(module.parameters())
        #     else:  # 卷积层
        #         params += list(module.parameters())
        # all_params = [
        #     {'params': params, 'lr': self.learning_rate},
        #     {'params': local_params, 'lr': self.learning_rate*5}
        # ]
        # optimizer = torch.optim.Adam(
        #     all_params, lr=self.learning_rate,
        #     weight_decay=self.weight_decay)
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

                # # residual value loss
                # error_loss = self.loss_fn(
                #     (data - model_output['X_hat']) * data_fill_mask,
                #     model_output['error_reg']* data_fill_mask
                # )
                # overall_error_loss.append(error_loss.item())

                # classify value loss
                mask_loss = self.classify_loss(model_output['feat_classify'], data_fill_mask)
                overall_mask_loss.append(mask_loss.item())
                mask_acc = ((model_output['feat_classify'].sigmoid() > 0.5) == data_fill_mask).sum() / (
                            data_fill_mask.shape[0] * data_fill_mask.shape[1])
                overall_mask_acc.append(mask_acc.item())

                loss = self.kl_loss_weight * kl_loss + recon_loss + 5 * error_loss + mask_loss

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
            # if np.mean(overall_mask_acc) >= self.best_class_acc and np.mean(overall_error_loss) < self.best_loss:
            if np.mean(overall_error_loss) < self.best_loss:
                print(
                    "epoch {ep} is the current best; loss={loss}; acc={acc}".format(
                        ep=epoch, loss=np.mean(overall_error_loss), acc=np.mean(overall_mask_acc)
                    ))
                self.best_loss = np.mean(overall_error_loss)
                self.best_class_acc = np.mean(overall_mask_acc)
                self.best_model_dict = deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                self.centroid = centroid

            # # # # todo: when error_loss 非常小的时候，对features进行聚类
            # if epoch > self.start_cluster_epoch:
            #     print(f"the error loss is {np.mean(overall_error_loss)}")
            #     torch.cuda.empty_cache()
            #     self.model.eval()
            #     with torch.no_grad():
            #         feat_embedding_list = []
            #         data_erro_list = []
            #         data_fill_mask_list = []
            #         for data, data_idx, _, _, data_fill_mask in test_loader:
            #             data_cuda = data.to(self.device).float()
            #             data_fill_mask = data_fill_mask.to(self.device)
            #             infer_res = self.model(data_cuda, data_fill_mask)
            #
            #             feat_embedding_list.append(infer_res['feat_embedding'])
            #             data_fill_mask_list.append(data_fill_mask)
            #             data_erro_list.append(data_fill_mask)
            #
            #         feat_embedding_list = torch.cat(feat_embedding_list, dim=0)
            #         data_fill_mask_list = torch.cat(data_fill_mask_list, dim=0)
            #         centroid = self.cluster_for_each_feat(feat_embedding_list, centroid, data_fill_mask_list)
                # evaluate the pseudo feature
                # if "feature_label" in kwargs:
                #     feature_label = kwargs['feature_label']
                #     y_true = kwargs['y_true']
                #     print(f"evaluate the predicted label at epoch {epoch}")


                        # cluster_matched_feature = np.intersect1d(noisy_label_index, np_feat_cluster_label_index)

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
            dataset = PyODDataset(X=X, mean=self.mean, std=self.std, fill_mask=self.fill_mask)
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
                data_mask_fill = data_mask_fill.to(self.device).float()
                model_res = self.model(data_cuda, data_mask_fill)

                # mask = self.feature_label.values
                # batch_dist = np.square(
                #     ((data_cuda - model_res['X_hat']).cpu().numpy())[np.bool_(mask[np.newaxis, :])]
                # )
                # print(batch_dist.max())
                # idata_weight = get_dist(
                #     model_res['feat_embedding'][data_mask_fill.bool(), :].unsqueeze(1),
                #     self.centroid[data_mask_fill[0].bool(), :].unsqueeze(1)
                # ).squeeze(1).t()
                # weight = idata_weight.sqrt().cpu().numpy()
                # mask_weights = model_res['feat_classify'].sigmoid()
                batch_dist = np.square(
                    # ((data_cuda - model_res['X_hat'])[mask.bool()]).cpu().numpy()
                    ((data_cuda - model_res['X_hat'] - model_res['error_reg']).cpu()).numpy()
                )

                # this is the outlier score
                outlier_scores[data_idx] = np.sqrt(np.sum(batch_dist)).ravel()
                # feat_dist[data_idx] = weight
                # outlier_mask[data_idx] = cos_sim[0].cpu().numpy()

        sample_ids = np.arange(outlier_scores.shape[0])
        for ig in np.unique(self.mask_labels):
            ind = self.mask_labels == ig

            print(sample_ids[ind][np.argsort(-outlier_scores[ind])])

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
