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
from torch.nn import init
from torch import nn
import torch.nn.functional as F

from DetectOutlier.model.baseline.base import BaseDetector
from DetectOutlier.utils.stat_models import pairwise_distances_no_broadcast
from DetectOutlier.utils.torch_utility import get_activation_by_name

import random


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

# InfoNCE: https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py


class AdaptiveModule(nn.Module):
    def __init__(self,
                 d_model, S=64
                 ):
        super(AdaptiveModule, self).__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=1,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model
        return out

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

        # set memory module:
        self.memory = None

        # set the importance mask
        self.pesudo_record = torch.nn.Parameter(torch.FloatTensor(n_samples, ), requires_grad=False)
        # init the important mask with
        nn.init.constant_(self.pesudo_record, 0)

    def forward(self, x):
        # we could return the latent representation here after the encoder
        # as the latent representation
        x_feat = self.adaptive_format(x)
        mask_h = self.encoder(x_feat)

        h = self.encoder(x)
        X_hat = self.decoder(h)

        dist_score = self.score_learner(X_hat, x, h)
        # version 1:
        # important_feature = self.important_mask.repeat(x.size(0), 1) * X_hat
        # version 2:
        return {
            "latent_mask_h": mask_h,
            "latent_h": h,
            "X_hat": X_hat,
            "dist_score": dist_score,
        }


class MUSEAttenCTRMemoryCE(BaseDetector):
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
        super(MUSEAttenCTRMemoryCE, self).__init__(contamination=contamination)

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

        n_samples, n_features = X.shape[0], X.shape[1]

        # conduct standardization if needed
        if self.preprocessing:
            mean, std = np.mean(X, axis=0), np.std(X, axis=0)
            # self.max, self.min = np.max(np.max(X)), np.min(np.min(X))
            test_set = PyODDataset(
                X=X, mean=mean, std=std
            )
        else:
            test_set = PyODDataset(
                X=X, y_train_label=y_true_label
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
            test_set, X
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

    def _train_autoencoder(self, test_set, X):
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

        for epoch in range(self.epochs):
            adjust_lr()

            cluster_loader = torch.utils.data.DataLoader(test_set,
                                                         batch_size=self.batch_size,
                                                         shuffle=False)
            # self.model.load_state_dict(self.best_model_dict, strict=False)

            with torch.no_grad():
                print("==> Create pseudo labels for unlabeled data")
                features = []
                outlier_scores = np.zeros([X.shape[0], ])
                for data, data_idx, _, _, _ in cluster_loader:
                    data_cuda = data.to(self.device).float()
                    infer_res = self.model(data_cuda)
                    features += infer_res['latent_h'].split(1)
                    outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                        data.cpu().numpy(),
                        infer_res['X_hat'].cpu().numpy()
                    )
            pseudo_threshold = np.percentile(
                outlier_scores,
                100 * (1 - self.contamination)
            )
            pseudo_labels = (outlier_scores > pseudo_threshold).astype('int').ravel()
            self.model.pesudo_record += torch.from_numpy(pseudo_labels).to(self.device)

            if epoch >= self.start_pesudo_epoch:
                pass
            else:
                if self.preprocessing:
                    # mean, std = np.mean(X[pseudo_inlier_mask], axis=0), np.std(X[pseudo_inlier_mask], axis=0)
                    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
                    pseudo_labeled_dataset = PyODDataset(
                        X=X, mean=mean, std=std,
                        y_true_label=pseudo_labels
                    )
                else:
                    pseudo_labeled_dataset = PyODDataset(
                        X=X,
                        y_true_label=pseudo_labels,
                    )

                train_loader = IterLoader(torch.utils.data.DataLoader(pseudo_labeled_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=True))

            train_loader.new_epoch()

            self.train_model(
                len(train_loader), train_loader,
                epoch, optimizer
            )


    def train_model(self, train_iters, data_loader, epoch, optimizer):
        overall_loss = []
        contrastive_acc = []
        for i in range(train_iters):
        # for data, data_idx, batch_y_label, batch_sample_ind in train_loader:
            data, data_idx, batch_y_pseudo_label, batch_sample_ind, _ = data_loader.next()
            data = data.to(self.device).float()
            batch_y_pseudo_label = batch_y_pseudo_label.to(self.device)

            model_outputs = self.model(data)

            # reconstruction loss
            if "MSE" in self.loss_fn_name:
                loss = self.loss_fn(data, model_outputs['X_hat'])
                # print(f"MSE loss: {loss.item()}")

            if "mask_loss" in self.loss_fn_name:
                # todo: make sure the self.important_mask is working
                mask_loss_res = self.mask_loss(
                    model_outputs['latent_mask_h'],
                    model_outputs['latent_h'],
                )

                mask_loss = mask_loss_res['loss_v'].item()
                loss += mask_loss
                # print(f"mask_loss: {0.1 * mask_loss}")
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
                memory_res = self.model.memory(
                    model_outputs['latent_mask_h'],
                    batch_y_pseudo_label.squeeze().long()
                )
                memory_ce_loss = memory_res['CE_loss'].item()
                loss += memory_ce_loss
                # print(f"memory_ce_loss: {memory_ce_loss}")

                # refine_label = torch.max(memory_res['logits'], dim=1)[1]
                # 拿到pseudo label之后，获取其对应到异常的权重
                # pseudo_outlier_mask = torch.where(batch_y_pseudo_label.squeeze() == 1)
                # true_mask = refine_label == batch_y_true_label.squeeze()
                # correct_acc.append(sum(true_mask.cpu().numpy())/self.batch_size)

                # if pseudo_outlier_mask[0].size(0) > 0:
                #     norm_weight = torch.softmax(
                #         # F.normalize(memory_res['logits'], dim=1),
                #         torch.log(memory_res['logits']),
                #         dim=1
                #     )
                #     # 由memory_res 提供新的pseudo label以及对应的概率
                #     overlap_info = loss_overlap(
                #         model_outputs['dist_score'], norm_weight[:, 1],
                #         seed=unique2(self.epochs, epoch),
                #         batch_y_label=batch_y_pseudo_label.squeeze(),
                #         device=self.device
                #     )
                #     overlap_info_loss = overlap_info['loss'].item()
                #     loss += overlap_info_loss
                #     print(f"overlap_info_loss: {overlap_info_loss}")


            self.model.zero_grad()
            loss.backward()

            optimizer.step()
            overall_loss.append(loss.item())

        print(
            f'epoch {epoch}: training loss {np.mean(overall_loss)}, contrastive accuracy {np.mean(contrastive_acc)}'
        )

        # track the best model so far
        # todo: find way to select the best weight
        # if (np.mean(contrastive_acc) >= self.best_ctr_acc) and\
        #         (np.mean(overall_loss) <= self.best_loss):
        if np.mean(overall_loss) <= self.best_loss:
            # self.best_ctr_acc = np.mean(contrastive_acc)
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
            mean, std = np.mean(X, axis=0), np.std(X, axis=0)

            dataset = PyODDataset(
                X=X, mean=mean, std=std
            )
        else:
            dataset = PyODDataset(X=X)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)

        # enable the evaluation mode
        self.model.eval()
        self.model.memory.eval()

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])
        distribution_score = np.zeros([X.shape[0], ])
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
                # memory_logits[data_idx] = self.model.memory(infer_res['latent_mask_h'])["cos_simi"].cpu().numpy()
                memory_logits[data_idx] = torch.softmax(
                    torch.log(
                        self.model.memory(infer_res['latent_mask_h'])["logits"]
                        # dim=1
                    ),
                    dim=1
                ).cpu().numpy()
                distribution_score[data_idx] = infer_res['dist_score'].squeeze().cpu()

        return {
            'score': outlier_scores,
            'memory_logits': memory_logits,
            'distribution_score': distribution_score,
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
        self.labels_ae = (self.decision_scores_['score'] > self.threshold_)

        self.threshold_dist = np.percentile(self.decision_scores_['distribution_score'],
                                     100 * (1 - self.contamination))
        self.labels_score = (self.decision_scores_['distribution_score'] > self.threshold_dist)
        # calculate for predict_proba()
        # norm_ae_score = (self.decision_scores_['score'] - np.min(self.decision_scores_['score'])) /\
        #                 (np.max(self.decision_scores_['score']) - np.min(self.decision_scores_['score']))
        # norm_labels_score = (self.decision_scores_['distribution_score'] - np.min(self.decision_scores_['distribution_score'])) /\
        #                     (np.max(self.decision_scores_['distribution_score']) - np.min(self.decision_scores_['distribution_score']))

        norm_ae_score = (self.decision_scores_['score'] - np.mean(self.decision_scores_['score'])) /\
                        np.std(self.decision_scores_['score'])
        norm_labels_score = (self.decision_scores_['distribution_score'] - np.mean(self.decision_scores_['distribution_score'])) / \
                        np.std(self.decision_scores_['distribution_score'])

        # norm_ae_score = self.decision_scores_['score']
        # norm_labels_score = self.decision_scores_['distribution_score']

        self.final_score = norm_ae_score + norm_labels_score * self.decision_scores_['memory_logits'][:, 1]
        # self.final_score /= (1 + self.decision_scores_['memory_logits'][:, 1])
        if sample_name is not None:
            sort_final_score = np.sort(self.final_score)[::-1]
            norm_ae_ind = np.argsort(norm_ae_score)[::-1]
            norm_labels_ind = np.argsort(norm_labels_score)[::-1]

            outlier_sample_ae = sample_name[norm_ae_ind]
            print(norm_ae_score[norm_ae_ind][:15])
            print(outlier_sample_ae[:15])

            outlier_sample_score = sample_name[norm_labels_ind]
            print(norm_labels_score[norm_labels_ind][:30])
            print(outlier_sample_score[:30])

            norm_weight = self.decision_scores_['memory_logits'][norm_ae_ind]
            print(norm_weight[:30])

            argsort_final_score = np.argsort(self.final_score)[::-1]
            outlier_sample_name = sample_name[argsort_final_score]
            print(outlier_sample_name[:30])
            print(self.final_score[argsort_final_score][:30])
        self.labels_ = np.logical_or(self.labels_score, self.labels_ae).astype(
            'int').ravel()


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

        # check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        self.decision_scores_ = self.decision_function(X, )
        sample_name = kwargs.get("sample_name", None)

        self._process_decision_scores(sample_name)
        prediction = self.labels_

        if return_confidence:
            confidence = self.predict_confidence(X)
            return prediction, confidence

        return prediction