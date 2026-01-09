# -*- coding: utf-8 -*-
"""Using AutoEncoder with Outlier Detection (PyTorch)
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from copy import deepcopy
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from statsmodels.sandbox.panel.sandwich_covariance_generic import kernel
from torch import nn

from .base import BaseDetector
from DetectOutlier.utils.stat_models import pairwise_distances_no_broadcast
from DetectOutlier.utils.torch_utility import get_activation_by_name
import torch.nn.functional as F

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



class EncoderLayer_selfattn(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer_selfattn, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        output, attn = self.attention(q, k, v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class VariationalAutoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 reduction=32,
                 hidden_neurons=(128, 64),
                 latent_dim=2,
                 dropout_rate=0.2,
                 batch_norm=True,
                 hidden_activation='LeakyReLU'):
        # initialize the super class
        super(VariationalAutoencoder, self).__init__()

        # save the default values
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.hidden_activation = hidden_activation
        self.reduction = reduction
        self.stride = 64
        self.d_model = 256
        self.d_inner = 512
        self.n_head = 8
        self.kernel_size = 128
        # get the object for the activations functions
        self.activation = get_activation_by_name(hidden_activation)

        # fill the local extraction layer
        self.atten = nn.ModuleList(
            [
                EncoderLayer_selfattn(
                    self.d_model,
                    self.d_inner,
                    self.n_head,
                    self.d_model // self.n_head,
                    self.d_model // self.n_head,
                    dropout=0.1,
                )
                for _ in range(1)
            ]
        )

        self.unfold = nn.Unfold(
            kernel_size=(1, self.kernel_size), dilation=1,
            padding=0, stride=(1, self.stride)
        )


        self.G_num = (self.n_features + 2 * 0 - 1 * (self.kernel_size - 1) - 1) // self.stride + 1

        self.emb_local = nn.Sequential(
            nn.Linear(self.kernel_size, self.d_model),
            nn.Tanh(),
        )
        self.out_linear = nn.Sequential(
            nn.Linear(self.d_model, hidden_neurons[-1]),
            nn.Tanh(),
        )
        # self.LG_fuse = nn.Linear(hidden_neurons[-1]+1, 1)
        self.LG_fuse = nn.Linear(hidden_neurons[-1]+1, 1)

        self.LG_de_fuse = AE_Block(
            hidden_neurons[-1]+1, self.kernel_size,
            activation=self.activation
        )
        self.fold = nn.Fold(
            kernel_size=(1, self.kernel_size), dilation=1,
            padding=0, stride=(1, self.stride), output_size=(1, self.n_features)
        )

        # create the dimensions for the input and hidden layers
        self.layers_neurons_encoder_ = [self.n_features, self.G_num, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_encoder_[::-1]



        # fill the encoder sequential with hidden layers
        self.encoder0 = AE_Block(
            self.layers_neurons_encoder_[0], self.layers_neurons_encoder_[1],
            activation=self.activation, dropout_rate=self.dropout_rate, batch_norm=True
        )
        self.encoder1 = AE_Block(
            self.layers_neurons_encoder_[1], self.layers_neurons_encoder_[2],
            activation=self.activation, dropout_rate=self.dropout_rate, batch_norm=True
        )
        self.encoder2 = AE_Block(
            self.layers_neurons_encoder_[2], self.layers_neurons_encoder_[3],
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
        self.decoder3 = AE_Block(
            self.layers_neurons_decoder_[3], self.layers_neurons_decoder_[4],
            activation=self.activation
        )

    def get_conditon(self, x_l):
        # local feature
        x_l = x_l.view(x_l.shape[0], 1, 1, -1)
        unfold_x = self.unfold(x_l) # split as Batch, kernel_siz(each group), X
        f_local = unfold_x.transpose(1, 2)  #
        f_local = self.emb_local(f_local)
        for enc_layer in self.atten:
            f_local, enc_slf_attn = enc_layer(f_local)
        f_local = self.out_linear(f_local)
        return f_local

    def forward(self, x):
        # we could return the latent representation here after the encoder
        # get the condition
        # residual = x
        f_local = self.get_conditon(x)

        # as the latent representation

        x = self.encoder0(x)
        LG_x = torch.cat([x.unsqueeze(2), f_local], dim=2)
        LG_x = LG_x.view(x.shape[0]*self.G_num, -1)
        LG_x = self.LG_fuse(LG_x)
        x = LG_x.view(-1, self.G_num)

        x = self.encoder1(x)
        x = self.encoder2(x)

        mu = self.mu_layer(x)
        std = torch.exp(0.5 * self.sigma_layer(x))
        eps = torch.randn_like(std).to(std.device)
        z = mu + eps * std

        x = self.decoder0(z)
        x = self.decoder1(x)
        x = self.decoder2(x)
        LG_x = torch.cat([x.unsqueeze(2), f_local], dim=2)
        LG_x = LG_x.view(x.shape[0] * self.G_num, -1)
        LG_x = self.LG_fuse(LG_x)
        x = LG_x.view(-1, self.G_num)
        x = self.decoder3(x)

        # LG_x = self.LG_de_fuse(LG_x)
        # LG_x = LG_x.view(-1, self.kernel_size, self.G_num)
        # x = self.fold(LG_x)


        # atten_weight = self.mask_learner(residual, x)

        return {
            "laten_vec": z,
            # "atten_weight": atten_weight,
            "score": x,
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
                 preprocessing=True,
                 loss_fn=None,
                 # verbose=1,
                 # random_state=None,
                 contamination=0.1,
                 device=None):
        super(LocalVAE, self).__init__(contamination=contamination)

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

        self.mask_simi_loss = torch.nn.CosineEmbeddingLoss()

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

        # optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=self.learning_rate,
        #     weight_decay=self.weight_decay)

        self.best_loss = float('inf')
        self.best_model_dict = None
        self.best_epoch = -1

        for epoch in range(self.epochs):
            overall_loss = []
            for data, data_idx in train_loader:
                data = data.to(self.device).float()
                model_res = self.model(data)
                # reconstruction loss
                recon_loss = self.loss_fn(data, model_res['score'])
                # kl loss
                kl_loss = (model_res['sigma'] ** 2 + model_res['mu'] ** 2 - torch.log(model_res['sigma']) - 1 / 2).sum()


                # mask = model_res['atten_weight']
                # recon_loss = self.loss_fn(data, model_res['score'])

                loss = kl_loss + recon_loss

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())
            print('epoch {epoch}: training loss {train_loss} '.format(
                epoch=epoch, train_loss=np.mean(overall_loss)))

            # track the best model so far
            if np.mean(overall_loss) <= self.best_loss:
                print("epoch {ep} is the current best; loss={loss}".format(ep=epoch, loss=np.mean(overall_loss)))
                self.best_loss = np.mean(overall_loss)
                self.best_model_dict = deepcopy(self.model.state_dict())
                self.best_epoch = epoch

    # noinspection PyUnresolvedReferences
    def fit(self, X, y=None):
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
            # reduction=reduction,
            hidden_activation=self.hidden_activation
        )

        # move to device and print model information
        self.model = self.model.to(self.device)
        print(self.model)

        # train the autoencoder to find the best one
        self._train_autoencoder(train_loader)

        print(f"using best model weights on {self.best_epoch}")
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
                                                 batch_size=1,
                                                 shuffle=False)
        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])
        feat_dist = np.zeros([X.shape[0], X.shape[1]])

        self.model.eval()
        with torch.no_grad():
            for data, data_idx in dataloader:
                data_cuda = data.to(self.device).float()
                model_res = self.model(data_cuda)
                batch_dist = np.square(
                    (data_cuda).cpu().numpy() - (model_res['score']).cpu().numpy()
                )
                # print(batch_dist)

                # this is the outlier score
                score0 = np.sqrt(np.sum(batch_dist, axis=1)).ravel()
                outlier_scores[data_idx] = score0
                feat_dist[data_idx, :] = batch_dist

        # print('featuer accuracy on test set: %d %% ' % (100 * correct / total))

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
