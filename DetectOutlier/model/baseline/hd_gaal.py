# -*- coding: utf-8 -*-
"""Single-Objective Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection
"""
# Author: Winston Li <jk_zhengli@hotmail.com>
# License: BSD 2 clause


import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base import BaseDetector
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from collections import defaultdict
from visualization.visulize import plot_gan_loss
from copy import deepcopy

class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None, noise_type='uniform', shuffle_ratio=0.1):
        super(PyODDataset, self).__init__()
        self.X = X
        self.latent_size = X.shape[1]
        self.noise_type = noise_type
        self.shuffle_ratio = shuffle_ratio

    def generate_noise(
            self, noise, _batch_size,
            noise_type='uniform',
            shuffle_ratio=0.1
    ):
        noise_size = _batch_size
        if noise_type == 'uniform':
            noise = np.random.uniform(0, 1, (int(noise_size), self.latent_size))
            # noise = torch.tensor(noise, dtype=torch.float32).to(device)
        elif noise_type == 'feature_shuffle':
            np.random.shuffle(noise)
            # noise = torch.from_numpy(np.float32(noise_df.values)).to(device)
        else:
            raise "unknown noise data generator"

        return noise

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            _batch_size = len(idx)
        else:
            _batch_size = 1
        sample = self.X[idx, :]
        noise = self.generate_noise(
            deepcopy(sample), _batch_size, noise_type=self.noise_type,
            shuffle_ratio=self.shuffle_ratio
        )

        return torch.from_numpy(sample), torch.from_numpy(np.float32(noise)).squeeze(), idx
        # return torch.from_numpy(sample), torch.from_numpy(sample), idx

# Generator
class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU()
        )
        # initialize the weights
        self.model.apply(self._identity_init)

        # checkpoint dir to save and load the model
        self.checkpoint_dir = './chkpt/generator/'

    def forward(self, input):
        return self.model(input)

    def _identity_init(self, m):
        if type(m) == nn.Linear:
            nn.init.eye_(m.weight)
            m.bias.data.fill_(1.e-5)

    def save(self, run):
        print('Saving {} to disk..'.format(self.__class__.__name__))
        torch.save(self.state_dict(), self.checkpoint_dir + str(run))

    def load(self, run):
        checkpoint = torch.load(self.checkpoint_dir + str(run))
        self.load_state_dict(checkpoint)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, latent_size, data_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, math.ceil(math.sqrt(data_size))),
            nn.ReLU(),
            nn.Linear(math.ceil(math.sqrt(data_size)), 1),
            nn.Sigmoid()
        )
        # initiliaze the weights
        self.model.apply(self._xavier_init)

        # checkpoint dir to save and load the model
        self.checkpoint_dir = './chkpt/discriminator/'

    def forward(self, input):
        return self.model(input)

    def _xavier_init(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def save(self, run):
        print('Saving {} to disk..'.format(self.__class__.__name__))
        torch.save(self.state_dict(), self.checkpoint_dir + str(run))

    def load(self, run):
        checkpoint = torch.load(self.checkpoint_dir + str(run))
        self.load_state_dict(checkpoint)

################################## TransFormer embedding start ##################################
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # permute: batch_size, text_len, embedding_size -> batch_size, embedding_size, text_len
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        # batch_size, out_channels, x.shape[1]-3+1
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
################################## TransFormer embedding end ##################################
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


################################## TransFormer encoder start ##################################
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list

class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout,
                                         output_attention=self.output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]

################################## TransFormer encoder end ##################################


class HDGAAL_Pytorch(BaseDetector):
    """Single-Objective Generative Adversarial Active Learning.

    SO-GAAL directly generates informative potential outliers to assist the
    classifier in describing a boundary that can separate outliers from normal
    data effectively. Moreover, to prevent the generator from falling into the
    mode collapsing problem, the network structure of SO-GAAL is expanded from
    a single generator (SO-GAAL) to multiple generators with different
    objectives (MO-GAAL) to generate a reasonable reference distribution for
    the whole dataset.
    Read more in the :cite:`liu2019generative`.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    stop_epochs : int, optional (default=20)
        The number of epochs of training. The number of total epochs equals to three times of stop_epochs.

    lr_d : float, optional (default=0.01)
        The learn rate of the discriminator.

    lr_g : float, optional (default=0.0001)
        The learn rate of the generator.

    decay : float, optional (default=1e-6)
        The decay parameter for SGD.

    momentum : float, optional (default=0.9)
        The momentum parameter for SGD.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, stop_epochs=500, lr_d=0.0001, lr_g=0.0004,
                 decay=1e-6, momentum=0.9, contamination=0.1, batch_size=64,
                 verbose=True, noise_type='uniform', shuffle_ratio=0.1):
        super(HDGAAL_Pytorch, self).__init__(contamination=contamination)
        self.stop_epochs = stop_epochs
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.decay = decay
        self.batch_size = batch_size
        self.momentum = momentum

        self.noise_type=noise_type
        self.shuffle_ratio=shuffle_ratio

        # specify the device on which the model will be trained
        self.TRAINING_ON_GPU = torch.cuda.is_available()
        print(f"using cuda: {self.TRAINING_ON_GPU}")
        self.device = torch.device("cuda:0" if self.TRAINING_ON_GPU else "cpu")
        # self.device = torch.device("cpu")

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
        X = check_array(X)
        self._set_n_classes(y)
        latent_size = X.shape[1]
        data_size = X.shape[0]
        stop = 0
        epochs = self.stop_epochs  * 3
        train_history = defaultdict(list)

        # Create a tensor dataset
        # create tensors from np array
        train_set = PyODDataset(X=X, noise_type=self.noise_type, shuffle_ratio=self.shuffle_ratio)
        # Create a train loader
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   drop_last=True)

        self.discriminator = Discriminator(latent_size, data_size)
        discriminator_optim = optim.SGD(self.discriminator.parameters(),
                                        lr=self.lr_d, dampening=self.decay,
                                        momentum=self.momentum
                                        )
        discriminator_criterion = F.binary_cross_entropy
        self.discriminator.to(self.device)  # send the network to the specified device

        # Create generator
        self.generator = Generator(latent_size)
        generator_optim = optim.SGD(self.generator.parameters(),
                                    lr=self.lr_g, dampening=self.decay,
                                    momentum=self.momentum
                                    )
        generator_criterion = F.binary_cross_entropy
        self.generator.to(self.device)

        # Start iteration
        for epoch in range(epochs):
            print('Epoch {} of {}'.format(epoch + 1, epochs))

            for i, data_batch in enumerate(train_loader):
                print('\tTesting for epoch {} batch_index {}/{}'.format(epoch + 1, i + 1, len(train_loader)))

                # Generate noise
                noise_size = self.batch_size
                true_data_cpu, noise_data_cpu, _ = data_batch
                true_data = true_data_cpu.to(self.device).to(torch.float32)
                noise_data = noise_data_cpu.to(self.device).to(torch.float32)
                # Generate potential outliers
                generated_data = self.generator(noise_data)

                # Concatenate real data to generated data
                # X = torch.tensor(np.concatenate([data_batch, generated_data]), dtype=torch.float32)
                X = torch.cat((true_data, generated_data))
                Y = torch.tensor(
                    np.array([0] * self.batch_size + [1] * int(noise_size)),
                    dtype=torch.float32
                ).unsqueeze(dim=1).to(self.device)

                # Train discriminator
                # enable training mode
                self.discriminator.train()
                # getting the prediction
                discriminator_pred = self.discriminator(X)
                # compute the loss
                discriminator_loss = discriminator_criterion(discriminator_pred, Y)
                # reset the gradients to avoid gradients accumulation
                discriminator_optim.zero_grad()
                # compute the gradients of loss w.r.t weights
                discriminator_loss.backward(retain_graph=True)
                # update the weights
                discriminator_optim.step()
                # Store the loss for later use
                train_history['discriminator_loss'].append(discriminator_loss.item())

                # Train generator
                # create fake labels
                trick = torch.tensor(
                    np.array([0] * noise_size), dtype=torch.float32
                ).unsqueeze(dim=1).to(self.device)
                self.discriminator.eval()  # freeze the discriminator
                if stop == 0:
                    self.generator.train()  # enable training mode for the generator
                    generator_loss = generator_criterion(self.discriminator(generated_data), trick)
                    generator_optim.zero_grad()
                    generator_loss.backward(retain_graph=True)
                    generator_optim.step()
                    train_history['generator_loss'].append(generator_loss.item())
                else:
                    self.generator.eval()  # enable evaluation mode
                    generator_loss = generator_criterion(self.discriminator(generated_data), trick)
                    train_history['generator_loss'].append(generator_loss.item())

                # unfreeze the discriminator's layers
                for param in self.discriminator.parameters():
                    param.requires_grad = True
            # Stop training generator
            if epoch + 1 > self.stop_epochs:
                stop = 1

            self.discriminator.eval()

        # assert GAN run successfully
        plot_gan_loss(train_history)

        # Detection result
        with torch.no_grad():
            self.discriminator.eval()
            data_x_tensor=torch.Tensor(train_set.X).to(self.device)
            decision_scores_ = self.discriminator(data_x_tensor)
            if self.TRAINING_ON_GPU:
                decision_scores_ = decision_scores_.cpu()
            self.decision_scores_ = decision_scores_.numpy()
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
        check_is_fitted(self, ['discriminator'])
        X = check_array(X)
        dataset = PyODDataset(X=X)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)
        # enable the evaluation mode
        self.discriminator.eval()
        outlier_scores = np.zeros([X.shape[0], ])
        with torch.no_grad():
            for data, _, data_idx in dataloader:
                data_cuda = data.to(self.device).float()
                # this is the outlier score
                outlier_scores[data_idx] = self.discriminator(data_cuda).cpu().squeeze().numpy()

        return outlier_scores