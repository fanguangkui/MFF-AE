# -*- coding: utf-8 -*-
"""Using AutoEncoder with Outlier Detection (PyTorch)
"""
# 利用监督学习，验证score损失函数的有效性，除了重构误差之外，score 分布误差的学习的确能够拉开异常值的得分

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

class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, mean=None, std=None, y_train_label=None, sample_index_id=None
                 ):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std
        if y_train_label is not None:
            self.y_train_label = np.expand_dims(y_train_label, axis=1)
        else:
            self.y_train_label = np.expand_dims(np.array([-100]*X.shape[0]), axis=1)

        if sample_index_id is not None:
            self.sample_index_id = np.expand_dims(sample_index_id, axis=1)
        else:
            self.sample_index_id = np.expand_dims(np.arange(0, X.shape[0]), axis=1)


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

        batch_y_train_label = self.y_train_label[idx, :]
        batch_sample_index_id = self.sample_index_id[idx, :]

        return torch.from_numpy(sample), idx, batch_y_train_label, batch_sample_index_id


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

def loss_overlap(score_all, assign_prob, seed, bw_u=1.0, bw_a=1.0, x_num=1000,
                 plot=False, pro=True, ensemble=False, device=None, topk_value=1, **kwargs):
    # set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sort_prob, sort_ind = torch.sort(assign_prob, dim=0)
    # 选取异常概率最大的top个值
    top_u_prob = sort_prob[:topk_value]
    top_u_ind = sort_ind[:topk_value]
    # todo: 这个阈值的选取方法？？？
    u_ind = top_u_ind[torch.where(top_u_prob < 0.5)]
    u_prob = top_u_prob[torch.where(top_u_prob < 0.5)]

    assign_history = torch.cat((kwargs['assign_history'], top_u_prob.unsqueeze(0).detach()), dim=0)

    if u_prob.size(0) > 0:
        # Min-Max scaling
        min_hist_prob = torch.min(assign_history)
        max_hist_prob = torch.tensor(0.5).to(min_hist_prob.device)
        weight_assign_prob = (assign_history - min_hist_prob) / (max_hist_prob - min_hist_prob)

        u_weight = torch.mean(1 - weight_assign_prob[-topk_value:])

        # trail1:
        mask = torch.isin(sort_ind, u_ind, invert=True)

        # trail2: 保证正样本的纯洁性  res: 混在正样本中的异常值得分过高，反而导致正常样本数量减少，结果精度下降
        # mask = torch.where(sort_prob > 0.5)
        a_ind = sort_ind[mask]

        if "batch_y_label" in kwargs:
            print(
                f"set pseudo normal label :"
                f"{torch.where(kwargs['batch_y_label'][a_ind] > 0)[0]}/{kwargs['batch_y_label'][a_ind].size(0)}"
            )

            print(
                f"set pseudo outlier label :"
                f"{kwargs['batch_y_label'][u_ind]}"
                f" with weight {u_weight} and assign probability {u_prob}"
            )
        # todo: 每一个伪异常值都构建一个分布？
        #  （但其实真正异常的就那一个，构建分布的数量决定了模型对于异常边界定义的严格程度，只选top的话，一个batch中的多个异常值剩下的就归于正常了？？？）
        s_u = score_all[u_ind]
        s_a = score_all[a_ind]
        # remove duplicates
        # s_u = torch.unique(s_u)
        # s_a = torch.unique(s_a)

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
            "loss": area * u_weight,
            "assign_history": assign_history
        }
    else: # 这种情况说明没有异常样本
        return {
            "loss": None,
            "assign_history": assign_history
        }

# InfoNCE: https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py
# https://discuss.pytorch.org/t/initialize-parameters-in-a-torch-autograd-function/98576
# https://stackoverflow.com/questions/61117361/how-to-use-custom-torch-autograd-function-in-nn-sequential-model
class MemoryLayer(Function):
    @staticmethod
    def forward(ctx, inputs, targets, memory, alpha=0.6):
        ctx.memory = memory
        ctx.alpha = alpha
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(memory.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.memory)
        for x, y in zip(inputs, targets):
            ctx.memory[y] = ctx.alpha * ctx.memory[y] + (1. - ctx.alpha) * x
            ctx.memory[y] /= ctx.memory[y].norm()
        return grad_inputs, None

class Memory(nn.Module):
    def __init__(
            self, num_features, num_classes, alpha=0.01
    ):
        super(Memory, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha

        self.mem = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)

    def forward(self, epoch, query, targets):
        # Normalize to unit vectors
        alpha = 0.5 * epoch / 60
        logits = MemoryLayer.apply(query, targets, self.mem, alpha)
        return logits

class MPLP(object):
    def __init__(self, t=0.6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.t = t

    def predict(self, memory, targets):
        mem_vec = memory[targets]
        mem_sim = mem_vec.mm(memory.t())
        m, n = mem_sim.size()
        mem_sim_sorted, index_sorted = torch.sort(mem_sim, dim=1, descending=True)
        multilabel = torch.zeros(mem_sim.size()).to(self.device)
        mask_num = torch.sum(mem_sim_sorted > self.t, dim=1)

        for i in range(m):
            topk = int(mask_num[i].item())
            topk = max(topk, 10)
            topk_idx = index_sorted[i, :topk]
            vec = memory[topk_idx]
            sim = vec.mm(memory.t())
            _, idx_sorted = torch.sort(sim.detach().clone(), dim=1, descending=True)
            step = 1
            for j in range(topk):
                pos = torch.nonzero(idx_sorted[j] == index_sorted[i, 0]).item()
                if pos > topk: break
                step = max(step, j)
            step = step + 1
            step = min(step, mask_num[i].item())
            if step <= 0: continue
            multilabel[i, index_sorted[i, 0:step]] = float(1)

        targets = torch.unsqueeze(targets, 1)
        multilabel.scatter_(1, targets, float(1))

        return multilabel

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
                 global_memo=False
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

        self.memo = Memory(self.n_features, num_classes=self.n_samples)

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

        # self.reg_layer = {}
        self.reg_1 = nn.Sequential(
            nn.Linear(self.n_features+self.layers_neurons_encoder_[-1]+1, 256),
            self.activation
        )

        self.reg_2 = nn.Sequential(
            nn.Linear(256+1, 32),
            self.activation
        )

        self.reg_3 = nn.Sequential(
            nn.Linear(32+1, 1),
            nn.BatchNorm1d(num_features=1)
        )

    def forward(self, x):
        x_feat = self.adaptive_format(x)
        # we could return the latent representation here after the encoder
        # as the latent representation
        h = self.encoder(x_feat)
        X_hat = self.decoder(h)

        ### for outlier score distribution
        # reconstruction residual vector
        r = torch.sub(X_hat, x_feat)
        # reconstruction error
        e = r.norm(dim=1).reshape(-1, 1)
        # normalized reconstruction residual vector
        r = torch.div(r, e) #div by broadcast
        # regression
        feature = self.reg_1(torch.cat((h, r, e), dim=1))
        feature = self.reg_2(torch.cat((feature, e), dim=1))
        score = self.reg_3(torch.cat((feature, e), dim=1))

        ### for contrastive learning

        return {
            "x_feat": X_hat,
            "score": score.squeeze()
        }


class MaskCTR_AEScore(BaseDetector):
    def __init__(self,
                 hidden_neurons=None,
                 hidden_activation='relu',
                 batch_norm=True,
                 learning_rate=1e-3,
                 epochs=100,
                 global_memo=False,
                 batch_size=32,
                 dropout_rate=0.2,
                 weight_decay=1e-5,
                 preprocessing=True,
                 loss_fn_name="MSE",
                 contamination=0.1,
                 topk_value=1,
                 device=None,
                 mplp_t=0.6
                 ):
        super(MaskCTR_AEScore, self).__init__(contamination=contamination)

        # save the initialization values
        # self.global_memo = global_memo

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
        self.mplp_t = mplp_t

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

        self.labelpred = MPLP(self.mplp_t)

    # noinspection PyUnresolvedReferences
    def fit(self, X, y=None, **kwargs):
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        if "y_train_label" in kwargs:
            y_train_label = kwargs['y_train_label']
        else:
            y_train_label = None

        if "sample_index_id" in kwargs:
            sample_index_id = kwargs["sample_index_id"]
        else:
            sample_index_id = None


        n_samples, n_features = X.shape[0], X.shape[1]

        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(
                X=X, mean=self.mean, std=self.std,
                sample_index_id=sample_index_id,
                y_train_label=y_train_label
            )
        else:
            train_set = PyODDataset(X=X, y_train_label=y_train_label, sample_index_id=sample_index_id)

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
            # global_memo=self.global_memo
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
        memory_para = []
        mask_para = []
        for iname, ipara in self.model.named_parameters():
            if "important_mask" in iname:
                mask_para.append(ipara)
            elif "memo" in iname:
                memory_para.append(ipara)
            else:
                AE_para.append(ipara)
        param_groups = [
            {'params': mask_para, 'lr_mult': 100},
            {'params': memory_para, 'lr_mult': 0.01},
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
            for data, data_idx, batch_y_label, batch_sample_index_id in train_loader:
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
                        torch.sigmoid(self.model.important_mask).repeat(format_data.size(0), 1) * format_data
                    )
                    loss += mask_loss_res['loss_v']
                    # t1 = batch_y_label[mask_loss_res['recog_mask']]
                    # print(f"unable to recognize outlier/inlier: {torch.where(t1==1)[0].size(0)}/{t1.size(0)}")
                    contrastive_acc.append(mask_loss_res['accuracy'].item())

                if (epoch>=3) and ("score_distribution_loss" in self.loss_fn_name):
                    logits = self.model.memo(epoch, model_outputs['x_feat'], batch_sample_index_id)
                    #
                    multilabel = self.labelpred.predict(
                        self.model.memo.mem.detach().clone(),
                        batch_sample_index_id.detach().clone()
                    )
                    overlap_info = loss_overlap(
                        model_outputs["score"], logits,
                        seed=unique2(self.epochs, epoch), topk_value=self.topk_value,
                        batch_y_label=batch_y_label,
                        assign_history=self.assign_history
                    )
                    if overlap_info['loss'] is not None:
                        loss += overlap_info['loss']
                        self.assign_history = overlap_info['assign_history']
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
        distribution_score = np.zeros([X.shape[0], ])
        with torch.no_grad():
            for data, data_idx, _, _ in dataloader:
                data_cuda = data.to(self.device).float()
                infer_res = self.model(data_cuda)
                # this is the outlier score
                outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                    self.model.adaptive_format(data).cpu().numpy(),
                    # data,
                    infer_res['x_feat'].cpu().numpy()
                )
                distribution_score[data_idx] = infer_res['score']

        return {
            'score': outlier_scores,
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
        # np.where(sample_name.values=='iBAQ 38')[0]=240
        # np.where(sample_name.values=='iBAQ 46')[0]=249
        # np.where(sample_name.values=='iBAQ 80')[0]=287
        # np.where(sample_name.values=='iBAQ 83')[0]=290
        # np.where(sample_name.values=='iBAQ 227')[0]=142
        # np.where(sample_name.values=='iBAQ 249')[0]=166
        # np.where(sample_name.values=='iBAQ 263')[0]=182
        # np.where(sample_name.values=='iBAQ 266')[0]=185
        self.threshold_dist = self.decision_scores_['score']
        # self.threshold_ = np.percentile(self.decision_scores_['score'],
        #                              100 * (1 - self.contamination))
        # self.labels_ = (self.decision_scores_['score'] > self.threshold_).astype(
        #     'int').ravel()
        self.threshold_ = np.percentile(self.decision_scores_['score'],
                                     100 * self.contamination)
        self.labels_ = (self.decision_scores_['score'] < self.threshold_).astype(
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