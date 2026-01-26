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
from sklearn import metrics
from .base import BaseDetector
import random
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from collections import defaultdict
from visualization.visulize import plot_gan_loss

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




class SOGAAL_Pytorch(BaseDetector):
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
                 decay=1e-6, momentum=0.9, contamination=0.1, batch_size=64, verbose=True):
        super(SOGAAL_Pytorch, self).__init__(contamination=contamination)
        self.stop_epochs = stop_epochs
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.decay = decay
        self.batch_size = batch_size
        self.momentum = momentum
        self.verbose = verbose



        # specify the device on which the model will be trained
        self.TRAINING_ON_GPU = torch.cuda.is_available()
        print(f"using cuda: {self.TRAINING_ON_GPU}")
        self.device = torch.device("cuda:0" if self.TRAINING_ON_GPU else "cpu")

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
        X = check_array(X)
        self._set_n_classes(y)
        latent_size = X.shape[1]
        data_size = X.shape[0]
        stop = 0
        epochs = self.stop_epochs  * 3
        train_history = defaultdict(list)

        # Create a tensor dataset
        # create tensors from np array
        data_x_tensor = torch.Tensor(X).to(self.device)
        train_set = torch.utils.data.TensorDataset(data_x_tensor)
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
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))
                noise = torch.tensor(noise, dtype=torch.float32).to(self.device)

                # Generate potential outliers
                generated_data = self.generator(noise)

                # Concatenate real data to generated data
                # X = torch.tensor(np.concatenate([data_batch, generated_data]), dtype=torch.float32)
                X = torch.cat((data_batch[0], generated_data))
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
        if self.verbose:
            plot_gan_loss(train_history)

        # Detection result
        with torch.no_grad():
            self.discriminator.eval()
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
            for data, data_idx in dataloader:
                data_cuda = data.to(self.device).float()
                # this is the outlier score
                outlier_scores[data_idx] = self.discriminator(data_cuda).cpu().squeeze().numpy()

        return {
            'score': outlier_scores,
        }



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