#!/usr/bin/env python
"""Module for Ensemble NN wrapper."""

import numpy as np
from torch.optim import SGD, Adam
from torch import randperm

from .ens import Ens_NN
from .learner import Learner
from ..nns.tchutils import npy, tch


def nnfit_1epoch(nnmodel, xtrn, ytrn, loss_fn, batch_size, optimizer):
    """
    Given an optimized model, this function optimizes the model for
    one epoch.
    Args:
        - nnmodel (NNWraper_Torch): model to be optmized.
        - xtrn (np.ndarray): input part of the training data.
        - ytrn (np.ndarray): target part of the training data.
        - batch_size (int): batch size used in the swag covariance estimation.
        - optimizer (torch optimizer): optimizer to use in the optimization.
    """
    try:
        device = nnmodel.device
    except AttributeError:
        device = "cpu"
    ntrn = len(xtrn)
    permutation = randperm(ntrn)
    xtrn = tch(xtrn, device=device, rgrad=False)
    ytrn = tch(ytrn, device=device, rgrad=False)
    xtrn = xtrn[permutation]
    ytrn = ytrn[permutation]
    for i in range(0, ntrn, batch_size):
        optimizer.zero_grad()
        loss = loss_fn(
            nnmodel,
            xtrn[i : i + batch_size, :],
            ytrn[i : i + batch_size, :],
            requires_grad=True,
        )
        loss.backward()
        optimizer.step()


class SWAG_NN(Ens_NN):
    """SWAG Ensemble NN Wrapper.
    Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019).
    A simple baseline for bayesian uncertainty in deep learning. Advances in neural
    information processing systems, 32.

    Attributes:
        - ndim (int): Number of model parameters.
        - learn_rate_swag (float, optional): Learning rate of the sampling for swag
        calcs. Defaults to 1e-5.
        - optim_swag (string): Optimizer type to use for the sampling for swag calcs.
        Options are: sgd.
        - n_steps (int): number of steps to consider in the sampling for swag calcs.
        - c (int): moment update frequency (sampling for swag).
        - k (int): max columns in deviation matrix  (swag).
        - loss_func : Loss function over which the mode is optimized. I takes, a NN model,
        x and y data, and requires_grad as input.
        - cov_type (string): type of covariance matrix approximation. Default is lowrank.
        Options are lowrank and diagonal.
        - means (list of np.ndarrays): list of means of the posterior over each member of the
        ensemble.
        - cov_diags (list of np.ndarrays): list of diagnoals of the covariance of the
        posterior over each member of the ensemble.
        - d_mats (list of np.ndarrays): np.ndarrays storing the low rank approximation
        of the covariance matrix (shape: (ndim x k)). Covariance matrix = d_mat*d_mat.T.
        - verbose (bool): Verbose or not.
    """

    def __init__(
        self,
        nnmodel,
        loss_func,
        ndim,
        nens=1,
        dfrac=1.0,
        learn_rate_swag=1e-5,
        optim_swag="sgd",
        n_steps=0,
        c=1,
        k=10,
        cov_type="lowrank",
        verbose=False,
    ):
        """Initialization.

        Args:
            - nnmodel (torch.nn.Module): NNWrapper class.
            - loss_func : Loss function over which the mode is optimized. I takes, a NN model,
            x and y data, and requires_grad as input.
            - ndim (int): Number of model parameters.
            - nens (int, optional): Number of ensemble members. Defaults to 1.
            - dfrac (float, optional): Fraction of data for each learner. Defaults to 1.0.
            - learn_rate_swag (float, optional): Learning rate of the sampling for swag
            calcs. Defaults to 1e-5.
            - optim_swag (string): Optimizer type to use for the sampling for swag calcs.
            Options are: sgd.
            - n_steps (int): number of steps to consider in the sampling for swag calcs.
            - c (int): moment update frequency (sampling for swag).
            - k (int): max columns in deviation matrix  (swag).
            - cov_type (string): type of covariance matrix approximation.
            - verbose (bool, optional): Verbose or not.
        """
        super().__init__(
            nnmodel, nens=nens, dfrac=dfrac, type_ens="swag", verbose=verbose
        )
        self.ndim = ndim
        self.learn_rate_swag = learn_rate_swag
        self.optim_swag = optim_swag
        self.nsteps = n_steps
        self.c = c
        self.k = k
        self.loss_func = loss_func
        self.cov_type = cov_type
        self.means = []
        self.cov_diags = []
        self.d_mats = []

        if self.verbose:
            self.print_params(names_only=True)

    def swag_calc(self, learner, xtrn, ytrn, batch_size):
        """
        Given an optimized model, this method stores in the corresponding lists
        the vectors and matrices defining the SWAG posterior.
        Args:
            - learner (Learner class): Instance of the Learner class including the model
            (torch.nn.Module) being analysed.
            - xtrn (np.ndarray): input part of the training data.
            - ytrn (np.ndarray): target part of the training data.
            - batch_size (int): batch size used in the swag covariance estimation.
        """
        moment_1st = npy(learner.nnmodel.p_flatten())
        moment_2nd = np.power(npy(learner.nnmodel.p_flatten()), 2)
        d_mat = []
        if self.optim_swag == "sgd":
            optim = SGD(learner.nnmodel.parameters(), lr=self.learn_rate_swag)
        else:
            raise NameError
        if self.cov_type == "lowrank":
            assert self.nsteps > self.k, "Number of steps after "
        for i in range(1, self.nsteps + 1):
            nnfit_1epoch(
                learner.nnmodel,
                xtrn,
                ytrn,
                self.loss_func,
                batch_size=batch_size,
                optimizer=optim,
            )
            if i % self.c == 0:
                n = i // self.c
                moment_1st = (n * moment_1st + npy(learner.nnmodel.p_flatten())) / (
                    n + 1
                )
                moment_2nd = (
                    n * moment_2nd + np.power(npy(learner.nnmodel.p_flatten()), 2)
                ) / (n + 1)
                if self.cov_type == "lowrank":
                    d_mat.append(npy(learner.nnmodel.p_flatten()) - moment_1st)
        self.means.append(np.squeeze(moment_1st))
        self.cov_diags.append(np.squeeze(moment_2nd - np.power(moment_1st, 2)))
        if self.cov_type == "lowrank":
            d_mat = np.transpose(np.array(d_mat[-self.k :]))
            self.d_mats.append(np.squeeze(d_mat))

    def fit_swag(self, xtrn, ytrn, **kwargs):
        """
        This method optimizes the models and then calculates the swag posterior for
        each of them.
        Args:
            - xtrn (np.ndarray): input part of the training data.
            - ytrn (np.ndarray): target part of the training data.
            - kwargs: they must include batch_size. Other arguments are to be used
            by the nnfit function.
        """
        self.fit(xtrn, ytrn, loss_fn="logposterior", loss=self.loss_func, **kwargs)
        for jens in range(self.nens):
            self.swag_calc(self.learners[jens], xtrn, ytrn, kwargs["batch_size"])

    def predict_sample(self, x, jens=None):
        """Predict a single, randomly selected sample.

        Args:
            x (np.ndarray): Input array of size `(N,d)`.
            jens (int): the ensemble index to use.
        Returns:
            np.ndarray: Output array of size `(N,o)`.
        """
        if jens is None:
            jens = np.random.randint(0, self.nens)
        z_1 = np.random.randn(self.ndim)
        z_2 = np.random.randn(self.k)
        theta = self.means[jens] + np.multiply(self.cov_diags[jens], z_1)
        if self.cov_type == "lowrank":
            theta = theta + np.dot(self.d_mats[jens], z_2)
        return self.learners[jens].nnmodel.predict(x, theta)
