#!/usr/bin/env python
"""Module for Ensemble NN wrapper."""

import numpy as np
from torch.optim import SGD
from torch import randperm

from .learner import Learner
from ..quinn import QUiNNBase
from ..nns.tchutils import npy, tch


def nnfit_1epoch(nnmodel, xtrn, ytrn, loss_fn, batch_size, learn_rate, optimizer="sgd"):
    if optimizer == "sgd":
        optimizer = SGD(nnmodel.parameters(), lr=learn_rate)
    else:
        raise NameError
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
        optimizer.step()


class SWAG_NN(QUiNNBase):
    """SWAG Ensemble NN Wrapper.
    Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019).
    A simple baseline for bayesian uncertainty in deep learning. Advances in neural
    information processing systems, 32.

    Attributes:
        nens (int): Number of ensemble members.
        learners (list[Learner]): List of learners.
        dfrac (float): Fraction of data each learner sees.
        verbose (bool): Verbose or not.
    """

    def __init__(
        self,
        nnmodel,
        loss_func,
        ndim,
        nens=1,
        dfrac=1.0,
        learn_rate=1e-2,
        n_steps=0,
        c=1,
        k=10,
        s=20,
        optimizer="sgd",
        cov_type="lowrank",
        verbose=False,
    ):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): NNWrapper class.
            nens (int, optional): Number of ensemble members. Defaults to 1.
            dfrac (float, optional): Fraction of data for each learner. Defaults to 1.0.
            verbose (bool, optional): Verbose or not.
        """
        super().__init__(nnmodel)
        self.verbose = verbose
        self.nens = nens
        self.dfrac = dfrac
        self.ndim = ndim
        self.learn_rate = learn_rate
        self.nsteps = n_steps
        self.c = c
        self.k = k
        self.s = s
        self.loss_func = loss_func
        self.cov_type = cov_type
        self.optimizer = optimizer
        self.learners = []
        self.means = []
        self.cov_diags = []
        self.d_mats = []
        for i in range(nens):
            self.learners.append(Learner(nnmodel))

        if self.verbose:
            self.print_params(names_only=True)

    def print_params(self, names_only=False):
        """Print model parameter names and optionally, values.

        Args:
            names_only (bool, optional): Print names only. Default is False.
        """
        for i, learner in enumerate(self.learners):
            print(f"==========  Learner {i+1}/{self.nens}  ============")
            learner.print_params(names_only=names_only)

    def swag_calc(self, learner, xtrn, ytrn, batch_size):
        """
        Given an optimized model, this method stores in the corresponding lists
        the vectors and matrices defining the SWAG posterior.
        """
        moment_1st = npy(learner.nnmodel.p_flatten())
        moment_2nd = np.power(npy(learner.nnmodel.p_flatten()), 2)
        d_mat = []
        if self.cov_type == "lowrank":
            assert self.nsteps > self.k, "Number of steps after "
        for i in range(1, self.nsteps + 1):
            nnfit_1epoch(
                learner.nnmodel,
                xtrn,
                ytrn,
                self.loss_func,
                learn_rate=self.learn_rate,
                batch_size=batch_size,
                optimizer=self.optimizer,
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

    def fit(self, xtrn, ytrn, **kwargs):
        """Fitting function for each ensemble member.

        Args:
            xtrn (np.ndarray): Input array of size `(N,d)`.
            ytrn (np.ndarray): Output array of size `(N,o)`.
            **kwargs (dict): Keyword arguments.
        """
        assert "batch_size" in kwargs, "batch_size is not present in kwargs"

        for jens in range(self.nens):
            print(f"======== Fitting Learner {jens+1}/{self.nens} =======")

            ntrn = ytrn.shape[0]
            permutation = np.random.permutation(ntrn)
            ind_this = permutation[: int(ntrn * self.dfrac)]

            this_learner = self.learners[jens]

            kwargs["lhist_suffix"] = f"_e{jens}"
            this_learner.fit(
                xtrn[ind_this],
                ytrn[ind_this],
                loss_fn="logposterior",
                loss=self.loss_func,
                **kwargs,
            )
            self.swag_calc(this_learner, xtrn, ytrn, kwargs["batch_size"])

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

    def predict_ens(self, x, nens=None):
        """Predict from all ensemble members.

        Args:
            x (np.ndarray): `(N,d)` input array.
            nens (int): number of models to use in the ensemble.
            s (int): number of samples from an individual gaussian posterior.

        Returns:
            np.ndarray: Array of size `(M, N, o)`, i.e. `M` random samples of `(N,o)` outputs.

        Note:
            This overloads QUiNN's base predict_ens function.
        """
        if nens is None:
            nens = self.nens * self.s

        assert (
            nens >= self.nens
        ), f"Requested number of samples must be larger than the number of ensembles ({self.nens})"

        s = nens // self.nens

        permuted_inds = np.random.permutation(self.nens)

        y_all = []
        for jens in range(self.nens):
            for _ in range(s):
                y = self.predict_sample(x, jens=permuted_inds[jens])
                y_all.append(y)

        return y_all
