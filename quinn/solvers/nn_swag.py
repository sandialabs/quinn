#!/usr/bin/env python
"""Module for SWAG NN wrapper."""

import numpy as np

from .nn_ens import NN_Ens
from ..nns.tchutils import npy
from ..nns.nnwrap import NNWrap


class NN_SWAG(NN_Ens):
    """SWAG NN Wrapper class.

    Attributes:
        c (int): Frequency of the moment update.
        cov_diags (list): List of diagonal covariances.
        cov_type (str): Covariance type.
        d_mats (list): List of D-matrices.
        datanoise (float): Data noise standard deviation.
        k (int): k-parameter of the method
        lr_swag (float): Learning rate.
        means (list): List of mean values of the history.
        n_steps (int): Number of steps in SWAG algorithm.
        nparams (int): Number of underlying NN module parameters.
        priorsigma (float): Standard deviation of the prior.
    """

    def __init__(self, nnmodel, k=10,
            n_steps=12, c=1, cov_type="lowrank", lr_swag=0.1,
            datanoise=0.1, priorsigma=1.0, **kwargs):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): NNWrapper class.
            k (int, optional): k-parameter of the method. Defaults to 10.
            n_steps (int, optional): Number of steps. Defaults to 12.
            c (int, optional): Frequency of moment update. Defaults to 1.
            cov_type (str, optional): Covariance type. Defaults to 'lowrank', anything else ignores low-rank approximation.
            lr_swag (float, optional): Learning rate. Defaults to 0.1.
            datanoise (float, optional): Data noise standard deviation. Defaults to 0.1.
            priorsigma (float, optional): Standard deviation of the prior. Defaults to 1.0.
            **kwargs: Any other keyword argument that :meth:`..nns.nnfit.nnfit` takes.
        """
        super().__init__(nnmodel, **kwargs)
        self.k = k
        assert(self.k>1)
        self.c = c
        self.n_steps = n_steps
        self.cov_type = cov_type
        if self.cov_type == "lowrank":
            assert(self.n_steps >= self.k)
        self.lr_swag = lr_swag
        self.datanoise = datanoise
        self.priorsigma = priorsigma
        self.nparams = sum(p.numel() for p in self.nnmodel.parameters())

        self.means = []
        self.cov_diags = []
        self.d_mats = []

    def fit(self, xtrn, ytrn, **kwargs):
        """Fitting function for each ensemble member.

        Args:
            xtrn (np.ndarray): Input array of size `(N,d)`.
            ytrn (np.ndarray): Output array of size `(N,o)`.
            **kwargs (dict): Any keyword argument that :meth:`..nns.nnfit.nnfit` takes.
        """
        for jens in range(self.nens):
            print(f"======== Fitting Learner {jens+1}/{self.nens} =======")

            ntrn = ytrn.shape[0]
            permutation = np.random.permutation(ntrn)
            ind_this = permutation[: int(ntrn * self.dfrac)]

            this_learner = self.learners[jens]

            kwargs["lhist_suffix"] = f"_e{jens}"
            kwargs["loss_fn"] = "logpost"
            kwargs["datanoise"] = self.datanoise
            #kwargs["priorparams"] = {'sigma': self.priorsigma}

            this_learner.fit(xtrn[ind_this], ytrn[ind_this], **kwargs)
            self.swag_calc(this_learner, xtrn[ind_this], ytrn[ind_this])


    def swag_calc(self, learner, xtrn, ytrn):
        """Given a learner, this method stores in the corresponding lists
        the vectors and matrices defining the posterior according to the
        laplace approximation.

        Args:
            learner (Learner): Instance of the Learner class including the model
            torch.nn.Module being used.
            xtrn (np.ndarray): input part of the training data.
            ytrn (np.ndarray): target part of the training data.
        """
        model = NNWrap(learner.nnmodel)

        moment1 = npy(model.p_flatten())
        moment2 = np.power(npy(model.p_flatten()), 2)


        d_mat = []
        for i in range(1, self.n_steps + 1):
            learner.fit(xtrn, ytrn, nepochs=1, optimizer='sgd', lrate=self.lr_swag) # TODO: does this need the main loss function, or the default is ok?

            if i % self.c == 0:
                n = i // self.c
                model = NNWrap(learner.nnmodel)
                moment1 = (n * moment1 + npy(model.p_flatten())) / (n + 1)
                moment2 = (n * moment2 + np.power(npy(model.p_flatten()), 2)) / (n + 1)
                if self.cov_type == "lowrank":
                    d_mat.append(npy(model.p_flatten()) - moment1)
                    if len(d_mat)>=self.k:
                        d_mat = d_mat[-self.k :]

        self.means.append(np.squeeze(moment1))
        self.cov_diags.append(np.squeeze(moment2 - np.power(moment1, 2)))
        if self.cov_type == "lowrank":
            self.d_mats.append(np.squeeze(np.array(d_mat).T))

    def predict_sample(self, x):
        """Predict a single sample.

        Args:
            x (np.ndarray): Input array `x` of size `(N,d)`.

        Returns:
            np.ndarray: Output array `x` of size `(N,o)`.
        """

        jens = np.random.randint(0, self.nens)

        z_1 = np.random.randn(self.nparams)
        z_2 = np.random.randn(self.k)
        theta = self.means[jens]
        theta_corr = np.multiply(np.sqrt(self.cov_diags[jens]), z_1)
        if self.cov_type == "lowrank":
            theta_corr = np.sqrt(0.5)*theta_corr + np.sqrt(0.5)*np.dot(self.d_mats[jens], z_2)/np.sqrt(self.k-1)

        theta += theta_corr
        model = NNWrap(self.learners[jens].nnmodel)

        return model.predict(x, theta)

    def predict_ens(self, x, nens=1):
        """Predict an ensemble of results.

        Args:
            x (np.ndarray): `(N,d)` input array.

        Returns:
            list[np.ndarray]: List of `M` arrays of size `(N, o)`, i.e. `M` random samples of `(N,o)` outputs.

        Note:
            This overloads NN_Ens's and QUiNN's base predict_ens function.
        """

        return self.predict_ens_fromsamples(x, nens=nens)
