#!/usr/bin/env python
"""Module for Swag NN wrapper."""

import numpy as np
import torch
from torch.optim import SGD, Adam
from torch import randperm

from .nn_ens import NN_Ens
from ..ens.learner import Learner
from ..nns.tchutils import npy, tch
from ..nns.nnwrap import NNWrap
from ..nns.losses import NegLogPost


class NN_SWAG(NN_Ens):

    def __init__(self, nnmodel, k=10,
            n_steps=12, c=1, cov_type="lowrank", lr_swag=0.1,
            datanoise=0.1, priorsigma=1.0, **kwargs):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): NNWrapper class.
            datanoise (float): Data noise size
            **kwargs: Description
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
            **kwargs (dict): Keyword arguments.
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
        """
        Given an optimized model, this method stores in the corresponding lists
        the vectors and matrices defining the posterior according to the
        laplace approximation.
        Args:
            - learner (Learner class): Instance of the Learner class including the model
            (torch.nn.Module) being analysed.
            - xtrn (np.ndarray): input part of the training data.
            - ytrn (np.ndarray): target part of the training data.
            - batch_size (int): batch size used in the hessian estimation.
            Defaults to None.
        """
        model = NNWrap(learner.nnmodel)

        moment1 = npy(model.p_flatten())
        moment2 = np.power(npy(model.p_flatten()), 2)


        d_mat = []
        for i in range(1, self.n_steps + 1):
            learner.fit(xtrn, ytrn, nepochs=1, optimizer='sgd', lrate=self.lr_swag)

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

        return self.predict_ens_fromsamples(x, nens=nens)
