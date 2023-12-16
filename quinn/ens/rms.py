#!/usr/bin/env python
"""Module for Ensemble NN wrapper."""

import numpy as np
import torch
from torch.optim import SGD, Adam
from torch import randperm

from .ens import Ens_NN
from .learner import Learner
from ..nns.tchutils import npy, tch


class RMS_NN(Ens_NN):
    """RMS Ensemble NN Wrapper.
    Pearce, Tim, Felix Leibfried, and Alexandra Brintrup. "Uncertainty in
    neural networks: Approximately bayesian ensembling." International conference
    on artificial intelligence and statistics. PMLR, 2020.
    Attributes:
        - loss_func : Loss function over which the mode is optimized. I takes,
        a NN model, x and y data, and requires_grad as input.
    """

    def __init__(
        self,
        nnmodel,
        loss_func,
        nens=1,
        dfrac=1.0,
        verbose=False,
    ):
        """Initialization.
        Args:
            nnmodel (torch.nn.Module): NNWrapper class.
            - loss_func : Loss function over which the mode is optimized. I takes,
            a NN model, x and y data, and requires_grad as input.
            - nens (int, optional): Number of ensemble members. Defaults to 1.
            - dfrac (float, optional): Fraction of data for each learner. Defaults
            to 1.0.
            - la_type (string): type of covariance matrix approximation. Default
            is full. Options are full, kfac and diag.
            - cov_scale (float): approximated covariances are scaled by this
            factor. Defaults to 1.
            - verbose (bool, optional): Verbose or not.
        """
        super().__init__(
            nnmodel, nens=nens, dfrac=dfrac, type_ens="ens", verbose=verbose
        )
        self.loss_func = loss_func
        assert hasattr(
            self.loss_func.prior_fn, "sample_anchor"
        ), "loss function must have attribute prior_fn with method sample_anchor"

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

            self.loss_func.prior_fn.sample_anchor()

            kwargs["lhist_suffix"] = f"_e{jens}"
            kwargs["loss"] = self.loss_func
            kwargs["loss_fn"] = "logposterior"
            this_learner.fit(xtrn[ind_this], ytrn[ind_this], **kwargs)
