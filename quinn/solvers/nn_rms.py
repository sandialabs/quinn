#!/usr/bin/env python
"""Module for RMS NN wrapper."""

import numpy as np
import torch
from torch.optim import SGD, Adam
from torch import randperm

from .nn_ens import NN_Ens
from ..ens.learner import Learner
from ..nns.tchutils import npy, tch


class NN_RMS(NN_Ens):
    """RMS Ensemble NN Wrapper.
    Pearce, Tim, Felix Leibfried, and Alexandra Brintrup. "Uncertainty in
    neural networks: Approximately bayesian ensembling." International conference
    on artificial intelligence and statistics. PMLR, 2020.
    Attributes:
        - loss_func : Loss function over which the mode is optimized. I takes,
        a NN model, x and y data, and requires_grad as input.
    """

    def __init__(self, nnmodel, datanoise=0.1, priorsigma=1.0, **kwargs):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): NNWrapper class.
            datanoise (float): Data noise size
            **kwargs: Description
        """
        super().__init__(nnmodel, **kwargs)
        self.datanoise = datanoise
        self.priorsigma = priorsigma
        self.nparams = sum(p.numel() for p in self.nnmodel.parameters())

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
            #kwargs["loss"] = torch.nn.MSELoss(reduction='mean') #Loss_Gaussian(self.nnmodel, 1.1)
            kwargs["loss_fn"] = "logpost"
            kwargs["datanoise"] = self.datanoise
            kwargs["priorparams"] = {'sigma': self.priorsigma, 'anchor': torch.randn(size=(self.nparams,)) * self.priorsigma}

            this_learner.fit(xtrn[ind_this], ytrn[ind_this], **kwargs)
