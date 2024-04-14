#!/usr/bin/env python
"""Module for RMS NN wrapper."""

import torch
import numpy as np

from .nn_ens import NN_Ens


class NN_RMS(NN_Ens):
    """RMS Ensemble NN Wrapper. For details of the method, see :cite:t:`pearce:2018`.

    Attributes:
        datanoise (float): Data noise standard deviation.
        nparams (int): Number of model parameters.
        priorsigma (float): Prior standard deviation.
    """

    def __init__(self, nnmodel, datanoise=0.1, priorsigma=1.0, **kwargs):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): NNWrapper class.
            datanoise (float, optional): Data noise standard deviation. Defaults to 0.1.
            priorsigma (float, optional): Gaussian prior standard deviation. Defaults to 1.0.
            **kwargs: Any keyword argument that :meth:`..nns.nnfit.nnfit` takes.
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
            **kwargs (dict): Any keyword argument that :meth:`..nns.nnfit.nnfit` takes.
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
