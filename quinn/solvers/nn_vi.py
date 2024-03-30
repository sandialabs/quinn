#!/usr/bin/env python
"""Module for the Variational Inference (VI) NN wrapper."""

import copy
import math
import torch

from ..vi.bnet import BNet

from ..solvers.quinn import QUiNNBase
from ..nns.tchutils import npy, tch, print_nnparams
from ..nns.nnfit import nnfit

class NN_VI(QUiNNBase):
    """VI wrapper class.

    Attributes:
        bmodel (BNet): The underlying Bayesian model.
        trained (bool): Whether the model is trained or not.
        verbose (bool): Whether to be verbose or not.
        best_epoch (int): The epoch where the best model is reached.
        best_loss (float): The best loss value.
        best_model (torch.nn.Module): The best PyTorch NN model found during training.
    """

    def __init__(self, nnmodel, verbose=False, pi=0.5, sigma1=1.0, sigma2=1.0,
                       mu_init_lower=-0.2, mu_init_upper=0.2,
                       rho_init_lower=-5.0, rho_init_upper=-4.0 ):
        """Instantiate a VI wrapper object.

        Args:
            nnmodel (torch.nn.Module): The underlying PyTorch NN model.
            verbose (bool, optional): Whether to print out model details or not.
            pi (float): Weight of the first gaussian. The second weight is 1-pi.
            sigma1 (float): Standard deviation of the first gaussian. Can also be a scalar torch.Tensor.
            sigma2 (float): Standard deviation of the second gaussian. Can also be a scalar torch.Tensor.
            mu_init_lower (float) : Initialization of mu lower value
            mu_init_upper (float) : Initialization of mu upper value
            rho_init_lower (float) : Initialization of rho lower value
            rho_init_upper (float) : Initialization of rho upper value

        """
        super().__init__(nnmodel)

        self.bmodel = BNet(nnmodel,pi=pi,sigma1=sigma1,sigma2=sigma2,
                                   mu_init_lower=mu_init_lower, mu_init_upper=mu_init_upper,
                                   rho_init_lower=rho_init_lower, rho_init_upper=rho_init_upper )
        try:
            self.device = nnmodel.device
        except AttributeError:
            self.device = 'cpu'

        self.bmodel.to(self.device)
        self.verbose = verbose

        if self.verbose:
            print("=========== Deterministic model parameters ================")
            self.print_params(names_only=True)
            print("=========== Variational model parameters ==================")
            print_nnparams(self.bmodel, names_only=True)
            print("===========================================================")

    def fit(self, xtrn, ytrn, val=None,
            nepochs=600, lrate=0.01, batch_size=None, freq_out=100,
            freq_plot=1000, wd=0,
            cooldown=100,
            factor=0.95,
            nsam=1,scheduler_lr=None, datanoise=0.05):
        """Fit function to train the network.

        Args:
            xtrn (np.ndarray): Training input array of size `(N,d)`.
            ytrn (np.ndarray): Training output array of size `(N,o)`.
            val (tuple, optional): `x,y` tuple of validation points. Default uses the training set for validation.
            nepochs (int, optional): Number of epochs.
            lrate (float, optional): Learning rate or learning rate schedule factor. Default is 0.01.
            batch_size (int, optional): Batch size. Default is None, i.e. single batch.
            freq_out (int, optional): Frequency, in epochs, of screen output. Defaults to 100.
            nsam (int, optional): Number of samples for ELBO computation. Defaults to 1.
            scheduler_lr(str,optional): Learning rate is adjusted during training according to the ReduceLROnPlateau method from pytTorch. 
            datanoise (float, optional): Datanoise for ELBO computation. Defaults to 0.05.
            freq_out (int, optional): Frequency, in epochs, of screen output. Defaults to 100.
            wd (float, optional): Optional weight decay (L2 regularization) parameter.
            cooldown (int, optional) : cooldown in ReduceLROnPlateau
            factor (float, optional) : factor in ReduceLROnPlateau
        """

        shape_xtrn = xtrn.shape
        ntrn = shape_xtrn[0]
        ntrn_, outdim = ytrn.shape
        assert(ntrn==ntrn_)

        if batch_size is None or batch_size > ntrn:
            batch_size = ntrn

        if batch_size == 1:
            num_batches = ntrn
        else:
            num_batches = (ntrn + 1) // batch_size

        self.bmodel.loss_params = [datanoise, nsam, num_batches]

        fit_info = nnfit(self.bmodel, xtrn, ytrn, val=val,
                         loss_xy=self.bmodel.viloss,
                         lrate=lrate, batch_size=batch_size,
                         nepochs=nepochs,
                         wd=wd,
                         cooldown=cooldown,
                         factor=factor,
                         freq_plot=freq_plot,
                         scheduler_lr=scheduler_lr, freq_out=freq_out)
        self.best_model = fit_info['best_nnmodel']
        self.trained = True

    def predict_sample(self, x):
        """Predict a single sample.

        Args:
            x (np.ndarray): Input array `x` of size `(N,d)`.

        Returns:
            np.ndarray: Output array `x` of size `(N,o)`.

        Note:
            predict_ens() from the parent class will use this to sample an ensemble.
        """
        assert(self.trained)
        device = self.best_model.device
        y = npy(self.best_model(tch(x, rgrad=False,device=device), sample=True))

        return y

######################################################################
######################################################################
######################################################################

