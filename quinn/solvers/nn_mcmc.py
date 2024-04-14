#!/usr/bin/env python
"""Module for MCMC NN wrapper."""

import sys
import copy
import numpy as np
from scipy.optimize import minimize

from ..mcmc.admcmc import AMCMC
from ..mcmc.hmc import HMC
from .quinn import QUiNNBase
from ..nns.nnwrap import nn_p, NNWrap
from ..nns.losses import NegLogPost

class NN_MCMC(QUiNNBase):
    """MCMC NN wrapper class.

    Attributes:
        cmode (np.ndarray): MAP values of all parameters, size `M`.
        lpinfo (dict): Dictionary that holds likelihood computation necessary information.
        pdim (int): Dimensonality `d` of chain.
        samples (np.ndarray): MCMC samples of all parameters, size `(M,d)`.
        verbose (bool): Whether to be verbose or not.
    """

    def __init__(self, nnmodel, verbose=True):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): PyTorch NN model.
            verbose (bool, optional): Verbose or not.
        """
        super().__init__(nnmodel)
        self.verbose = verbose
        self.pdim = sum(p.numel() for p in self.nnmodel.parameters())
        print("Number of parameters:", self.pdim)

        if self.verbose:
            self.print_params(names_only=True)

        self.samples = None
        self.cmode = None

    def logpost(self, modelpars, lpinfo):
        """Function that computes log-posterior given model parameters.

        Args:
            modelpars (np.ndarray): Log-posterior input parameters.
            lpinfo (dict): Dictionary of arguments needed for likelihood computation.

        Returns:
            float: log-posterior value.
        """
        model = NNWrap(self.nnmodel)
        model.p_unflatten(modelpars)


        # Data
        ydata = lpinfo['yd']
        nd = len(ydata)

        if lpinfo['ltype'] == 'classical':
            loss = NegLogPost(self.nnmodel, nd, lpinfo['lparams']['sigma'], None)

            lpostm = - model.calc_loss(modelpars, loss, lpinfo['xd'], ydata)
        else:
            print('Likelihood type is not recognized. Exiting.')
            sys.exit()

        return lpostm

    def logpostgrad(self, modelpars, lpinfo):
        """Function that computes log-posterior given model parameters.

        Args:
            modelpars (np.ndarray): Log-posterior input parameters.
            lpinfo (dict): Dictionary of arguments needed for likelihood computation.

        Returns:
            np.ndarray: log-posterior gradient array.
        """
        model = NNWrap(self.nnmodel)
        model.p_unflatten(modelpars)


        # Data
        ydata = lpinfo['yd']
        nd = len(ydata)
        if lpinfo['ltype'] == 'classical':
            loss = NegLogPost(self.nnmodel, nd, lpinfo['lparams']['sigma'], None)
            #lpostm = - npy(loss(tch(lpinfo['xd'], rgrad=False), tch(ydata, rgrad=False), requires_grad=False))
            lpostm = - model.calc_lossgrad(modelpars, loss, lpinfo['xd'], ydata)
        else:
            print('Likelihood type is not recognized. Exiting')
            sys.exit()

        return lpostm

    def fit(self, xtrn, ytrn, zflag=True, datanoise=0.05, nmcmc=6000, param_ini=None, sampler='amcmc', sampler_params=None):
        """Fit function that perfoms MCMC on NN parameters.

        Args:
            xtrn (np.ndarray): Input data array `x` of size `(N,d)`.
            ytrn (np.ndarray): Output data array `y` of size `(N,o)`.
            zflag (bool, optional): Whether to precede MCMC with a LBFGS optimization. Default is True.
            datanoise (float, optional): Datanoise size. Defaults to 0.05.
            nmcmc (int, optional): Number of MCMC steps, `M`.
            param_ini (None, optional): Initial parameter array of size `p`. Default samples randomly.
            sampler (str, optional): Sampler method ('amcmc', 'hmc', 'mala'). Defaults to 'amcmc'.
            sampler_params (dict, optional): Sampler parameter dictionary.
        """
        shape_xtrn = xtrn.shape
        ntrn = shape_xtrn[0]
        ntrn_, outdim = ytrn.shape

        # Set dictionary info for posterior computation
        self.lpinfo = {'model': nn_p,
                  'xd': xtrn, 'yd': [y for y in ytrn],
                  'ltype': 'classical',
                  'lparams': {'sigma': datanoise}}

        if param_ini is None:
            param_ini = np.random.rand(self.pdim)  # initial parameter values
            if zflag:
                res = minimize((lambda x, fcn: -fcn(x)), param_ini, args=(self.logpost,), method='BFGS',options={'gtol': 1e-13})
                param_ini = res.x


        if sampler == 'amcmc':
            mymcmc = AMCMC(**sampler_params)
            mymcmc.setLogPost(self.logpost, None, lpinfo=self.lpinfo)
        elif sampler == 'hmc':
            mymcmc = HMC(**sampler_params)
            mymcmc.setLogPost(self.logpost, self.logpostgrad, lpinfo=self.lpinfo)


        mcmc_results = mymcmc.run(param_ini=param_ini, nmcmc=nmcmc)
        self.samples, self.cmode, pmode, acc_rate = mcmc_results['chain'], mcmc_results['mapparams'], mcmc_results['maxpost'], mcmc_results['accrate']


    def get_best_model(self, param):
        """Creates a PyTorch NN module with parameters set with a given flattened parameter array.

        Args:
            param (np.ndarray): A flattened weight parameter vector.

        Returns:
            torch.nn.Module: PyTorch NN module with the given parameters.
        """
        nnw = NNWrap(self.nnmodel)
        nnw.p_unflatten(param)

        return copy.deepcopy(nnw.nnmodel)


    def predict_MAP(self, x):
        """Predict with the max a posteriori (MAP) parameter setting.

        Args:
            x (np.ndarray): Input array of size `(N,d)`.

        Returns:
            np.ndarray: Outpur array of size `(N,o)`.
        """
        return nn_p(self.cmode, x, self.nnmodel)

    def predict_sample(self, x, param):
        """Predict with a given parameter array.

        Args:
            x (np.ndarray): Input array of size `(N,d)`.
            param (np.ndarray): Flattened weight parameter array.

        Returns:
            np.ndarray: Outpur array of size `(N,o)`.
        """
        return nn_p(param, x, self.nnmodel)

    def predict_ens(self, x, nens=10, nburn=1000):
        """Predict an ensemble of results.

        Args:
            x (np.ndarray): `(N,d)` input array.
            nens (int, optional): Number of ensemble members requested, `M`. Defaults to 10.
            nburn (int, optional): Burn-in for the MCMC chain. Defaults to 1000.

        Returns:
            np.ndarray: Array of size `(M, N, o)`, i.e. `M` random samples of `(N,o)` outputs.

        Note:
            This overloads QUiNN's base predict_ens functions
        """
        nevery = int((self.samples.shape[0]-nburn)/nens)
        for j in range(nens):
            yy = self.predict_sample(x, self.samples[nburn+j*nevery,:])
            if j == 0:
                y = np.empty((nens, yy.shape[0], yy.shape[1]))
            y[j, :, :] = yy
        return y
