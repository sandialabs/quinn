#!/usr/bin/env python
"""Module for MCMC NN wrapper."""

import sys
import copy
import numpy as np
from scipy.optimize import minimize

from .admcmc import AMCMC
from ..quinn import QUiNNBase
from ..nns.nnwrap import nn_p, NNWrap


class MCMC_NN(QUiNNBase):
    """MCMC NN wrapper class.
    ----------
    Attributes:
        - lpinfo (dict): Dictionary that holds likelihood computation necessary information.
        - pdim (int): Dimensonality `d` of chain.
        - verbose (bool): Whether to be verbose or not.
        - samples (np.ndarray): MCMC samples of all parameters, size `(M,d)`.
        - cmode (np.ndarray): MAP values of all parameters, size `M`.
        - sampler: sampling algorithm.
        - log_post: log posterior
        - nnmodel: NN model defined as an NNWrapp class instance.
    """

    def __init__(self, nnmodel, verbose=True, sampler=None, log_post=None):
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

        self.nnmodel = nnmodel
        self.sampler = sampler
        self.log_posterior = log_post

    def logpost(self, modelpars):
        """Function that computes log-posterior given model parameters.
        This method allows to compute the posterior for gaussian likelihood
        and prior if no log_Posterior method is provided when creating the
        MCMC_NN instance.
        ----------
        Args:
            modelpars (np.ndarray): Log-posterior input parameters.
        ----------
        Returns:
            float: log-posterior value.
        """
        if self.log_posterior is None:
            # Model prediction
            ypred = self.lpinfo["model"](
                modelpars, self.lpinfo["xd"], self.lpinfo["otherpars"]
            )
            # Data
            ydata = self.lpinfo["yd"]
            nd = len(ydata)
            if self.lpinfo["ltype"] == "classical":
                lpostm = 0.0
                for i in range(nd):
                    for yy in ydata[i]:
                        lpostm -= (
                            0.5
                            * np.sum((ypred[i] - yy) ** 2)
                            / self.lpinfo["lparams"]["sigma"] ** 2
                        )
                        lpostm -= 0.5 * np.log(2 * np.pi)
                        lpostm -= np.log(self.lpinfo["lparams"]["sigma"])
            else:
                print("Likelihood type is not recognized. Exiting")
                sys.exit()

            return lpostm
        else:
            self.nnmodel.p_unflatten(modelpars)

            return self.log_posterior(
                self.nnmodel, self.lpinfo["xd"], self.lpinfo["yd"], requires_grad=False
            )

    def fit(
        self,
        xtrn,
        ytrn,
        zflag=True,
        datanoise=0.05,
        nmcmc=6000,
        gamma=0.1,
        param_ini=None,
        cov_ini=None,
        t0=100,
        tadapt=1000,
    ):
        r"""Fit function that perfoms MCMC on NN parameters.

        Args:
            xtrn (np.ndarray): Input data array `x` of size `(N,d)`.
            ytrn (np.ndarray): Output data array `y` of size `(N,o)`.
            zflag (bool, optional): Whether to precede MCMC with a LBFGS optimization. Default is True.
            datanoise (float, optional): Datanoise size. Defaults to 0.05.
            nmcmc (int, optional): Number of MCMC steps, `M`.
            gamma (float, optional): Proposal jump size factor :math:`\gamma`. Defaults to 0.1.
            param_ini (None, optional): Initial parameter array of size `p`. Default samples randomly.
            cov_ini (None, optional): Initial covariance array of size `(p,p)`. Default generates initial diagonal covariance that is a 0.01 factor of initial parameters (with a cushion to avoid zero variance).
            t0 (int, optional): Step where adaptivity begins. Defaults to 100.
            tadapt (int, optional): Adapt/update covariance every `tadapt` steps. Defaults to 1000.
        """
        shape_xtrn = xtrn.shape
        ntrn = shape_xtrn[0]
        ntrn_, outdim = ytrn.shape

        # Set dictionary info for posterior computation
        self.lpinfo = {
            "model": nn_p,
            "xd": xtrn,
            "otherpars": self.nnmodel,
            "yd": [y for y in ytrn],
            "ltype": "classical",
            "lparams": {"sigma": datanoise},
        }

        if param_ini is None:
            param_ini = np.random.rand(self.pdim)  # initial parameter values
            if zflag:
                res = minimize(
                    (lambda x, fcn: -fcn(x)),
                    param_ini,
                    args=(self.logpost,),
                    method="BFGS",
                    options={"gtol": 1e-13},
                )
                param_ini = res.x
        if cov_ini is None:
            cov_ini = np.diag(0.01 * np.abs(param_ini + 1.0e-3))  # initial covariance

        my_amcmc = AMCMC()
        my_amcmc.setParams(
            param_ini, cov_ini, t0=t0, tadapt=tadapt, gamma=gamma, nmcmc=nmcmc
        )

        mcmc_results = my_amcmc.run(self.logpost)
        self.samples, self.cmode, pmode, acc_rate = (
            mcmc_results["chain"],
            mcmc_results["mapparams"],
            mcmc_results["maxpost"],
            mcmc_results["accrate"],
        )

    def get_best_model(self, param):
        """Creates a PyTorch NN module with parameters set to a given flattened parameter array.

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
        """Summary

        Args:
            x (np.ndarray): `(N,d)` input array.
            nens (int, optional): Number of ensemble members requested, `M`. Defaults to 10.
            nburn (int, optional): Burn-in for the MCMC chain. Defaults to 1000.

        Returns:
            np.ndarray: Array of size `(M, N, o)`, i.e. `M` random samples of `(N,o)` outputs.

        Note:
            This overloads QUiNN's base predict_ens functions
        """
        nevery = int((self.samples.shape[0] - nburn) / nens)
        for j in range(nens):
            yy = self.predict_sample(x, self.samples[nburn + j * nevery, :])
            if j == 0:
                y = np.empty((nens, yy.shape[0], yy.shape[1]))
            y[j, :, :] = yy
        return y
