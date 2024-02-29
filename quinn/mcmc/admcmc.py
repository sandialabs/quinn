#!/usr/bin/env python
"""Module for Adaptive MCMC run."""

import numpy as np
from .mcmc import MCMCBase

class AMCMC(MCMCBase):
    r"""Adaptive MCMC class.

    Attributes:
        cov_ini (np.ndarray): Initial covariance array of size `(p,p)`.
        gamma (float): Proposal jump size factor :math:`\gamma`.
        t0 (int): Step where adaptivity begins.
        tadapt (int): Adapt/update covariance every `tadapt` steps.
    """
    def __init__(self, cov_ini=None, gamma=0.1, t0=100, tadapt=1000):
        """Initialization.
        """
        super().__init__()

        self.cov_ini = cov_ini
        self.t0 = t0
        self.tadapt = tadapt
        self.gamma = gamma

    def sampler(self, current, imcmc):
        current_proposal = current.copy()
        cdim = len(current)

        # Compute covariance matrix
        if imcmc == 0:
            self._Xm = current.copy()
            self._cov = np.zeros((cdim, cdim))
        else:
            self._Xm = (imcmc * self._Xm + current) / (imcmc + 1.0)
            rt = (imcmc - 1.0) / imcmc
            st = (imcmc + 1.0) / imcmc**2
            self._cov = rt * self._cov + st * np.dot(np.reshape(current - self._Xm, (cdim, 1)), np.reshape(current - self._Xm, (1, cdim)))

        if imcmc == 0:
            if self.cov_ini is not None:
                self.propcov = self.cov_ini
            else:
                self.propcov = 0.01 + np.diag(0.09*np.abs(current))
        elif (imcmc > self.t0) and (imcmc % self.tadapt == 0):
                self.propcov = (self.gamma * 2.4**2 / cdim) * (self._cov + 10**(-8) * np.eye(cdim))

        # Generate proposal candidate
        current_proposal += np.random.multivariate_normal(np.zeros(cdim,), self.propcov)
        proposed_K = 0.0
        current_K = 0.0

        return current_proposal, current_K, proposed_K


