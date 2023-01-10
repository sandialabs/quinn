#!/usr/bin/env python
"""Module for Adaptive MCMC run."""

import numpy as np


class AMCMC():
    r"""Adaptive MCMC class.

    Attributes:
        nmcmc (int): Number of MCMC steps.
        param_ini (np.ndarray): Initial parameter array of size `p`.
        cov_ini (np.ndarray): Initial covariance array of size `(p,p)`.
        gamma (float): Proposal jump size factor :math:`\gamma`.
        t0 (int): Step where adaptivity begins.
        tadapt (int): Adapt/update covariance every `tadapt` steps.
    """
    def __init__(self):
        """Initialization.
        """
        super().__init__()

    def setParams(self,
                  param_ini, cov_ini,
                  nmcmc=10000, gamma=0.1,
                  t0=100, tadapt=1000):
        r"""Set parameters of adaptive MCMC.

        Args:
            nmcmc (int): Number of MCMC steps.
            param_ini (np.ndarray): Initial parameter array of size `p`.
            cov_ini (np.ndarray): Initial covariance array of size `(p,p)`.
            gamma (float, optional): Proposal jump size factor :math:`\gamma`. Defaults to 0.1.
            t0 (int, optional): Step where adaptivity begins. Defaults to 100.
            tadapt (int, optional): Adapt/update covariance every `tadapt` steps. Defaults to 1000.
        """
        self.param_ini = param_ini
        self.cov_ini = cov_ini
        self.t0 = t0
        self.tadapt = tadapt
        self.gamma = gamma
        self.nmcmc = nmcmc

    def run(self, logpostFcn, **postInfo):
        """Adaptive Markov chain Monte Carlo.

        Args:
            logpostFcn (callable): Log-posterior evaluator function.
            **postInfo: All information needed for log-posterior evaluation.

        Returns:
            dict: Dictionary of results. Keys are 'chain' (chain samples array), 'mapparams' (MAP parameters array), 'maxpost' (maximal log-post value), 'accrate' (acceptance rate)
        """
        cdim = self.param_ini.shape[0]            # chain dimensionality
        cov = np.zeros((cdim, cdim))   # covariance matrix
        samples = np.zeros((self.nmcmc, cdim))  # MCMC samples
        alphas = np.zeros((self.nmcmc,))  # Store alphas (posterior ratios)
        logposts = np.zeros((self.nmcmc,))  # Log-posterior values
        na = 0                        # counter for accepted steps
        sigcv = self.gamma * 2.4**2 / cdim
        samples[0] = self.param_ini                  # first step
        p1 = -logpostFcn(samples[0], **postInfo)  # NEGATIVE logposterior
        pmode = p1  # record MCMC 'mode', which is the current MAP value (maximum posterior)
        cmode = samples[0]  # MAP sample
        acc_rate = 0.0  # Initial acceptance rate

        # Loop over MCMC steps
        for k in range(self.nmcmc - 1):

            # Compute covariance matrix
            if k == 0:
                Xm = samples[0]
            else:
                Xm = (k * Xm + samples[k]) / (k + 1.0)
                rt = (k - 1.0) / k
                st = (k + 1.0) / k**2
                cov = rt * cov + st * np.dot(np.reshape(samples[k] - Xm, (cdim, 1)), np.reshape(samples[k] - Xm, (1, cdim)))
            if k == 0:
                propcov = self.cov_ini
            else:
                if (k > self.t0) and (k % self.tadapt == 0):
                    propcov = sigcv * (cov + 10**(-8) * np.identity(cdim))

            # Generate proposal candidate
            u = np.random.multivariate_normal(samples[k], propcov)
            p2 = -logpostFcn(u, **postInfo)
            #print(p1, p2)
            pr = np.exp(p1 - p2)
            alphas[k + 1] = pr
            logposts[k + 1] = -p2
            # Accept...
            if np.random.random_sample() <= pr:
                samples[k + 1] = u
                na = na + 1  # Acceptance counter
                p1 = p2
                if p1 <= pmode:
                    pmode = p1
                    cmode = samples[k + 1]
            # ... or reject
            else:
                samples[k + 1] = samples[k]

            acc_rate = float(na) / (k+1)

            if((k + 2) % (self.nmcmc / 10) == 0) or k == self.nmcmc - 2:
                print('%d / %d completed, acceptance rate %lg' % (k + 2, self.nmcmc, acc_rate))

        mcmc_results = {'chain' : samples, 'mapparams' : cmode,
        'maxpost' : pmode, 'accrate' : acc_rate, 'logpost' : logposts, 'alphas' : alphas}

        return mcmc_results
