#!/usr/bin/env python
"""Module for MALA MCMC run."""

import numpy as np
from .mcmc import MCMCBase



class MALA(MCMCBase):
    # Note: MALA is actually exactly HMC with L=1.
    # See Girolami paper https://statmodeling.stat.columbia.edu/wp-content/uploads/2010/04/RMHMC_MG_BC_SC_REV_08_04_10.pdf
    def __init__(self, epsilon=0.05):
        super().__init__()
        self.epsilon = epsilon

    def sampler(self, current, imcmc):
        assert(self.logPostGrad is not None)

        current_proposal = current.copy()
        cdim = len(current)


        p = np.random.randn(cdim)

        grad_current = self.logPostGrad(current, **self.postInfo)
        current_proposal += 0.5*self.epsilon**2 * grad_current + self.epsilon * p

        grad_prop = self.logPostGrad(current_proposal, **self.postInfo)
        current_K = np.sum(np.square(p)) / 2

        p += self.epsilon * (grad_current+grad_prop)/ 2
        proposed_K = np.sum(np.square(p)) / 2

        return current_proposal, current_K, proposed_K
