#!/usr/bin/env python
r"""Module for Metropolis Adjusted Langevin (MALA) MCMC sampling. For details of the method, see :cite:t:`girolami:2011` or https://arxiv.org/pdf/1206.1901.pdf"""

import numpy as np
from .mcmc import MCMCBase


class MALA(MCMCBase):
    """MALA MCMC class.

    Attributes:
        epsilon (float): Step size of the method.
    """

    def __init__(self, epsilon=0.05):
        """Initialization.

        Args:
            epsilon (float, optional): Step size of the method. Defaults to 0.05.
        """
        super().__init__()
        self.epsilon = epsilon

    def sampler(self, current, imcmc):
        """Sampler method.

        Args:
            current (np.ndarray): Current chain state.
            imcmc (int): Current chain step number.

        Returns:
            tuple(current_proposal, current_K, proposed_K): A tuple containing the current proposal sample, current and proposed kinetic energies.

        Note: When the dust settles, MALA is actually exactly HMC with L=1.
        """
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
