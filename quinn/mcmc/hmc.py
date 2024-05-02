#!/usr/bin/env python
r"""Module for Hamiltonian MCMC (HMC) sampling. For details of the method, see Chapter 5 of :cite:t:`brooks:2011` or https://arxiv.org/pdf/1206.1901.pdf"""

import numpy as np
from .mcmc import MCMCBase


class HMC(MCMCBase):
    """Hamiltonian MCMC class.

    Attributes:
        epsilon (float): Step size of the method.
        L (int): Number of steps in the Hamiltonian integrator.
    """

    def __init__(self, epsilon=0.05, L=3):
        """Initialization.

        Args:
            epsilon (float, optional): Step size of the method. Defaults to 0.05.
            L (int, optional): Number of steps in the Hamiltonian integrator. Defaults to 3.
        """
        super().__init__()
        self.epsilon = epsilon
        self.L = L

    def sampler(self, current, imcmc):
        """Sampler method.

        Args:
            current (np.ndarray): Current chain state.
            imcmc (int): Current chain step number.

        Returns:
            tuple(current_proposal, current_K, proposed_K): A tuple containing the current proposal sample, current and proposed kinetic energies.
        """
        assert(self.logPostGrad is not None)

        current_proposal = current.copy()
        cdim = len(current)


        p = np.random.randn(cdim)
        current_K = np.sum(np.square(p)) / 2

        # Make a half step for momentum at the beginning (Leapfrog Method step starts here)

        p += self.epsilon * self.logPostGrad(current_proposal, **self.postInfo) / 2

        for jj in range(self.L):
            # Make a full step for the position
            current_proposal += self.epsilon * p

            # Make a full step for the momentum, expecpt at the end of the trajectory

            if jj != self.L - 1:
                p += self.epsilon * self.logPostGrad(current_proposal, **self.postInfo)

        # Make a half step for momentum at the end (Leapfrog Method step ends here)
        p += self.epsilon* self.logPostGrad(current_proposal, **self.postInfo) / 2


        # Negate momentum to make proposal symmetric
        # This is really not necessary, but we kept it per original paper
        p = -p

        # Evaluate kinetic and potential energies
        proposed_K = np.sum(np.square(p)) / 2

        return current_proposal, current_K, proposed_K

