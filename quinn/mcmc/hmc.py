#!/usr/bin/env python
"""Module for HMC run."""

import numpy as np
from .mcmc import MCMCBase


class HMC(MCMCBase):
    # Implementation based on Neal, 2011. https://arxiv.org/pdf/1206.1901.pdf
    def __init__(self, epsilon=0.05, L=3):
        super().__init__()
        self.epsilon = epsilon
        self.L = L

    def sampler(self, current, imcmc):
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
        p = -p # TODO: is this really necessary?

        # Evaluate kinetic and potential energies
        proposed_K = np.sum(np.square(p)) / 2

        return current_proposal, current_K, proposed_K

