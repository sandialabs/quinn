#!/usr/bin/env python
"""Base module for MCMC methods."""

import sys
import copy
import numpy as np


class MCMCBase(object):
    r"""Base class for MCMC.

    Attributes:
        logPost (callable): Log-posterior evaluator function. It has a singature of logPost(model_parameters, \**postInfo) returning a float.
        logPostGrad (callable): Log-posterior gradient evaluator function. It has a singature of logPostGrad(model_parameters, \**postInfo) returning an np.ndarray of size model_parameters. Defaults to None, for non-gradient based methods.
        postInfo (dict): Dictionary that holds auxiliary parameters for posterior evaluations.
    """

    def __init__(self):
        """Dummy instantiation."""
        self.logPost = None
        self.logPostGrad = None
        self.postInfo = {}


    def setLogPost(self, logPost, logPostGrad, **postInfo):
        r"""Setting logposterior and its gradient functions.

        Args:
            logPost (callable): Log-posterior evaluator function. It has a singature of logPost(model_parameters, \**postInfo) returning a float.
            logPostGrad (callable): Log-posterior gradient evaluator function. It has a singature of logPostGrad(model_parameters, \**postInfo) returning an np.ndarray of size model_parameters. Can be None, for non-gradient based methods.
            postInfo (dict): Dictionary that holds auxiliary parameters for posterior evaluations.
        """
        self.logPost = logPost
        self.logPostGrad = logPostGrad
        self.postInfo = postInfo



    def run(self, nmcmc, param_ini):
        """Markov chain Monte Carlo running function.

        Args:
            nmcmc (int): Number of steps.
            param_ini (np.ndarray): Initial state of the chain.

        Returns:
            dict: Dictionary of results. Keys are 'chain' (chain samples array), 'mapparams' (MAP parameters array), 'maxpost' (maximal log-post value), 'accrate' (acceptance rate), 'logpost' (log-posterior array), 'alphas' (array of Metropolis-Hastings probability ratios).
        """
        assert(self.logPost is not None)
        samples = []  # MCMC samples
        alphas = [] # Store alphas (posterior ratios)
        logposts = []  # Log-posterior values]
        na = 0                        # counter for accepted steps

        current = param_ini.copy()                # first step
        current_U = -self.logPost(current, **self.postInfo)  # Negative logposterior
        cmode = current  # MAP value (maximum a posteriori)
        pmode = -current_U  # record MCMC mode, which is where the current MAP value is achieved

        samples.append(current)
        logposts.append(-current_U)
        alphas.append(0.0)

        # Loop over MCMC steps
        for imcmc in range(nmcmc):
            current_proposal, current_K, proposed_K = self.sampler(current, imcmc)

            proposed_U = -self.logPost(current_proposal, **self.postInfo)
            proposed_H = proposed_U + proposed_K
            current_H = current_U + current_K

            mh_prob = np.exp(current_H - proposed_H)

            # Accept block
            if np.random.random_sample() < mh_prob:
                na += 1  # Acceptance counter
                current = current_proposal + 0.0
                current_U = proposed_U + 0.0
                if -current_U >= pmode:
                    pmode = -current_U
                    cmode = current + 0.0

            samples.append(current)
            alphas.append(mh_prob)
            logposts.append(-current_U)

            acc_rate = float(na) / (imcmc+1)

            if((imcmc + 2) % (nmcmc / 10) == 0) or imcmc == nmcmc - 2:
                print('%d / %d completed, acceptance rate %lg' % (imcmc + 2, nmcmc, acc_rate))

        results = {
            'chain' : np.array(samples),
            'mapparams' : cmode,
            'maxpost' : pmode,
            'accrate' : acc_rate,
            'logpost' : np.array(logposts),
            'alphas' : np.array(alphas)
            }

        return results


    def sampler(self, current, imcmc):
        """Sampler method.

        Args:
            current (np.ndarray): Current chain state.
            imcmc (int): Current chain step number.

        Raises:
            NotImplementedError: Not implemented in the base class.
        """
        raise NotImplementedError("sampler method not implemented in the base class and should be implemented in children.")
