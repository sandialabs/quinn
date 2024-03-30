#!/usr/bin/env python
"""Module for random variable classes."""

import math
import torch



class RV(torch.nn.Module):
    """Parent class for random variables."""

    def __init__(self):
        """Initialization."""
        super().__init__()


    def sample(self, num_samples=1):
        """Sampling function.

        Raises:
            NotImplementedError: Expected to be implemented in children classes
        """
        raise NotImplementedError

    def log_prob(self, x):
        """Evaluate log-probability.

        Raises:
            NotImplementedError: Expected to be implemented in children classes
        """
        raise NotImplementedError

########################################
########################################
########################################


class MVN(RV):
    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.distribution = torch.distributions.MultivariateNormal(self.mean, self.cov)

    def sample(self, num_samples):
        return self.distribution.sample((num_samples,))

    def log_prob(self, x):
        return self.distribution.log_prob(x)


########################################
########################################
########################################

class Gaussian_1d(RV):
    r"""One dimensional gaussian random variable.

    Attributes:
        mu (torch.Tensor): Mean tensor.
        rho (torch.Tensor): :math:`\rho` tensor, where :math:`\rho=\log{(e^\sigma-1)}` or, equivalently, :math:`\sigma=\log{(1+e^\rho)}`. This is the parameterization used in :cite:t:`blundell:2015`.
        logsigma (torch.Tensor): A more typical parameterization of the gaussian standard deviation :math:`\sigma` via its natural logarithm :math:`\log{\sigma}`.
        normal (torch.distributions.Normal): The underlying torch-based normal random variable.
    """

    def __init__(self, mu, rho=None, logsigma=None):
        r"""Instantiate the random variable.

        Args:
            mu (torch.Tensor): Mean tensor.
            rho (torch.Tensor, optional): Parameterization that relates to standard deviation as :math:`\sigma=\log{(1+e^\rho)}`.
            logsigma (torch.Tensor, optional): Parameterization that relates to standard deviation as :math:`\log{\sigma}`.

        Note:
            Exactly one of rho or logsigma should be not None.
        Note:
            rho and logsigma, if not None, should have same shape as mu.
        """
        super().__init__()

        self.mu = mu
        self.rho = None
        self.logsigma = None

        if rho is not None:
            assert(logsigma is None)
            assert(rho.shape==self.mu.shape)
            self.rho = rho
        else:
            assert(logsigma is not None)
            assert(logsigma.shape==self.mu.shape)
            self.logsigma = logsigma

        self.normal = torch.distributions.Normal(0,1)


    def sample(self):
        r"""Sampling function.

        Returns:
            torch.Tensor: A torch tensor of the same shape as :math:`\mu` and :math:`\rho` (or `\log{\sigma}`).
        """
        if self.rho is not None:
            sigma = torch.log1p(torch.exp(self.rho))
        else:
            sigma = torch.exp(self.logsigma)
        # FIXME: compute epsilon with pyTorch to avoid transfer data from host to device
        epsilon = self.normal.sample(sigma.size()).to(self.mu.device)
        return self.mu + sigma * epsilon

    def log_prob(self, x):
        """Evaluate the natural logarithm of the probability density function.

        Args:
            x (torch.Tensor): An input tensor of same shape (or broadcastable to) as mu and rho (logsigma).

        Returns:
            float: scalar torch.Tensor.
        """
        if self.rho is not None:
            sigma = torch.log1p(torch.exp(self.rho))
        else:
            sigma = torch.exp(self.logsigma)

        logprob = (-math.log(math.sqrt(2 * math.pi))
                - torch.log(sigma)
                - ((x - self.mu) ** 2) / (2 * sigma ** 2)).sum()
        return logprob

########################################
########################################
########################################

class GMM2_1d(RV):
    """One dimensional gaussian mixture random variable with two gaussians that have zero mean and user-defined standard deviations.

    Attributes:
        pi (float): Weight of the first gaussian. The second weight is 1-pi.
        sigma1 (float): Standard deviation of the first gaussian. Can also be a scalar torch.Tensor.
        sigma2 (float): Standard deviation of the second gaussian. Can also be a scalar torch.Tensor.
        normal1 (torch.distributions.Normal): The underlying torch-based normal random variable for the first gaussian.
        normal2 (torch.distributions.Normal): The underlying torch-based normal random variable for the second gaussian.
    """

    def __init__(self, pi, sigma1, sigma2):
        """Instantiation of the GMM2 object.

        Args:
            pi (float): Weight of the first gaussian. The second weight is 1-pi.
            sigma1 (float): Standard deviation of the first gaussian. Can also be a scalar torch.Tensor.
            sigma2 (float): Standard deviation of the second gaussian. Can also be a scalar torch.Tensor.
        """
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.normal1 = torch.distributions.Normal(0,sigma1)
        self.normal2 = torch.distributions.Normal(0,sigma2)

    def log_prob(self, x):
        """Evaluate the natural logarithm of the probability density function.

        Args:
            x (torch.Tensor): An input tensor.

        Returns:
            float: scalar torch.Tensor.
        """

        prob1 = torch.exp(self.normal1.log_prob(x))
        prob2 = torch.exp(self.normal2.log_prob(x))
        logprob = (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()

        return logprob
