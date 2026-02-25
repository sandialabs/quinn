#!/usr/bin/env python
"""Test script for random variable classes."""

import math
import torch
import numpy as np
from quinn.rvar.rvs import Gaussian_1d, GMM2_1d, MVN


def test_gaussian_1d_rho_sample_shape():
    # Sample shape should match mu shape
    mu = torch.zeros(5)
    rho = torch.zeros(5)
    rv = Gaussian_1d(mu, rho=rho)

    sample = rv.sample()
    assert sample.shape == mu.shape


def test_gaussian_1d_logsigma_sample_shape():
    # Sample with logsigma parameterization
    mu = torch.zeros(3)
    logsigma = torch.zeros(3)
    rv = Gaussian_1d(mu, logsigma=logsigma)

    sample = rv.sample()
    assert sample.shape == mu.shape


def test_gaussian_1d_log_prob_scalar():
    # log_prob of mean should be the maximum
    mu = torch.tensor([0.0])
    rho = torch.tensor([0.0])  # sigma = log(1 + exp(0)) = log(2) ~ 0.693
    rv = Gaussian_1d(mu, rho=rho)

    lp_at_mean = rv.log_prob(mu)
    lp_away = rv.log_prob(torch.tensor([5.0]))

    assert lp_at_mean > lp_away


def test_gaussian_1d_log_prob_formula():
    # Verify log-prob matches manual calculation with rho parameterization
    mu = torch.tensor([1.0])
    rho = torch.tensor([0.0])
    rv = Gaussian_1d(mu, rho=rho)

    sigma = torch.log1p(torch.exp(rho))
    x = torch.tensor([1.5])
    expected = (-math.log(math.sqrt(2 * math.pi))
                - torch.log(sigma)
                - ((x - mu) ** 2) / (2 * sigma ** 2)).sum()

    assert torch.isclose(rv.log_prob(x), expected)


def test_gaussian_1d_logsigma_log_prob():
    # Verify log-prob with logsigma parameterization
    mu = torch.tensor([0.0])
    logsigma = torch.tensor([0.0])  # sigma = exp(0) = 1, i.e., standard normal
    rv = Gaussian_1d(mu, logsigma=logsigma)

    x = torch.tensor([0.0])
    expected = -0.5 * math.log(2 * math.pi)  # log N(0|0,1)

    assert torch.isclose(rv.log_prob(x), torch.tensor(expected), atol=1e-6)


def test_gaussian_1d_samples_statistics():
    # Sampling many times should give mean close to mu
    mu = torch.tensor([3.0])
    rho = torch.tensor([0.0])
    rv = Gaussian_1d(mu, rho=rho)

    samples = torch.stack([rv.sample() for _ in range(5000)])
    assert abs(samples.mean().item() - 3.0) < 0.1


def test_gmm2_1d_log_prob():
    # GMM2 log_prob should be computable
    rv = GMM2_1d(pi=0.5, sigma1=1.0, sigma2=2.0)

    x = torch.tensor([0.0])
    lp = rv.log_prob(x)

    assert torch.isfinite(lp)


def test_gmm2_1d_log_prob_at_zero():
    # At x=0, GMM with zero means should give log(pi * N(0|0,s1) + (1-pi) * N(0|0,s2))
    pi = 0.5
    sigma1, sigma2 = 1.0, 2.0
    rv = GMM2_1d(pi=pi, sigma1=sigma1, sigma2=sigma2)

    x = torch.tensor([0.0])
    lp = rv.log_prob(x)

    p1 = 1.0 / (math.sqrt(2 * math.pi) * sigma1)
    p2 = 1.0 / (math.sqrt(2 * math.pi) * sigma2)
    expected = math.log(pi * p1 + (1 - pi) * p2)

    assert abs(lp.item() - expected) < 1e-5


def test_gmm2_1d_symmetry():
    # GMM with zero means should be symmetric: log_prob(x) == log_prob(-x)
    rv = GMM2_1d(pi=0.5, sigma1=1.0, sigma2=2.0)

    x = torch.tensor([1.5])
    assert torch.isclose(rv.log_prob(x), rv.log_prob(-x))


def test_mvn_sample_shape():
    # MVN sample shape should be (num_samples, dim)
    dim = 4
    mean = torch.zeros(dim)
    cov = torch.eye(dim)
    rv = MVN(mean, cov)

    samples = rv.sample(num_samples=100)
    assert samples.shape == (100, dim)


def test_mvn_log_prob():
    # log_prob at the mean should be finite
    dim = 3
    mean = torch.zeros(dim)
    cov = torch.eye(dim)
    rv = MVN(mean, cov)

    lp = rv.log_prob(mean)
    assert torch.isfinite(lp)


def test_mvn_sample_mean():
    # Mean of many samples should be close to the true mean
    dim = 2
    mean = torch.tensor([1.0, -2.0])
    cov = torch.eye(dim)
    rv = MVN(mean, cov)

    samples = rv.sample(num_samples=5000)
    sample_mean = samples.mean(dim=0)

    assert torch.allclose(sample_mean, mean, atol=0.1)


if __name__ == '__main__':
    test_gaussian_1d_rho_sample_shape()
    test_gaussian_1d_logsigma_sample_shape()
    test_gaussian_1d_log_prob_scalar()
    test_gaussian_1d_log_prob_formula()
    test_gaussian_1d_logsigma_log_prob()
    test_gaussian_1d_samples_statistics()
    test_gmm2_1d_log_prob()
    test_gmm2_1d_log_prob_at_zero()
    test_gmm2_1d_symmetry()
    test_mvn_sample_shape()
    test_mvn_log_prob()
    test_mvn_sample_mean()
