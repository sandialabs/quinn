#!/usr/bin/env python
"""Test script for MCMC samplers."""

import numpy as np
from quinn.mcmc.admcmc import AMCMC
from quinn.mcmc.hmc import HMC
from quinn.mcmc.mala import MALA


def make_gaussian_logpost(mean, cov):
    """Create a simple Gaussian log-posterior for testing."""
    cov_inv = np.linalg.inv(cov)

    def logpost(x):
        diff = x - mean
        return -0.5 * diff @ cov_inv @ diff

    def logpost_grad(x):
        diff = x - mean
        return -cov_inv @ diff

    return logpost, logpost_grad


def test_amcmc_creation():
    # AMCMC should be creatable with default parameters
    sampler = AMCMC()
    assert sampler.gamma == 0.1
    assert sampler.t0 == 100
    assert sampler.tadapt == 1000


def test_amcmc_run():
    # AMCMC should sample from a 2D Gaussian
    np.random.seed(42)
    mean = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])

    logpost, logpost_grad = make_gaussian_logpost(mean, cov)

    sampler = AMCMC(gamma=0.5, t0=50, tadapt=100)
    sampler.setLogPost(logpost, None)

    param_ini = np.zeros(2)
    results = sampler.run(3000, param_ini)

    assert 'chain' in results
    assert 'mapparams' in results
    assert 'accrate' in results
    assert results['chain'].shape[1] == 2

    # MAP should be close to the mean
    assert np.allclose(results['mapparams'], mean, atol=0.5)


def test_amcmc_acceptance_rate():
    # AMCMC should have reasonable acceptance rate
    np.random.seed(42)
    mean = np.array([0.0])
    cov = np.array([[1.0]])

    logpost, _ = make_gaussian_logpost(mean, cov)

    sampler = AMCMC(gamma=0.5)
    sampler.setLogPost(logpost, None)

    results = sampler.run(2000, np.array([0.0]))

    # Acceptance rate should be between 0.1 and 0.9
    assert 0.05 < results['accrate'] < 0.95


def test_amcmc_chain_shape():
    # Chain should have correct shape
    np.random.seed(42)
    ndim = 3
    mean = np.zeros(ndim)
    cov = np.eye(ndim)

    logpost, _ = make_gaussian_logpost(mean, cov)

    sampler = AMCMC(gamma=0.5)
    sampler.setLogPost(logpost, None)

    nmcmc = 500
    results = sampler.run(nmcmc, np.zeros(ndim))

    assert results['chain'].shape == (nmcmc + 1, ndim)
    assert results['logpost'].shape == (nmcmc + 1,)
    assert results['alphas'].shape == (nmcmc + 1,)


def test_hmc_run():
    # HMC should sample from a 2D Gaussian
    np.random.seed(42)
    mean = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])

    logpost, logpost_grad = make_gaussian_logpost(mean, cov)

    sampler = HMC(epsilon=0.1, L=10)
    sampler.setLogPost(logpost, logpost_grad)

    param_ini = np.zeros(2)
    results = sampler.run(1000, param_ini)

    assert results['chain'].shape[1] == 2
    # MAP should be close to the mean
    assert np.allclose(results['mapparams'], mean, atol=0.5)


def test_mala_run():
    # MALA should sample from a 2D Gaussian
    np.random.seed(42)
    mean = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])

    logpost, logpost_grad = make_gaussian_logpost(mean, cov)

    sampler = MALA(epsilon=0.1)
    sampler.setLogPost(logpost, logpost_grad)

    param_ini = np.zeros(2)
    results = sampler.run(1000, param_ini)

    assert results['chain'].shape[1] == 2


def test_amcmc_logpost_stored():
    # Log-posterior values should be stored
    np.random.seed(42)
    mean = np.array([0.0, 0.0])
    cov = np.eye(2)

    logpost, _ = make_gaussian_logpost(mean, cov)

    sampler = AMCMC(gamma=0.5)
    sampler.setLogPost(logpost, None)

    results = sampler.run(500, np.zeros(2))

    # maxpost should correspond to the MAP
    assert results['maxpost'] >= results['logpost'].max() - 1e-10


def test_amcmc_custom_covariance():
    # AMCMC with custom initial covariance
    np.random.seed(42)
    ndim = 2
    mean = np.zeros(ndim)
    cov = np.eye(ndim)

    logpost, _ = make_gaussian_logpost(mean, cov)

    cov_ini = 0.01 * np.eye(ndim)
    sampler = AMCMC(cov_ini=cov_ini, gamma=0.5)
    sampler.setLogPost(logpost, None)

    results = sampler.run(500, np.zeros(ndim))

    assert results['chain'].shape == (501, ndim)


if __name__ == '__main__':
    test_amcmc_creation()
    test_amcmc_run()
    test_amcmc_acceptance_rate()
    test_amcmc_chain_shape()
    test_hmc_run()
    test_mala_run()
    test_amcmc_logpost_stored()
    test_amcmc_custom_covariance()
