#!/usr/bin/env python
"""Test script for UQ solvers: NN_MCMC, NN_Laplace, NN_SWAG, NN_RMS."""

import numpy as np
import torch
from quinn.nns.mlp import MLP
from quinn.solvers.nn_mcmc import NN_MCMC
from quinn.solvers.nn_laplace import NN_Laplace
from quinn.solvers.nn_swag import NN_SWAG
from quinn.solvers.nn_rms import NN_RMS
import pytest


# --- NN_MCMC tests ---

def test_nn_mcmc_creation():
    # NN_MCMC should be creatable
    net = MLP(1, 1, (5,), activ='tanh')
    mcmc = NN_MCMC(net, verbose=False)

    assert mcmc.pdim > 0
    assert mcmc.samples is None


def test_nn_mcmc_fit_amcmc():
    # NN_MCMC fit with AMCMC sampler
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (5,), activ='tanh')
    mcmc = NN_MCMC(net, verbose=False)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    mcmc.fit(xtrn, ytrn, nmcmc=200, datanoise=0.1,
             sampler='amcmc', zflag=False, sampler_params={})

    assert mcmc.samples is not None
    assert mcmc.cmode is not None


def test_nn_mcmc_predict_sample():
    # NN_MCMC predict_sample after fit (needs a param array)
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (5,), activ='tanh')
    mcmc = NN_MCMC(net, verbose=False)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    mcmc.fit(xtrn, ytrn, nmcmc=200, datanoise=0.1,
             sampler='amcmc', zflag=False, sampler_params={})

    # predict_sample requires a param vector
    y = mcmc.predict_sample(xtrn, mcmc.cmode)
    assert y.shape == (N, 1)


def test_nn_mcmc_predict_ens():
    # NN_MCMC predict_ens should return ensemble
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (5,), activ='tanh')
    mcmc = NN_MCMC(net, verbose=False)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    mcmc.fit(xtrn, ytrn, nmcmc=300, datanoise=0.1,
             sampler='amcmc', zflag=False, sampler_params={})

    nens = 5
    y_ens = mcmc.predict_ens(xtrn, nens=nens, nburn=100)
    assert y_ens.shape[0] == nens


def test_nn_mcmc_predict_map():
    # NN_MCMC predict_MAP should use MAP weights
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (5,), activ='tanh')
    mcmc = NN_MCMC(net, verbose=False)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    mcmc.fit(xtrn, ytrn, nmcmc=200, datanoise=0.1,
             sampler='amcmc', zflag=False, sampler_params={})

    y_map = mcmc.predict_MAP(xtrn)
    assert y_map.shape == (N, 1)


# --- NN_Laplace tests ---

def test_nn_laplace_creation():
    # NN_Laplace should be creatable
    net = MLP(1, 1, (5,), activ='tanh')
    lap = NN_Laplace(net, la_type='diag')

    assert lap.la_type == 'diag'
    assert lap.nparams > 0


def test_nn_laplace_fit_predict():
    # NN_Laplace should fit and predict
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (5,), activ='tanh')
    lap = NN_Laplace(net, la_type='diag', datanoise=0.1, priorsigma=1.0)

    N = 30
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1).astype(np.float32)
    ytrn = np.sin(xtrn).astype(np.float32)

    lap.fit(xtrn, ytrn, nepochs=100, lrate=0.01, freq_out=1000)

    y_sample = lap.predict_sample(xtrn)
    assert y_sample.shape == (N, 1)


def test_nn_laplace_predict_ens():
    # NN_Laplace predict_ens
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (5,), activ='tanh')
    lap = NN_Laplace(net, la_type='diag', datanoise=0.1, priorsigma=1.0)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1).astype(np.float32)
    ytrn = np.sin(xtrn).astype(np.float32)

    lap.fit(xtrn, ytrn, nepochs=100, lrate=0.01, freq_out=1000)

    nens = 5
    y_ens = lap.predict_ens(xtrn, nens=nens)
    assert y_ens.shape == (nens, N, 1)


# --- NN_SWAG tests ---

def test_nn_swag_creation():
    # NN_SWAG should be creatable
    net = MLP(1, 1, (5,), activ='tanh')
    swag = NN_SWAG(net, k=3, n_steps=5)

    assert swag.k == 3
    assert swag.n_steps == 5


def test_nn_swag_fit_predict():
    # NN_SWAG should fit and predict
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (5,), activ='tanh')
    swag = NN_SWAG(net, k=3, n_steps=5, datanoise=0.1, priorsigma=1.0)

    N = 30
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1).astype(np.float32)
    ytrn = np.sin(xtrn).astype(np.float32)

    swag.fit(xtrn, ytrn, nepochs=50, lrate=0.01, freq_out=1000)

    y_sample = swag.predict_sample(xtrn)
    assert y_sample.shape == (N, 1)


def test_nn_swag_predict_ens():
    # NN_SWAG predict_ens
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (5,), activ='tanh')
    swag = NN_SWAG(net, k=3, n_steps=5, datanoise=0.1)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1).astype(np.float32)
    ytrn = np.sin(xtrn).astype(np.float32)

    swag.fit(xtrn, ytrn, nepochs=50, lrate=0.01, freq_out=1000)

    nens = 5
    y_ens = swag.predict_ens(xtrn, nens=nens)
    assert y_ens.shape == (nens, N, 1)


# --- NN_RMS tests ---

def test_nn_rms_creation():
    # NN_RMS should be creatable
    net = MLP(1, 1, (5,), activ='tanh')
    rms = NN_RMS(net, nens=2, datanoise=0.1, priorsigma=1.0)

    assert rms.datanoise == 0.1
    assert rms.priorsigma == 1.0


def test_nn_rms_fit_predict():
    # NN_RMS should fit and predict
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (5,), activ='tanh')
    rms = NN_RMS(net, nens=2, datanoise=0.1, priorsigma=1.0)

    N = 30
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    rms.fit(xtrn, ytrn, nepochs=100, lrate=0.01, freq_out=1000)

    ypred = rms.predict(xtrn)
    assert ypred.shape == (N, 1)


def test_nn_rms_predict_ens():
    # NN_RMS predict_ens
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (5,), activ='tanh')
    nens = 2
    rms = NN_RMS(net, nens=nens, datanoise=0.1, priorsigma=1.0)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    rms.fit(xtrn, ytrn, nepochs=100, lrate=0.01, freq_out=1000)

    y_ens = rms.predict_ens(xtrn, nens=nens)
    assert y_ens.shape == (nens, N, 1)


if __name__ == '__main__':
    test_nn_mcmc_creation()
    test_nn_mcmc_fit_amcmc()
    test_nn_mcmc_predict_sample()
    test_nn_mcmc_predict_ens()
    test_nn_mcmc_predict_map()
    test_nn_laplace_creation()
    test_nn_laplace_fit_predict()
    test_nn_laplace_predict_ens()
    test_nn_swag_creation()
    test_nn_swag_fit_predict()
    test_nn_swag_predict_ens()
    test_nn_rms_creation()
    test_nn_rms_fit_predict()
    test_nn_rms_predict_ens()
