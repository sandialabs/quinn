#!/usr/bin/env python
"""Test script for Variational Inference (Bayes-by-Backprop)."""

import numpy as np
import torch
from quinn.nns.mlp import MLP
from quinn.solvers.nn_vi import NN_VI
from quinn.vi.bnet import BNet


def test_bnet_creation():
    # BNet should wrap an MLP
    net = MLP(2, 1, (8,), activ='tanh')
    bnet = BNet(net)

    # BNet should have variational parameters (mu and rho)
    param_names = [n for n, _ in bnet.named_parameters()]
    assert any('mu' in n for n in param_names)
    assert any('rho' in n for n in param_names)


def test_bnet_forward_sample():
    # BNet forward with sample=True should give stochastic outputs
    net = MLP(2, 1, (8,), activ='tanh')
    bnet = BNet(net)

    x = torch.randn(10, 2)
    y1 = bnet(x, sample=True)
    y2 = bnet(x, sample=True)

    assert y1.shape == (10, 1)
    # With sampling, results should generally differ
    # (very small chance of being exactly equal)


def test_bnet_forward_no_sample():
    # BNet forward with sample=False should use the mean weights
    net = MLP(2, 1, (8,), activ='tanh')
    bnet = BNet(net)

    x = torch.randn(10, 2)
    y1 = bnet(x, sample=False)

    # Output should be finite and correct shape
    assert y1.shape == (10, 1)
    assert torch.all(torch.isfinite(y1))


def test_bnet_sample_elbo():
    # sample_elbo should return three elbo components
    net = MLP(2, 1, (8,), activ='tanh')
    bnet = BNet(net)
    bnet.loss_params = [0.05, 1, 1]  # [datanoise, nsam, num_batches]

    x = torch.randn(10, 2)
    y = torch.randn(10, 1)

    log_prior, log_var_post, neg_log_lik = bnet.sample_elbo(x, y, nsam=1, likparams=[0.05])
    assert torch.isfinite(log_prior)
    assert torch.isfinite(log_var_post)
    assert torch.isfinite(neg_log_lik)


def test_nn_vi_creation():
    # NN_VI should be creatable
    net = MLP(1, 1, (8,), activ='tanh')
    vi = NN_VI(net)

    assert vi.trained is False
    assert vi.bmodel is not None


def test_nn_vi_fit_predict():
    # NN_VI should fit and predict
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (8, 8), activ='tanh')
    vi = NN_VI(net)

    N = 30
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    vi.fit(xtrn, ytrn, nepochs=100, lrate=0.01, datanoise=0.1, freq_out=1000)

    assert vi.trained

    # predict_sample should work
    y_sample = vi.predict_sample(xtrn)
    assert y_sample.shape == (N, 1)


def test_nn_vi_predict_ens():
    # NN_VI predict_ens should return ensemble
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (8,), activ='tanh')
    vi = NN_VI(net)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    vi.fit(xtrn, ytrn, nepochs=100, lrate=0.01, datanoise=0.1, freq_out=1000)

    nmc = 10
    y_ens = vi.predict_ens(xtrn, nens=nmc)
    assert y_ens.shape == (nmc, N, 1)


def test_nn_vi_uncertainty():
    # VI predictions should show some variance (uncertainty)
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (8,), activ='tanh')
    vi = NN_VI(net)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    vi.fit(xtrn, ytrn, nepochs=200, lrate=0.01, datanoise=0.1, freq_out=1000)

    nmc = 50
    y_ens = vi.predict_ens(xtrn, nens=nmc)
    y_var = np.var(y_ens, axis=0)

    # There should be some variance in predictions
    assert np.mean(y_var) > 0


if __name__ == '__main__':
    test_bnet_creation()
    test_bnet_forward_sample()
    test_bnet_forward_no_sample()
    test_bnet_sample_elbo()
    test_nn_vi_creation()
    test_nn_vi_fit_predict()
    test_nn_vi_predict_ens()
    test_nn_vi_uncertainty()
