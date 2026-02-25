#!/usr/bin/env python
"""Test script for NN fitting (nnfit)."""

import numpy as np
import torch
from quinn.nns.mlp import MLP
from quinn.nns.nnfit import nnfit


def test_nnfit_basic():
    # nnfit should train an MLP and return a result dict
    np.random.seed(42)
    torch.manual_seed(42)

    indim, outdim = 1, 1
    net = MLP(indim, outdim, (16, 16), activ='tanh')

    N = 50
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    result = nnfit(net, xtrn, ytrn, nepochs=200, lrate=0.01, freq_out=1000)

    assert 'best_nnmodel' in result
    assert 'best_loss' in result
    assert 'best_epoch' in result
    assert 'history' in result
    assert result['best_loss'] < 1.0  # should have learned something


def test_nnfit_decreasing_loss():
    # Loss should generally decrease during training
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (16,), activ='tanh')

    N = 50
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = xtrn ** 2

    result = nnfit(net, xtrn, ytrn, nepochs=300, lrate=0.01, freq_out=1000)

    history = result['history']
    # history is list of [epoch, batch_loss, trn_loss, val_loss]
    # Early trn_loss should be bigger than later trn_loss
    assert history[-1][2] < history[0][2]


def test_nnfit_with_validation():
    # nnfit should accept validation data
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (16,), activ='tanh')

    N = 50
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    xval = np.random.rand(10, 1) * 2 - 1
    yval = np.sin(xval)

    result = nnfit(net, xtrn, ytrn, val=[xval, yval],
                   nepochs=200, lrate=0.01, freq_out=1000)

    assert result['best_loss'] < 1.0


def test_nnfit_batch_training():
    # nnfit with batch_size should work
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (16,), activ='tanh')

    N = 100
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = xtrn ** 3

    result = nnfit(net, xtrn, ytrn, nepochs=200, lrate=0.01,
                   batch_size=20, freq_out=1000)

    assert result['best_loss'] < 1.0


def test_mlpbase_fit():
    # MLPBase.fit() wrapper should work correctly
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (16,), activ='tanh')

    N = 50
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    net.fit(xtrn, ytrn, nepochs=200, lrate=0.01, freq_out=1000)

    # After fit, predict should use the best model
    ypred = net.predict(xtrn)
    residual = np.mean((ypred - ytrn) ** 2)
    assert residual < 0.5


def test_nnfit_multioutput():
    # nnfit should handle multiple outputs
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(2, 3, (16,), activ='tanh')

    N = 50
    xtrn = np.random.rand(N, 2)
    ytrn = np.column_stack([np.sum(xtrn, axis=1),
                             np.prod(xtrn, axis=1),
                             xtrn[:, 0] - xtrn[:, 1]])

    result = nnfit(net, xtrn, ytrn, nepochs=300, lrate=0.01, freq_out=1000)

    assert result['best_nnmodel'] is not None


def test_nnfit_weight_decay():
    # nnfit with weight decay should work
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (16,), activ='tanh')

    N = 50
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    result = nnfit(net, xtrn, ytrn, nepochs=200, lrate=0.01,
                   wd=0.001, freq_out=1000)

    assert result['best_loss'] < 1.0


if __name__ == '__main__':
    test_nnfit_basic()
    test_nnfit_decreasing_loss()
    test_nnfit_with_validation()
    test_nnfit_batch_training()
    test_mlpbase_fit()
    test_nnfit_multioutput()
    test_nnfit_weight_decay()
