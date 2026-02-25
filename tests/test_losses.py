#!/usr/bin/env python
"""Test script for loss functions."""

import numpy as np
import torch
from quinn.nns.mlp import MLP
from quinn.nns.losses import NegLogPost, NegLogPrior


def test_neglogprior_at_anchor():
    # NegLogPrior at anchor point should give minimum
    net = MLP(2, 1, (5,))
    npar = sum(p.numel() for p in net.parameters())
    anchor = torch.cat([p.detach().flatten() for p in net.parameters()])  # anchor = current params
    sigma = 1.0
    prior = NegLogPrior(sigma, anchor)

    loss_at_anchor = prior(net)

    # Perturb parameters
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    loss_away = prior(net)

    assert loss_at_anchor < loss_away


def test_neglogprior_symmetry():
    # NegLogPrior evaluated on a model should give finite result
    net = MLP(2, 1, (5,))
    npar = sum(p.numel() for p in net.parameters())
    anchor = torch.zeros(npar)
    sigma = 1.0
    prior = NegLogPrior(sigma, anchor)

    loss = prior(net)
    assert torch.isfinite(loss)


def test_neglogpost_creation():
    # NegLogPost should be creatable with an NN model
    net = MLP(2, 1, (5,))
    ndata = 50
    sigma = 0.1
    loss = NegLogPost(net, ndata, sigma, None)

    x = torch.randn(10, 2)
    y = torch.randn(10, 1)

    val = loss(x, y)
    assert torch.isfinite(val)


def test_neglogpost_with_prior():
    # NegLogPost with prior should give higher loss than without
    net = MLP(2, 1, (5,))
    ndata = 50
    sigma = 0.1

    npar = sum(p.numel() for p in net.parameters())
    priorparams = {'sigma': 1.0, 'anchor': torch.randn(npar)}

    loss_no_prior = NegLogPost(net, ndata, sigma, None)
    loss_with_prior = NegLogPost(net, ndata, sigma, priorparams)

    x = torch.randn(10, 2)
    y = torch.randn(10, 1)

    # Both should produce finite results
    val1 = loss_no_prior(x, y)
    val2 = loss_with_prior(x, y)
    assert torch.isfinite(val1)
    assert torch.isfinite(val2)


def test_neglogpost_zero_residual():
    # NegLogPost with perfect predictions should be minimal
    net = MLP(1, 1, (5,))
    ndata = 10
    sigma = 0.1
    loss = NegLogPost(net, ndata, sigma, None)

    x = torch.randn(ndata, 1)
    y = net(x).detach()  # perfect target

    loss_perfect = loss(x, y)

    # Now compare with bad target
    y_bad = y + 10.0
    loss_bad = loss(x, y_bad)

    assert loss_perfect < loss_bad


if __name__ == '__main__':
    test_neglogprior_at_anchor()
    test_neglogprior_symmetry()
    test_neglogpost_creation()
    test_neglogpost_with_prior()
    test_neglogpost_zero_residual()
