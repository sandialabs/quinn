#!/usr/bin/env python
"""Test script for NNWrap and related utilities."""

import numpy as np
import torch
from quinn.nns.mlp import MLP
from quinn.nns.nnwrap import NNWrap, nnwrapper, nn_p


def test_nnwrap_call():
    # NNWrap should accept numpy and return numpy
    net = MLP(2, 1, (10,))
    wrap = NNWrap(net)

    x = np.random.rand(15, 2)
    y = wrap(x)

    assert isinstance(y, np.ndarray)
    assert y.shape == (15, 1)


def test_nnwrap_flatten_unflatten():
    # Flatten and unflatten should preserve parameters
    net = MLP(2, 1, (10,))
    wrap = NNWrap(net)

    flat = wrap.p_flatten().detach().numpy().flatten()
    npar = len(flat)
    assert npar == net.numpar()

    # Unflatten with same values should give same predictions
    x = np.random.rand(5, 2)
    y1 = wrap(x)

    wrap.p_unflatten(flat)
    y2 = wrap(x)

    assert np.allclose(y1, y2)


def test_nnwrap_predict_with_weights():
    # predict() with custom weights
    net = MLP(2, 1, (5,))
    wrap = NNWrap(net)

    flat = wrap.p_flatten().detach().numpy().flatten()
    x = np.random.rand(10, 2)

    y = wrap.predict(x, flat)
    assert y.shape == (10, 1)


def test_nnwrap_calc_loss():
    # calc_loss should return a scalar
    net = MLP(2, 2, (5,))
    wrap = NNWrap(net)

    weights = wrap.p_flatten().detach().numpy().flatten()
    loss_fn = torch.nn.MSELoss()

    x = np.random.rand(10, 2)
    y = np.random.rand(10, 2)

    loss_val = wrap.calc_loss(weights, loss_fn, x, y)
    assert isinstance(loss_val, float)
    assert loss_val >= 0.0


def test_nnwrap_calc_lossgrad():
    # calc_lossgrad should return gradient of correct shape
    # NegLogPost is needed because loss_fn must call model internally
    from quinn.nns.losses import NegLogPost
    net = MLP(2, 1, (5,))
    wrap = NNWrap(net)

    weights = wrap.p_flatten().detach().numpy().flatten()
    loss_fn = NegLogPost(net, 10, 0.1, None)

    x = np.random.rand(10, 2)
    y = np.random.rand(10, 1)

    grad = wrap.calc_lossgrad(weights, loss_fn, x, y)
    assert grad.shape == weights.shape


def test_nnwrap_calc_hess_diag():
    # calc_hess_diag should return diagonal hessian
    from quinn.nns.losses import NegLogPost
    net = MLP(2, 1, (5,))
    wrap = NNWrap(net)

    weights = wrap.p_flatten().detach().numpy().flatten()
    loss_fn = NegLogPost(net, 10, 0.1, None)

    x = np.random.rand(10, 2)
    y = np.random.rand(10, 1)

    hess_diag = wrap.calc_hess_diag(weights, loss_fn, x, y)
    assert hess_diag.shape == (len(weights), len(weights))


def test_nnwrapper():
    # nnwrapper should be a simple numpy wrapper
    net = MLP(2, 1, (10,))

    x = np.random.rand(10, 2)
    y = nnwrapper(x, net)

    assert isinstance(y, np.ndarray)
    assert y.shape == (10, 1)


def test_nn_p():
    # nn_p should evaluate NN with flattened parameter vector
    net = MLP(2, 1, (5,))
    wrap = NNWrap(net)
    p = wrap.p_flatten().detach().numpy().flatten()

    x = np.random.rand(10, 2)
    y = nn_p(p, x, net)

    assert isinstance(y, np.ndarray)
    assert y.shape == (10, 1)


def test_nn_p_different_weights():
    # nn_p with different weights should give different predictions
    net = MLP(2, 1, (5,))
    wrap = NNWrap(net)
    p1 = wrap.p_flatten().detach().numpy().flatten()
    p2 = p1 + 0.1  # perturbed weights

    x = np.random.rand(10, 2)
    y1 = nn_p(p1, x, net)
    y2 = nn_p(p2, x, net)

    assert not np.allclose(y1, y2)


if __name__ == '__main__':
    test_nnwrap_call()
    test_nnwrap_flatten_unflatten()
    test_nnwrap_predict_with_weights()
    test_nnwrap_calc_loss()
    test_nnwrap_calc_lossgrad()
    test_nnwrap_calc_hess_diag()
    test_nnwrapper()
    test_nn_p()
    test_nn_p_different_weights()
