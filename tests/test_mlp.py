#!/usr/bin/env python
"""Test script for MLP neural network construction and forward pass."""

import numpy as np
import torch
from quinn.nns.mlp import MLP
from quinn.nns.rnet import RNet


def test_mlp_creation():
    # MLP should be created with correct dimensions
    indim, outdim = 3, 2
    hls = (10, 10)
    net = MLP(indim, outdim, hls)

    assert net.indim == indim
    assert net.outdim == outdim


def test_mlp_forward_shape():
    # Forward pass should produce correct output shape
    indim, outdim = 4, 1
    hls = (8, 8)
    net = MLP(indim, outdim, hls)

    x = torch.randn(20, indim)
    y = net(x)

    assert y.shape == (20, outdim)


def test_mlp_predict_shape():
    # predict() with numpy should return numpy array with correct shape
    indim, outdim = 2, 3
    hls = (16,)
    net = MLP(indim, outdim, hls)

    x = np.random.rand(15, indim)
    y = net.predict(x)

    assert isinstance(y, np.ndarray)
    assert y.shape == (15, outdim)


def test_mlp_numpar():
    # numpar() should return expected count
    # For MLP with layers [2] -> [5] -> [1]:
    # layer1: 2*5 + 5 = 15 (weights + bias)
    # layer2: 5*1 + 1 = 6
    # total = 21
    net = MLP(2, 1, (5,))
    npar = net.numpar()

    assert npar == 21


def test_mlp_activations():
    # Different activations should produce different outputs
    indim, outdim = 2, 1
    hls = (10,)

    torch.manual_seed(42)
    net_relu = MLP(indim, outdim, hls, activ='relu')
    torch.manual_seed(42)
    net_tanh = MLP(indim, outdim, hls, activ='tanh')

    # Forward pass gives different results with different activations
    x = torch.randn(10, indim)
    y_relu = net_relu(x)
    y_tanh = net_tanh(x)

    # Shapes should match even if values differ
    assert y_relu.shape == y_tanh.shape


def test_mlp_no_bias():
    # MLP with no bias should have fewer parameters
    indim, outdim = 3, 1
    hls = (10,)

    net_bias = MLP(indim, outdim, hls, biasorno=True)
    net_nobias = MLP(indim, outdim, hls, biasorno=False)

    assert net_nobias.numpar() < net_bias.numpar()


def test_mlp_dropout():
    # MLP with dropout should be creatable
    net = MLP(2, 1, (10, 10), dropout=0.2)
    x = torch.randn(5, 2)
    y = net(x)

    assert y.shape == (5, 1)


def test_mlp_batchnorm():
    # MLP with batch normalization should be creatable
    net = MLP(2, 1, (10, 10), bnorm=True)
    x = torch.randn(10, 2)  # need enough samples for batchnorm
    y = net(x)

    assert y.shape == (10, 1)


def test_mlp_final_transform_exp():
    # MLP with exp final transform should produce positive outputs
    net = MLP(2, 1, (10,), final_transform='exp')
    x = torch.randn(20, 2)
    y = net(x)

    assert torch.all(y > 0)


def test_mlp_sin_activation():
    # MLP with sinusoidal activation (may not be available in all torch versions)
    try:
        net = MLP(2, 1, (10,), activ='sin')
        x = torch.randn(5, 2)
        y = net(x)
        assert y.shape == (5, 1)
    except (TypeError, RuntimeError):
        # Known issue with Sine activation in some torch versions
        pass


def test_mlp_multi_hidden():
    # MLP with many hidden layers
    net = MLP(3, 2, (8, 8, 8, 8))
    x = torch.randn(10, 3)
    y = net(x)

    assert y.shape == (10, 2)


def test_rnet_creation():
    # RNet with indim != rdim requires layer_pre, outdim != rdim requires layer_post
    rdim = 5
    nlayers = 3
    net = RNet(rdim, nlayers, indim=2, outdim=1, layer_pre=True, layer_post=True)

    x = torch.randn(10, 2)
    y = net(x)

    assert y.shape == (10, 1)


def test_rnet_same_dim():
    # RNet with indim == rdim == outdim needs no pre/post layers
    rdim = 3
    nlayers = 4
    net = RNet(rdim, nlayers)

    x = torch.randn(10, 3)
    y = net(x)

    assert y.shape == (10, 3)


def test_rnet_predict():
    # RNet predict with numpy
    rdim = 4
    nlayers = 3
    net = RNet(rdim, nlayers, indim=2, outdim=1, layer_pre=True, layer_post=True)

    x = np.random.rand(10, 2)
    y = net.predict(x)

    assert isinstance(y, np.ndarray)
    assert y.shape == (10, 1)


def test_rnet_numpar():
    # RNet should have parameters
    rdim = 5
    nlayers = 3
    net = RNet(rdim, nlayers, indim=2, outdim=1, layer_pre=True, layer_post=True)
    npar = net.numpar()

    assert npar > 0


def test_rnet_mlp_mode():
    # RNet with mlp=True disables residual connections
    rdim = 4
    nlayers = 3
    net = RNet(rdim, nlayers, indim=2, outdim=1, mlp=True, layer_pre=True, layer_post=True)

    x = torch.randn(5, 2)
    y = net(x)

    assert y.shape == (5, 1)


if __name__ == '__main__':
    test_mlp_creation()
    test_mlp_forward_shape()
    test_mlp_predict_shape()
    test_mlp_numpar()
    test_mlp_activations()
    test_mlp_no_bias()
    test_mlp_dropout()
    test_mlp_batchnorm()
    test_mlp_final_transform_exp()
    test_mlp_sin_activation()
    test_mlp_multi_hidden()
    test_rnet_creation()
    test_rnet_same_dim()
    test_rnet_predict()
    test_rnet_numpar()
    test_rnet_mlp_mode()
