#!/usr/bin/env python
"""Test script for PyTorch utilities (tchutils)."""

import numpy as np
import torch
from quinn.nns.tchutils import tch, npy, flatten_params, recover_flattened
from quinn.nns.mlp import MLP


def test_tch_from_numpy():
    # tch should convert numpy array to torch tensor
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    x_t = tch(x_np)

    assert isinstance(x_t, torch.Tensor)
    assert x_t.shape == (2, 2)
    assert np.allclose(x_t.numpy(), x_np)


def test_tch_with_grad():
    # tch with rgrad=True should enable gradients
    x_np = np.array([[1.0, 2.0]])
    x_t = tch(x_np, rgrad=True)

    assert x_t.requires_grad


def test_tch_without_grad():
    # tch with rgrad=False should not enable gradients
    x_np = np.array([[1.0, 2.0]])
    x_t = tch(x_np, rgrad=False)

    assert not x_t.requires_grad


def test_npy_from_tensor():
    # npy should convert torch tensor to numpy
    x_t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_np = npy(x_t)

    assert isinstance(x_np, np.ndarray)
    assert x_np.shape == (2, 2)
    assert np.allclose(x_np, x_t.numpy())


def test_tch_npy_roundtrip():
    # tch -> npy roundtrip should preserve values
    x_orig = np.random.rand(5, 3)
    x_back = npy(tch(x_orig))

    assert np.allclose(x_orig, x_back)


def test_flatten_params():
    # flatten_params should flatten all model parameters
    net = MLP(2, 1, (5,))
    params = list(net.parameters())

    flat, indices = flatten_params(params)

    total = sum(p.numel() for p in params)
    assert flat.shape == (total, 1)
    assert len(indices) == len(params)


def test_flatten_recover_roundtrip():
    # flatten and recover should preserve model predictions
    net = MLP(2, 1, (5,))
    x = torch.randn(10, 2)

    y_before = net(x).detach()

    params = list(net.parameters())
    flat, indices = flatten_params(params)

    # Modify params, then recover
    recovered = recover_flattened(flat, indices, net)

    y_after = net(x).detach()
    assert torch.allclose(y_before, y_after)


def test_tch_1d_input():
    # tch should handle 1d arrays
    x_np = np.array([1.0, 2.0, 3.0])
    x_t = tch(x_np)

    assert isinstance(x_t, torch.Tensor)


def test_npy_with_grad():
    # npy should work on tensors with gradients
    x_t = torch.tensor([1.0, 2.0], requires_grad=True)
    x_np = npy(x_t)

    assert isinstance(x_np, np.ndarray)


if __name__ == '__main__':
    test_tch_from_numpy()
    test_tch_with_grad()
    test_tch_without_grad()
    test_npy_from_tensor()
    test_tch_npy_roundtrip()
    test_flatten_params()
    test_flatten_recover_roundtrip()
    test_tch_1d_input()
    test_npy_with_grad()
