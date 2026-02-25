#!/usr/bin/env python
"""Test script for analytical test functions."""

import numpy as np
from quinn.func.funcs import blundell, Sine, Sine10, Summation, Ackley, x5


def test_blundell_shape():
    # blundell output shape should match input shape
    N, d = 20, 1
    xx = np.random.rand(N, d)
    yy = blundell(xx, datanoise=0.0)

    assert yy.shape == (N, d)


def test_blundell_no_noise():
    # Without noise, blundell should be deterministic
    np.random.seed(42)
    N, d = 50, 1
    xx = np.linspace(-1, 1, N).reshape(-1, 1)
    yy1 = blundell(xx, datanoise=0.0)
    yy2 = blundell(xx, datanoise=0.0)

    assert np.allclose(yy1, yy2)


def test_blundell_formula():
    # Verify formula: f(x) = x + 0.3*sin(2*pi*x) + 0.3*sin(4*pi*x)
    N = 30
    xx = np.linspace(-1, 1, N).reshape(-1, 1)
    yy = blundell(xx, datanoise=0.0)

    expected = xx + 0.3 * np.sin(2. * np.pi * xx) + 0.3 * np.sin(4. * np.pi * xx)
    assert np.allclose(yy, expected)


def test_blundell_multidim():
    # blundell should work for multi-dimensional input
    N, d = 20, 3
    xx = np.random.rand(N, d)
    yy = blundell(xx, datanoise=0.0)

    assert yy.shape == (N, d)


def test_blundell_with_noise():
    # With noise, results should differ across calls
    N, d = 100, 1
    xx = np.random.rand(N, d)
    yy1 = blundell(xx, datanoise=0.5)
    yy2 = blundell(xx, datanoise=0.5)

    assert not np.allclose(yy1, yy2)


def test_sine_shape():
    # Sine should return (N, 1) regardless of input dim
    N, d = 25, 4
    xx = np.random.rand(N, d)
    yy = Sine(xx, datanoise=0.0)

    assert yy.shape == (N, 1)


def test_sine_formula():
    # Verify formula: f(x) = sum(sin(x_i))
    N, d = 30, 3
    xx = np.random.rand(N, d)
    yy = Sine(xx, datanoise=0.0)

    expected = np.sum(np.sin(xx), axis=1).reshape(-1, 1)
    assert np.allclose(yy, expected)


def test_sine_1d():
    # Sine in 1D reduces to sin(x)
    N = 20
    xx = np.linspace(0, 2 * np.pi, N).reshape(-1, 1)
    yy = Sine(xx, datanoise=0.0)

    expected = np.sin(xx)
    assert np.allclose(yy, expected)


def test_sine10_shape():
    # Sine10 returns (N, 10) outputs
    N, d = 20, 3
    xx = np.random.rand(N, d)
    yy = Sine10(xx, datanoise=0.0)

    assert yy.shape == (N, 10)


def test_sine10_columns_equal():
    # Without noise, all 10 columns of Sine10 should be equal
    N, d = 30, 2
    xx = np.random.rand(N, d)
    yy = Sine10(xx, datanoise=0.0)

    for j in range(1, 10):
        assert np.allclose(yy[:, 0], yy[:, j])


def test_sine10_matches_sine():
    # Each column of Sine10 should match Sine (without noise)
    N, d = 30, 2
    xx = np.random.rand(N, d)
    yy_10 = Sine10(xx, datanoise=0.0)
    yy_1 = Sine(xx, datanoise=0.0)

    assert np.allclose(yy_10[:, 0].reshape(-1, 1), yy_1)


def test_summation_shape():
    # Summation returns (N, 1)
    N, d = 25, 5
    xx = np.random.rand(N, d)
    yy = Summation(xx, datanoise=0.0)

    assert yy.shape == (N, 1)


def test_summation_formula():
    # Verify formula: f(x) = sum(x_i)
    N, d = 30, 4
    xx = np.random.rand(N, d)
    yy = Summation(xx, datanoise=0.0)

    expected = np.sum(xx, axis=1).reshape(-1, 1)
    assert np.allclose(yy, expected)


def test_ackley_shape():
    # Ackley should return (N, 1)
    N, d = 20, 3
    xx = np.random.rand(N, d)
    yy = Ackley(xx, datanoise=0.0)

    assert yy.shape == (N, 1)


def test_ackley_2d():
    # Verify Ackley formula for d=2
    N = 20
    xx = np.random.rand(N, 2)
    yy = Ackley(xx, datanoise=0.0)

    expected = (np.exp(-0.2) * np.sqrt(xx[:, 0]**2 + xx[:, 1]**2)
                + 3 * (np.cos(2 * xx[:, 0]) + np.sin(2 * xx[:, 1]))).reshape(-1, 1)
    assert np.allclose(yy, expected)


def test_x5_shape():
    # x5 should return (N, 1)
    N = 25
    xx = np.random.rand(N, 2)
    yy = x5(xx, datanoise=0.0)

    assert yy.shape == (N, 1)


def test_x5_formula():
    # Verify formula: f(x) = x_0^5
    N = 30
    xx = np.linspace(-1, 1, N).reshape(-1, 1)
    yy = x5(xx, datanoise=0.0)

    expected = xx**5
    assert np.allclose(yy, expected)


def test_x5_only_first_dim():
    # x5 should only depend on the first input dimension
    N = 20
    xx1 = np.column_stack([np.linspace(0, 1, N), np.zeros(N)])
    xx2 = np.column_stack([np.linspace(0, 1, N), np.ones(N)])

    yy1 = x5(xx1, datanoise=0.0)
    yy2 = x5(xx2, datanoise=0.0)

    assert np.allclose(yy1, yy2)


if __name__ == '__main__':
    test_blundell_shape()
    test_blundell_no_noise()
    test_blundell_formula()
    test_blundell_multidim()
    test_blundell_with_noise()
    test_sine_shape()
    test_sine_formula()
    test_sine_1d()
    test_sine10_shape()
    test_sine10_columns_equal()
    test_sine10_matches_sine()
    test_summation_shape()
    test_summation_formula()
    test_ackley_shape()
    test_ackley_2d()
    test_x5_shape()
    test_x5_formula()
    test_x5_only_first_dim()
