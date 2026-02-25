#!/usr/bin/env python
"""Test script for mapping utilities."""

import numpy as np
from quinn.utils.maps import (scale01ToDom, scaleDomTo01, scaleTo01,
                               standardize, Normalizer, Standardizer,
                               Expon, Logar, ComposeMap, Domainizer, Affine)


def test_scale_roundtrip():
    # Roundtrip: [0,1] -> domain -> [0,1]
    sam = 100
    dim = 4
    xsam_01 = np.random.rand(sam, dim)

    domain = np.array([[-5., 10.], [0., 100.], [-1., 1.], [2., 7.]])

    xsam = scale01ToDom(xsam_01, domain)
    xsam_01_back = scaleDomTo01(xsam, domain)

    assert np.allclose(xsam_01, xsam_01_back)


def test_scale_boundaries():
    # Corners should map to domain corners
    dim = 3
    domain = np.array([[-2., 3.], [0., 10.], [5., 15.]])

    zeros = np.zeros((1, dim))
    ones = np.ones((1, dim))

    assert np.allclose(scale01ToDom(zeros, domain), domain[:, 0].reshape(1, -1))
    assert np.allclose(scale01ToDom(ones, domain), domain[:, 1].reshape(1, -1))


def test_scaleTo01():
    # Data scaled to [0,1] should have min 0 and max 1
    xx = np.random.rand(50, 3) * 100 - 25
    xx_scaled = scaleTo01(xx)

    assert np.allclose(xx_scaled.min(axis=0), 0.0)
    assert np.allclose(xx_scaled.max(axis=0), 1.0)


def test_standardize():
    # Standardized data should have global zero mean and unit variance
    xx = np.random.rand(200, 3) * 10 + 5
    xx_std = standardize(xx)

    assert np.isclose(xx_std.mean(), 0.0, atol=1.e-10)
    assert np.isclose(xx_std.std(), 1.0, atol=1.e-10)


def test_normalizer_roundtrip():
    # Normalizer forward+inverse roundtrip
    xx = np.random.rand(50, 3) * 10 + 2
    norm = Normalizer(xx)

    xx_norm = norm.forw(xx)
    xx_back = norm.inv(xx_norm)

    assert np.allclose(xx, xx_back)


def test_normalizer_bounds():
    # Normalized data should be in [0, 1]
    xx = np.random.rand(50, 3) * 10 + 2
    norm = Normalizer(xx)

    xx_norm = norm.forw(xx)
    assert np.all(xx_norm >= -1.e-10)
    assert np.all(xx_norm <= 1.0 + 1.e-10)


def test_standardizer_roundtrip():
    # Standardizer forward+inverse roundtrip
    xx = np.random.rand(50, 3) * 10 + 2
    std = Standardizer(xx)

    xx_std = std.forw(xx)
    xx_back = std.inv(xx_std)

    assert np.allclose(xx, xx_back)


def test_standardizer_moments():
    # Standardized data should have close to zero mean, unit std per dimension
    xx = np.random.rand(200, 3) * 10 + 5
    std = Standardizer(xx)
    xx_std = std.forw(xx)

    assert np.allclose(xx_std.mean(axis=0), 0.0, atol=1.e-10)
    assert np.allclose(xx_std.std(axis=0), 1.0, atol=1.e-10)


def test_expon_logar_roundtrip():
    # Exp and Log are inverses of each other
    xx = np.random.rand(20, 2) + 0.1  # positive values

    exp_map = Expon()
    log_map = Logar()

    assert np.allclose(log_map.forw(exp_map.forw(xx)), xx)
    assert np.allclose(exp_map.forw(log_map.forw(xx)), xx)


def test_compose_map():
    # Compose Exp and Log should be identity
    exp_map = Expon()
    log_map = Logar()

    composed = ComposeMap(exp_map, log_map)
    xx = np.random.rand(20, 2) + 0.1

    assert np.allclose(composed.forw(xx), xx)
    assert np.allclose(composed.inv(xx), xx)


def test_domainizer_roundtrip():
    # Domainizer forward+inverse roundtrip
    dom = np.array([[-2., 5.], [0., 10.]])
    dmap = Domainizer(dom)

    xx_01 = np.random.rand(30, 2)
    xx_dom = dmap.inv(xx_01)   # [0,1] -> domain
    xx_back = dmap.forw(xx_dom)  # domain -> [0,1]

    assert np.allclose(xx_01, xx_back)


def test_expon_values():
    # Expon should compute exp element-wise
    xx = np.array([[0.0, 1.0], [2.0, 3.0]])
    exp_map = Expon()
    yy = exp_map.forw(xx)

    assert np.allclose(yy, np.exp(xx))


def test_logar_values():
    # Logar should compute log element-wise
    xx = np.array([[1.0, 2.0], [3.0, 4.0]])
    log_map = Logar()
    yy = log_map.forw(xx)

    assert np.allclose(yy, np.log(xx))


def test_compose_map_order():
    # ComposeMap(inner, outer): first applies inner, then outer
    # ComposeMap(log, exp)(x) = exp(log(x)) = x for x > 0
    exp_map = Expon()
    log_map = Logar()

    composed = ComposeMap(log_map, exp_map)
    xx = np.random.rand(10, 3) + 0.1

    assert np.allclose(composed.forw(xx), xx)


if __name__ == '__main__':
    test_scale_roundtrip()
    test_scale_boundaries()
    test_scaleTo01()
    test_standardize()
    test_normalizer_roundtrip()
    test_normalizer_bounds()
    test_standardizer_roundtrip()
    test_standardizer_moments()
    test_expon_logar_roundtrip()
    test_compose_map()
    test_domainizer_roundtrip()
    test_expon_values()
    test_logar_values()
    test_compose_map_order()
