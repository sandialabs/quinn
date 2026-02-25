#!/usr/bin/env python
"""Test script for statistics utilities."""

import numpy as np
from quinn.utils.stats import get_stats, get_domain, intersect_domain, diam


def test_get_stats_mean():
    # get_stats with qt=False should return mean and mean±std bounds
    np.random.seed(42)
    yy = np.random.randn(10000, 3) + np.array([1.0, 2.0, 3.0])

    mid, lb, ub = get_stats(yy, qt=False)

    assert np.allclose(mid, [1.0, 2.0, 3.0], atol=0.05)


def test_get_stats_mean_bounds_symmetric():
    # With qt=False, lb and ub should be equal (both are std)
    np.random.seed(42)
    yy = np.random.randn(5000, 2)

    mid, lb, ub = get_stats(yy, qt=False)

    assert np.allclose(lb, ub)


def test_get_stats_quantiles():
    # get_stats with qt=True should return median and half-IQR widths
    np.random.seed(42)
    yy = np.random.randn(10000, 2)

    mid, lb, ub = get_stats(yy, qt=True)

    # Median should be near 0
    assert np.allclose(mid, 0.0, atol=0.05)
    # lb = median - Q25, ub = Q75 - median, both should be positive
    assert np.all(lb > 0)
    assert np.all(ub > 0)


def test_get_domain_shape():
    # get_domain should return (dim, 2) array
    np.random.seed(42)
    xx = np.random.rand(100, 5)

    dom = get_domain(xx)

    assert dom.shape == (5, 2)
    assert np.all(dom[:, 0] < dom[:, 1])


def test_get_domain_values():
    # Domain boundaries should match min/max
    np.random.seed(42)
    xx = np.random.rand(50, 3) * 10 - 5

    dom = get_domain(xx)

    assert np.allclose(dom[:, 0], np.min(xx, axis=0))
    assert np.allclose(dom[:, 1], np.max(xx, axis=0))


def test_intersect_domain():
    # Intersection of overlapping domains
    dom1 = np.array([[0., 5.], [0., 5.]])
    dom2 = np.array([[2., 7.], [3., 8.]])

    dom_int = intersect_domain(dom1, dom2)

    assert dom_int is not None
    assert np.allclose(dom_int, [[2., 5.], [3., 5.]])


def test_intersect_domain_none():
    # Non-overlapping domains should return None
    dom1 = np.array([[0., 1.]])
    dom2 = np.array([[2., 3.]])

    dom_int = intersect_domain(dom1, dom2)

    assert dom_int is None


def test_intersect_domain_identical():
    # Identical domains should return the same domain
    dom = np.array([[0., 5.], [0., 10.]])

    dom_int = intersect_domain(dom, dom)

    assert np.allclose(dom_int, dom)


def test_diam_unit_square():
    # Diameter of points in [0,1]^2 <= sqrt(2)
    np.random.seed(42)
    xx = np.random.rand(500, 2)

    diameter = diam(xx)

    assert diameter <= np.sqrt(2)
    assert diameter > 1.0  # should be close to sqrt(2)


def test_diam_collinear():
    # Diameter along a line should be max pairwise distance
    xx = np.array([[0.0], [1.0], [3.0], [5.0]])
    diameter = diam(xx)

    assert np.isclose(diameter, 5.0)


if __name__ == '__main__':
    test_get_stats_mean()
    test_get_stats_mean_bounds_symmetric()
    test_get_stats_quantiles()
    test_get_domain_shape()
    test_get_domain_values()
    test_intersect_domain()
    test_intersect_domain_none()
    test_intersect_domain_identical()
    test_diam_unit_square()
    test_diam_collinear()
