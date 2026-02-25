#!/usr/bin/env python
"""Test script for general utilities (xutils)."""

import os
import numpy as np
from quinn.utils.xutils import (idt, savepk, loadpk, cartes_list,
                                  sample_sphere, get_opt_bw, get_pdf,
                                  safe_cholesky)


def test_idt():
    # Identity function should return input unchanged
    assert idt(42) == 42
    assert idt("hello") == "hello"

    x = np.array([1.0, 2.0])
    assert np.array_equal(idt(x), x)


def test_cartes_list():
    # Cartesian product of lists
    result = cartes_list([['a', 'b'], [1, 2]])
    expected = [('a', 1), ('a', 2), ('b', 1), ('b', 2)]

    assert result == expected


def test_cartes_list_single():
    # Cartesian product of single list
    result = cartes_list([[1, 2, 3]])
    expected = [(1,), (2,), (3,)]

    assert result == expected


def test_cartes_list_three():
    # Cartesian product of three lists
    result = cartes_list([[1], [2], [3]])
    expected = [(1, 2, 3)]

    assert result == expected


def test_savepk_loadpk(tmp_path=None):
    # Save and load pickle roundtrip
    import tempfile
    tmpdir = tempfile.mkdtemp()
    fpath = os.path.join(tmpdir, 'test_obj')

    obj = {'key': [1, 2, 3], 'value': np.array([4.0, 5.0])}
    savepk(obj, nameprefix=fpath)
    loaded = loadpk(nameprefix=fpath)

    assert loaded['key'] == [1, 2, 3]
    assert np.allclose(loaded['value'], np.array([4.0, 5.0]))

    # Clean up
    os.remove(fpath + '.pk')


def test_sample_sphere_shape():
    # sample_sphere should return correct shape
    center = np.array([0.0, 0.0, 0.0])
    rad = 1.0
    nsam = 50

    samples = sample_sphere(center, rad, nsam)
    assert samples.shape == (nsam, 3)


def test_sample_sphere_distance():
    # All samples should be at approximately the given radius from center
    center = np.array([1.0, 2.0])
    rad = 3.0
    nsam = 100

    samples = sample_sphere(center, rad, nsam)
    distances = np.linalg.norm(samples - center, axis=1)

    assert np.allclose(distances, rad, atol=1.e-10)


def test_get_opt_bw():
    # get_opt_bw should return a positive bandwidth
    np.random.seed(42)
    xsam = np.random.randn(200).reshape(-1, 1)

    bw = get_opt_bw(xsam)
    assert bw > 0.0


def test_get_pdf():
    # get_pdf should return non-negative values
    np.random.seed(42)
    data = np.random.randn(500)
    target = np.linspace(-3, 3, 50)

    pdf_vals = get_pdf(data, target)
    assert len(pdf_vals) == len(target)
    assert np.all(pdf_vals >= 0)


def test_safe_cholesky():
    # safe_cholesky on well-conditioned SPD matrix
    np.random.seed(42)
    A = np.random.rand(5, 5)
    cov = A @ A.T + 0.1 * np.eye(5)

    L = safe_cholesky(cov)

    # L @ L.T should reconstruct original matrix
    assert np.allclose(L @ L.T, cov, atol=1.e-8)


def test_safe_cholesky_near_singular():
    # safe_cholesky should handle near-singular matrices via SVD fallback
    np.random.seed(42)
    n = 5
    v = np.random.rand(n, 1)
    cov = v @ v.T + 1e-10 * np.eye(n)

    L = safe_cholesky(cov)

    # Should at least produce a matrix
    assert L.shape == (n, n)


if __name__ == '__main__':
    test_idt()
    test_cartes_list()
    test_cartes_list_single()
    test_cartes_list_three()
    test_savepk_loadpk()
    test_sample_sphere_shape()
    test_sample_sphere_distance()
    test_get_opt_bw()
    test_get_pdf()
    test_safe_cholesky()
    test_safe_cholesky_near_singular()
