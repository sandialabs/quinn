#!/usr/bin/env python
"""Utilities for descriptive statistics of data."""


import numpy as np


def get_stats(yy, qt):
    """Gets stats of a given dataset to help with plotting.

    Args:
        yy (np.ndarray): array of predicted values
        qt (bool): whether to compute quantiles or not

    Returns:
        tuple: tuple of np.ndarray, (mean, std, std) or
               (median, q50-q25, q75-q50)
    """
    yy_mean = np.mean(yy, axis=0)
    yy_std = np.std(yy, axis=0)
    yy_qt = np.quantile(yy, [0.25, 0.5, 0.75], axis=0)

    if qt:
        yy_lb = yy_qt[1] - yy_qt[0]
        yy_ub = yy_qt[2] - yy_qt[1]
        yy_mb = yy_qt[1]
    else:
        yy_lb = yy_std
        yy_ub = yy_std
        yy_mb = yy_mean

    return yy_mb, yy_lb, yy_ub


def get_domain(xx):
    """Get the domain of a given data array.

    Args:
        xx (np.ndarray): A data array of size `(N,d)`.

    Returns:
        np.ndarray: `(d,2)` domain array.
    """
    _, ndim = xx.shape
    domain = np.empty((ndim, 2))
    domain[:, 0] = np.min(xx, axis=0)
    domain[:, 1] = np.max(xx, axis=0)

    return domain


def diam(xx):
    """Get the diameter of a given data array.

    Args:
        xx (np.ndarray): A data array of size `(N,d)`.

    Returns:
        float: diameter, i.e. max pairwise distance.
    """
    pdist = np.linalg.norm(xx[:, None, :] - xx[None, :, :], axis=-1)
    diameter = np.max(pdist)

    return diameter
