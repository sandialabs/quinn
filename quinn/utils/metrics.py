#!/usr/bin/env python
"""Module for various metrics of comparison."""

import numpy as np

def rel_l2_ens(predictions, targets):
    """Relative L2 metric for ensamble.

    Args:
        predictions (np.ndarray): prediction array
        targets (np.ndarray): target array

    Returns:
        np.ndarray: relative L2 error
    """
    return np.linalg.norm(np.linalg.norm(predictions - targets,axis=1),axis=1)/ np.linalg.norm(targets)
    

def rel_l2(predictions, targets):
    """Relative L2 metric.

    Args:
        predictions (np.ndarray): prediction array
        targets (np.ndarray): target array

    Returns:
        float: relative L2 error
    """
    return np.linalg.norm(predictions - targets) / np.linalg.norm(targets)


def rmse(predictions, targets):
    """Root-mean-square error (RMSE).

    Args:
        predictions (np.ndarray): prediction array
        targets (np.ndarray): target array

    Returns:
        float: RMSE error
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def fast_auc(y_true, y_prob):
    """Fast Area-Under-Curve (AUC) computation.

    See: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013

    Args:
        y_true (int np.ndarray): true array
        y_prob (float np.ndarray): predicted probabilities

    Returns:
        float: AUC metric
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def mae(y_true, y_pred):
    """Mean absolute error (MAE).

    Args:
        y_true (np.ndarray): true array
        y_pred (np.ndarray): predicted array

    Returns:
        float: MAE metric
    """

    return np.abs(y_true - y_pred).mean()
