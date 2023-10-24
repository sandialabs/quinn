#!/usr/bin/env python
"""Various analytical functions for testing the methods."""

import numpy as np


def blundell(xx, datanoise=0.0):
    r"""Classical example from :cite:t:`blundell:2015`.

    .. math::
        f(x)=x+0.3 \sin(2\pi(x+\sigma\:{\cal N}(0,1)))+0.3 \sin(4\pi(x+\sigma\:{\cal N}(0,1)))+\sigma\:{\cal N}(0,1)

    Args:
        xx (np.ndarray): Input array :math:`x` of size `(N,d)`.
        datanoise (float, optional): Standard deviation :math:`\sigma` of i.i.d. gaussian noise, both on the input and output.

    Returns:
        np.ndarray: Output array of size `(N,d)`.
    Note:
        This function is typically used in `d=1` setting.
    """
    noise = datanoise * np.random.randn(xx.shape[0], xx.shape[1])
    yy = (
        xx
        + 0.3 * np.sin(2.0 * np.pi * (xx + noise))
        + 0.3 * np.sin(4.0 * np.pi * (xx + noise))
        + noise
    )
    return yy


def Sine(xx, datanoise=0.0):
    r"""Simple sum of sines function

    .. math::
        f(x)=\sin(x_1)+...+\sin(x_d) + \sigma \: {\cal N} (0,1)

    Args:
        xx (np.ndarray): Input array :math:`x` of size `(N,d)`.
        datanoise (float, optional): Standard deviation :math:`\sigma` of i.i.d. gaussian noise.

    Returns:
        np.ndarray: Output array of size `(N,1)`.
    """
    yy = datanoise * np.random.randn(xx.shape[0], 1)
    yy += np.sum(np.sin(xx), axis=1).reshape(-1, 1)

    return yy


def Sine10(xx, datanoise=0.02):
    r"""Sum of sines function with 10 outputs

    .. math::
        f_1(x)=\sin(x_1)+...+\sin(x_d) + \sigma \: {\cal N} (0,1)\\
        \dots \qquad\qquad\qquad\qquad\\
        f_{10}(x)=\sin(x_1)+...+\sin(x_d) + \sigma \: {\cal N} (0,1)


    Args:
        xx (np.ndarray): Input array :math:`x` of size `(N,d)`.
        datanoise (float, optional): Standard deviation :math:`\sigma` of i.i.d. gaussian noise.

    Returns:
        np.ndarray: Output array of size `(N,10)`.
    """
    yy = datanoise * np.random.randn(xx.shape[0], 10)
    yy += np.sum(np.sin(xx), axis=1).reshape(-1, 1)

    return yy


def Ackley(x, datanoise=0.02):
    r"""Ackley4 or Modified Ackley function from https://arxiv.org/pdf/1308.4008v1.pdf.

    .. math::
        f(x)=\sum_{i=1}^{d-1} \left(\exp(-0.2)\sqrt{x_i^2+x_{i+1}^2} + 3 (\cos{2x_i}+\sin{2x_{i+1}})\right) + \sigma \: {\cal N} (0,1)

    Args:
        xx (np.ndarray): Input array :math:`x` of size `(N,d)`.
        datanoise (float, optional): Standard deviation :math:`\sigma` of i.i.d. gaussian noise.

    Returns:
        np.ndarray: Output array of size `(N,1)`.
    """
    yy = datanoise * np.random.randn(
        x.shape[0],
    )
    ndim = x.shape[1]

    for i in range(ndim - 1):
        yy += np.exp(-0.2) * np.sqrt(x[:, i] ** 2 + x[:, i + 1] ** 2) + 3 * (
            np.cos(2 * x[:, i]) + np.sin(2 * x[:, i + 1])
        )
    return yy.reshape(-1, 1)
