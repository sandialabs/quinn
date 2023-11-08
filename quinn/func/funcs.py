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


def genz_funcs(x, a=None, u=None, func_type="continuous", datanoise=0.02):
    r"""Genz functions shown in http://www.sfu.ca/~ssurjano/integration.html.
    this functions are meant to be used in the domain [0,1]^d
    "a" parameter vector defaults to 1 if none is provided.
    Types:
        - Continuous intergrand family ("continuous")
            .. math:: f(x)=exp(-\sum_{i=0}^d (a_i \vert x_i - u_i \vert))
                parameters:
                    a_i: higher values result in a sharper peak.
                    u_i: chosen randomly from [0,1] if not provided.

        - Corner peak intergrand family ("corner")
            .. math:: f(x)=(1+\sum_{i=1}^d (a_i * x_i))^{-(d+1)}
                parameters:
                    a_i: higher values result in a sharper peak.

        - Discontinuous intergrand family ("discontinuous")
            .. math:: f(x)=0 if any x_i > u_i
                          =exp(\sum_{i=1}^d (a_i * x_i)) otherwise
                parameters:
                    a_i: higher values result in a sharper peak.
                    u_i: chosen randomly from [0,1] if not provided.

        - Gaussian peak intergrand family ("gaussian")
            .. math:: f(x)=exp(-\sum_{i=0}^d (a_i^2 ( x_i - u_i )^2)
                parameters:
                    a_i: higher values result in a sharper peak.
                    u_i: chosen randomly from [0,1] if not provided.

        - Oscillatory intergrand family ("oscillatory")
            .. math:: f(x)=cos(2\pi*u_1 + \sum_{i=1}^d (a_i * x_i))
                parameters:
                    a_i: higher values result in higher frequency of
                        oscillations.
                    u_1: chosen randomly from [0,1] if not provided.

        - Product intergrand family ("product")
            .. math:: f(x)=\Pi_{i=1}^d 1 / (a_i^-2 + (x_i-u_i)^2)
                parameters:
                    a_i: higher values result in a sharper peak.
                    u_i: chosen randomly from [0,1] if not provided.

    Args:
        xx (np.ndarray): Input array :math:`x` of size `(N,d)`.
        a (np.ndarray, float): array with a parameters.
        u (np.ndarray, float): array with u parameters.
        func_type (string): indicates the type of genz function to be used.
        datanoise (float, optional): Standard deviation :math:`\sigma` of i.i.d. gaussian noise.

    Returns:
        np.ndarray: Output array of size `(N,1)`.
    """
    yy = datanoise * np.random.randn(
        x.shape[0],
    )
    ndim = x.shape[1]
    if a is None:
        a = np.ones(ndim)
    a = np.repeat(np.reshape(a, newshape=(1, ndim)), len(x), axis=0)
    if u is None:
        if func_type == "oscillatory":
            u = np.random.rand()
            u = np.repeat(u, len(x))
        else:
            u = np.random.rand(ndim)
            u = np.repeat(np.reshape(u, newshape=(1, ndim)), len(x), axis=0)
    if func_type == "continuous":
        y = np.squeeze(np.exp(-np.sum(np.multiply(np.abs(x - u), a), axis=1)))
        yy = yy + y
    elif func_type == "corner":
        y = np.squeeze(np.power(1 + np.sum(np.multiply(x, a), axis=1), -ndim - 1))
        yy = yy + y
    elif func_type == "discontinuous":
        y = np.squeeze(np.exp(np.sum(np.multiply(x, a), axis=1)))
        y = np.multiply(y, np.sum(x > u, axis=1) == 0)
        yy = yy + y
    elif func_type == "gaussian":
        y = np.squeeze(
            np.exp(-np.sum(np.multiply(np.power(x - u, 2), np.power(a, 2)), axis=1))
        )
        yy = yy + y
    elif func_type == "oscillatory":
        y = np.squeeze(np.cos(2 * np.pi * u + np.sum(np.multiply(x, a), axis=1)))
        yy = yy + y
    elif func_type == "product":
        y = np.squeeze(
            np.prod(np.power(np.power(a, -2) + np.power(x - u, 2), -1), axis=1)
        )
        yy = yy + y

    return yy.reshape(-1, 1)
