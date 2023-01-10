#!/usr/bin/env python
"""Module for various useful wrappers to NN functions."""

import torch
import numpy as np

from .tchutils import tch, npy


class NNWrap():
    """Wrapper class to any PyTorch NN module.

    Attributes:
        nnmodel (torch.nn.Module): The original PyTorch NN module.
        indices (list): List containing [start index, end index) for each model parameter. Useful for flattening/unflattening of parameter arrays.
    """

    def __init__(self, nnmodel):
        """Instantiate a NN Wrapper object.

        Args:
            nnmodel (torch.nn.Module): The original PyTorch NN module.
        """
        self.nnmodel = nnmodel
        self.indices = None
        _ = self.p_flatten()

    def __call__(self, x):
        """Calling the wrapper function.

        Args:
            x (np.ndarray): A numpy input array of size `(N,d)`.

        Returns:
            np.ndarray: A numpy output array of size `(N,o)`.
        """
        return npy(self.nnmodel.forward(tch(x)))

    def p_flatten(self):
        """Flattens all parameters into an array.

        Returns:
            torch.Tensor: A flattened (1d) torch tensor.
        """
        l = [torch.flatten(p) for p in self.nnmodel.parameters()]
        self.indices = []
        s = 0
        for p in l:
            size = p.shape[0]
            self.indices.append((s, s+size))
            s += size
        flat_parameter = torch.cat(l).view(-1, 1)

        return flat_parameter

    def p_unflatten(self, flat_parameter):
        """Fills the values of corresponding parameters given the flattened form.

        Args:
            flat_parameter (np.ndarray): A flattened form of parameters.

        Returns:
            list[torch.Tensor]: List of recovered parameters, reshaped and ordered to match the model.

        Note:
            Returning the list is secondary. The most important result is that this function internally fills the values of corresponding parameters.
        """

        ll = [tch(flat_parameter[s:e]) for (s, e) in self.indices]
        for i, p in enumerate(self.nnmodel.parameters()):
            if len(p.shape)>0:
                ll[i] = ll[i].view(*p.shape)

            p.data = ll[i]

        return ll

###############################################################
###############################################################
###############################################################

def nnwrapper(x, nnmodel):
    r"""A simple wrapper function to any PyTorch NN module :math:`f(x)=\textrm{NN}(x)`.

    Args:
        x (np.ndarray): An input numpy array `x` of size `(N,d)`.
        nnmodel (torch.nn.Module): The underlying PyTorch NN module.

    Returns:
        np.ndarray: An output numpy array of size `(N,o)`.
    """
    return npy(nnmodel.forward(tch(x)))


def nn_surrogate(x, *otherpars):
    r"""A simple wrapper function as a surrogate to a PyTorch NN module :math:`f(x)=\textrm{NN}(x)`.

    Args:
        x (np.ndarray): An input numpy array `x` of size `(N,d)`.
        otherpars (list): List containing one element, the PyTorch NN module of interest.

    Returns:
        np.ndarray: An output numpy array of size `(N,o)`.

    Note:
        This is effectively the same as nnwrapper.
    """
    nnmodule = otherpars[0]

    return nnwrapper(x, nnmodule)

def nn_surrogate_multi(par, *otherpars):
    r"""A simple wrapper function as a surrogate to a PyTorch NN module :math:`f_i(x)=\textrm{NN}_i(x)` for `i=1,...,o`.

    Args:
        x (np.ndarray): An input numpy array `x` of size `(N,d)`.
        otherpars (list[list]): List containing one element, a list of PyTorch NN modules of interest (a total of `o` modules).

    Returns:
        np.ndarray: An output numpy array of size `(N,o)`.
    """
    nnmodules = otherpars[0]

    nout = len(nnmodules)
    yy = np.empty((par.shape[0], nout))
    for iout in range(nout):
        yy[:, iout] = nnwrapper(par, nnmodules[iout]).reshape(-1,)

    return yy

def nn_p(p, x, *otherpars):
    r"""A NN wrapper that evaluates a given PyTorch NN module given input `x` and flattened parameter vector `p`. In other words, :math:`f(p,x)=\textrm{NN}_p(x).`

    Args:
        p (np.ndarray): Flattened parameter (weights) vector.
        x (np.ndarray): An input numpy array `x` of size `(N,d)`.
        otherpars (list): List containing one element, the PyTorch NN module of interest.

    Returns:
        np.ndarray: A numpy output array of size `(N,o)`.

    Note:
        The size checks on `p` are missing: wherever this is used in QUiNN, the size checks are implied and correct. Use with care outside QUiNN.
    """
    nnmodule = otherpars[0]
    nnw = NNWrap(nnmodule)
    nnw.p_unflatten(p)
    return nnw(x)
