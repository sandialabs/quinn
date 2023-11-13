#!/usr/bin/env python
"""Module for various useful wrappers to NN functions."""

import torch
import numpy as np

from .tchutils import tch, npy


class NNWrap:
    """Wrapper class to any PyTorch NN module.

    Attributes:
        - nnmodel: The original PyTorch NN module. Type: torch.nn.Module.
        - indices: List containing [start index, end index) for each
        model parameter. Useful for flattening/unflattening of parameter
        arrays. Type: list.
    """

    def __init__(self, nnmodel):
        """Instantiate a NN Wrapper object.
        ----------
        Args:
            - nnmodel: The original PyTorch NN module. Type: torch.nn.Module.
        """
        self.nnmodel = nnmodel
        self.indices = None
        _ = self.p_flatten()

        try:
            self.device = self.nnmodel.device
        except AttributeError:
            self.device = "cpu"

    def __call__(self, x):
        """Calling the wrapper function.
        ----------
        Args:
            - x (np.ndarray): A numpy input array of size `(N,d)`.
            - type: indicates whether the output should be a numpy array or a
            torch tensor.
        ----------
        Returns:
            - np.ndarray: A numpy output array of size `(N,o)`.
        """
        try:
            device = self.nnmodel.device
        except AttributeError:
            device = "cpu"

        return npy(self.nnmodel.forward(tch(x, device=device)))

    def p_flatten(self):
        """Flattens all parameters into an array.
        ----------
        Returns:
            torch.Tensor: A flattened (1d) torch tensor.
        """
        l = [torch.flatten(p) for p in self.nnmodel.parameters()]
        self.indices = []
        s = 0
        for p in l:
            size = p.shape[0]
            self.indices.append((s, s + size))
            s += size
        flat_parameter = torch.cat(l).view(-1, 1)

        return flat_parameter

    def p_unflatten(self, flat_parameter):
        """Fills the values of corresponding parameters given the flattened form.
        ----------
        Args:
            flat_parameter (np.ndarray): A flattened form of parameters.
        ----------
        Returns:
            - list[torch.Tensor]: List of recovered parameters, reshaped and
            ordered to match the model.
        ----------
        Note:
            Returning the list is secondary. The most important result is that
            this function internally fills the values of corresponding parameters.
        """
        # FIXME: we should only allocate tensors in initialization.
        try:
            device = self.nnmodel.device
        except AttributeError:
            device = "cpu"

        ll = [tch(flat_parameter[s:e], device=device) for (s, e) in self.indices]
        for i, p in enumerate(self.nnmodel.parameters()):
            if len(p.shape) > 0:
                ll[i] = ll[i].view(*p.shape)

            p.data = ll[i]

        return ll

    def predict(self, x_in, weights):
        """Model prediction given new weights.
        ----------
        Args:
            - x (np.ndarray): A numpy input array of size `(N,d)`.
        ----------
        Returns:
            - np.ndarray: A numpy output array of size `(N,o)`.
        """
        x_in = tch(x_in)
        self.p_unflatten(weights)
        y_out = self.model(x_in).squeeze().detach().numpy()
        return y_out


###############################################################
###############################################################
###############################################################


class NNWrap_Torch(torch.nn.Module):
    """Wrapper class to any PyTorch NN module. This class inherits the properties of
    torch.nn.Module to facilitate some of the tasks. This is initially though to be used
    in MCMC sampling problems. It adds two additional methods to the previous class
    to calculate a loss and the gradients of the loss with respect to the model parameters
    (given the model parameters, loss function (nn.Module) and data points).

    Attributes:
        - nnmodel: The original PyTorch NN module. Type: torch.nn.Module.
        - indices: List containing [start index, end index) for each
        model parameter. Useful for flattening/unflattening of parameter
        arrays. Type: list.
    """

    def __init__(self, nnmodel):
        """Instantiate a NN Wrapper object.
        ----------
        Args:
            - nnmodel: The original PyTorch NN module. Type: torch.nn.Module.
        """
        super().__init__()
        self.nnmodel = nnmodel
        self.indices = None
        _ = self.p_flatten()

        try:
            self.device = self.nnmodel.device
        except AttributeError:
            self.device = "cpu"

    def __call__(self, x):
        """Calling the wrapper function.
        ----------
        Args:
            - x (np.ndarray): A numpy input array of size `(N,d)`.
            - type: indicates whether the output should be a numpy array or sa
            torch tensor.
        ----------
        Returns:
            - torch.Tensor: `(N,o)`.
        """
        try:
            device = self.nnmodel.device
        except AttributeError:
            device = "cpu"

        return self.nnmodel(tch(x, device=device))

    def forward(self, x):
        """Calling the wrapper function.
        ----------
        Args:
            - x (np.ndarray): A numpy input array of size `(N,d)`.
            - type: indicates whether the output should be a numpy array or sa
            torch tensor.
        ----------
        Returns:
            - np.ndarray: A numpy output array of size `(N,o)`.
        """
        try:
            device = self.nnmodel.device
        except AttributeError:
            device = "cpu"

        return npy(self.nnmodel(tch(x, device=device)))

    def p_flatten(self):
        """Flattens all parameters into an array.
        ----------
        Returns:
            torch.Tensor: A flattened (1d) torch tensor.
        """
        l = [torch.flatten(p) for p in self.nnmodel.parameters()]
        self.indices = []
        s = 0
        for p in l:
            size = p.shape[0]
            self.indices.append((s, s + size))
            s += size
        flat_parameter = torch.cat(l).view(-1, 1)

        return flat_parameter

    def p_unflatten(self, flat_parameter):
        """Fills the values of corresponding parameters given the flattened form.
        ----------
        Args:
            flat_parameter (np.ndarray): A flattened form of parameters.
        ----------
        Returns:
            - list[torch.Tensor]: List of recovered parameters, reshaped and
            ordered to match the model.
        ----------
        Note:
            Returning the list is secondary. The most important result is that
            this function internally fills the values of corresponding parameters.
        """
        # FIXME: we should only allocate tensors in initialization.
        try:
            device = self.nnmodel.device
        except AttributeError:
            device = "cpu"
        ll = [tch(flat_parameter[s:e], device=device) for (s, e) in self.indices]
        for i, p in enumerate(self.nnmodel.parameters()):
            if len(p.shape) > 0:
                ll[i] = ll[i].view(*p.shape)

            p.data = ll[i]

        return ll

    def predict(self, x_in, weights):
        """Model prediction given new weights.
        ----------
        Args:
            - x (np.ndarray): A numpy input array of size `(N,d)`.
        ----------
        Returns:
            - np.ndarray: A numpy output array of size `(N,o)`.
        """
        x_in = tch(x_in)
        self.p_unflatten(weights)
        y_out = self.nnmodel(x_in).detach().numpy()
        return y_out

    def calc_loss(self, weights, loss_fn, input, targets):
        """Calculates the loss given a loss function.
        ----------
        Args:
            - weights (numpy.ndarray): weights of the model.
            - loss_fn (torch.nn.Module): pytorch module calculating the loss
            given the following args:
                - model (NNWrap class): model from which the parameters are
                taken.
                - input (numpy.ndarray): data that is input in the model.
                - target (numpy.ndarray): data corresponding to the output of
                the model.
                - requires_grad (bool): whether we will need to compute the
                gradients with respect to the log posterior
            - input (numpy.ndarray): input to the model.
            - target (numpy.ndarray): targets of the output of the model.
        ---------
        Returns:
            -loss (torch.Tensor): loss of the model given the data.
        """
        input = tch(input, rgrad=False)
        targets = tch(targets, rgrad=False)
        self.p_unflatten(weights)
        loss = loss_fn(self.nnmodel, input, targets, requires_grad=False)
        return loss.item()

    def calc_grad_wrt_loss(self, weights, loss_fn, input, targets):
        """Calculates the loss given a loss function.
        ----------
        Args:
            - weights (numpy.ndarray): weights of the model.
            - loss_fn (torch.nn.Module): pytorch module calculating the loss
            given the following args:
                - model (NNWrap class): model from which the parameters are
                taken.
                - input (numpy.ndarray): data that is input in the model.
                - target (numpy.ndarray): data corresponding to the output of
                the model.
                - requires_grad (bool): whether we will need to compute the
                gradients with respect to the log posterior
            - input (numpy.ndarray): input to the model.
            - target (numpy.ndarray): targets of the output of the model.
        ---------
        Returns:
            - np.ndarray: A numpy array of the loss w.r.t. the gradients of the
            model parameters.
        """
        input = tch(input, rgrad=False)
        targets = tch(targets, rgrad=False)
        self.p_unflatten(weights)
        loss = loss_fn(self.nnmodel, input, targets, requires_grad=True)
        loss.backward()
        gradients = []
        for p in self.nnmodel.parameters():
            gradients.append(npy(p.grad).flatten())
            p.grad = None
        return np.concatenate(gradients, axis=0)


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
    device = self.nnmodel.device
    return npy(nnmodel.forward(tch(x, device=device)))


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
        yy[:, iout] = nnwrapper(par, nnmodules[iout]).reshape(
            -1,
        )

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
