#!/usr/bin/env python
"""Module for various useful wrappers to NN functions."""

import torch
import numpy as np

from .nnbase import MLPBase
from .tchutils import tch, npy

class NNWrap():
    """Wrapper class to any PyTorch NN module to make it work as a numpy function. Basic usage is therefore :math:`f=NNWrap(); y=f(x)`

    Attributes:
        indices (list): List containing [start index, end index) for each model parameter. Useful for flattening/unflattening of parameter arrays.
        nnmodel (torch.nn.Module): The original PyTorch NN module.
    """

    def __init__(self, nnmodel):
        """Instantiate a NN Wrapper object.

        Args:
            nnmodel (torch.nn.Module): The original PyTorch NN module.
        """
        self.nnmodel = nnmodel
        self.indices = None
        _ = self.p_flatten()

    def reinitialize_instance(self):
        """Reinitialize the underlying NN module."""
        self.nnmodel.reinitialize_instance()


    def __call__(self, x):
        """Calling the wrapper function.

        Args:
            x (np.ndarray): A numpy input array of size `(N,d)`.

        Returns:
            np.ndarray: A numpy output array of size `(N,o)`.
        """
        try:
            device = self.nnmodel.device
        except AttributeError:
            device = 'cpu'

        return npy(self.nnmodel.forward(tch(x, device=device)))

    def predict(self, x_in, weights):
        """Model prediction given new weights.

        Args:
            x_in (np.ndarray): A numpy input array of size `(N,d)`.
            weights (np.ndarray): flattened parameter vector.

        Returns:
            np.ndarray: A numpy output array of size `(N,o)`.
        """
        x_in = tch(x_in)
        self.p_unflatten(weights)
        y_out = self.nnmodel(x_in).detach().numpy()
        return y_out

    def p_flatten(self):
        """Flattens all parameters of the underlying NN module into an array.

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
        """Fills the values of corresponding parameters given the flattened numpy form.

        Args:
            flat_parameter (np.ndarray): A flattened form of parameters.

        Returns:
            list[torch.Tensor]: List of recovered parameters, reshaped and ordered to match the model.

        Note:
            Returning the list is secondary. The most important result is that this function internally fills the values of corresponding parameters.
        """
        # FIXME: we should only allocate tensors in initialization. 
        try:
            device = self.nnmodel.device
        except AttributeError:
            device = 'cpu'

        ll = [tch(flat_parameter[s:e],device=device) for (s, e) in self.indices]
        for i, p in enumerate(self.nnmodel.parameters()):
            if len(p.shape)>0:
                ll[i] = ll[i].view(*p.shape)

            p.data = ll[i]

        return ll


    def calc_loss(self, weights, loss_fn, inputs, targets):
        """Calculates a given loss function with respect to model parameters.

        Args:
            weights (np.ndarray): weights of the model.
            loss_fn (torch.nn.Module): pytorch loss module of signature loss(inputs, targets)
            inputs (np.ndarray): inputs to the model.
            targets (np.ndarray): target outputs that get compared to model outputs.

        Returns:
            loss (float): loss of the model given the data.
        """
        inputs = tch(inputs, rgrad=False)
        targets = tch(targets, rgrad=False)
        self.p_unflatten(weights) # TODO: this is not always necessary if loss_fn already incorporates the weights?

        loss = loss_fn(inputs, targets)
        return loss.item()

    def calc_lossgrad(self, weights, loss_fn, inputs, targets):
        """Calculates the gradients of a given loss function with respect to model parameters.

        Args:
            weights (np.ndarray): weights of the model.
            loss_fn (torch.nn.Module): pytorch loss module of signature loss(inputs, targets)
            inputs (np.ndarray): inputs to the model.
            targets (np.ndarray): target outputs that get compared to model outputs.

        Returns:
            np.ndarray: A numpy array of the loss gradient w.r.t. to the model parameters at inputs.
        """
        inputs = tch(inputs, rgrad=False)
        targets = tch(targets, rgrad=False)
        self.p_unflatten(weights) # TODO: this is not always necessary if loss_fn already incorporates the weights?

        loss = loss_fn(inputs, targets)
        loss.backward()
        gradients = []
        for p in self.nnmodel.parameters():
            gradients.append(npy(p.grad).flatten())
            p.grad = None
        return np.concatenate(gradients, axis=0)


    def calc_hess_full(self, weigths, loss_fn, inputs, targets):
        """Calculates the hessian of a given loss function with respect to model parameters.

        Args:
            weights (np.ndarray): weights of the model.
            loss_fn (torch.nn.Module): pytorch loss module of signature loss(inputs, targets)
            inputs (np.ndarray): inputs to the model.
            targets (np.ndarray): target outputs that get compared to model outputs.

        Returns:
            np.ndarray: Hessian matrix of the loss with respect to the model parameters at inputs.
        """
        inputs = tch(inputs, rgrad=False)
        targets = tch(targets, rgrad=False)
        self.p_unflatten(weigths) # TODO: this is not always necessary if loss_fn already incorporates the weights?

        # Calculate the gradient
        loss = loss_fn(inputs, targets)

        ## One method...
        # loss.backward()
        # gradients = []
        # for p in self.nnmodel.parameters():
        #     gradients.append(npy(p.grad).flatten())
        #     p.grad = None
        # gradients = np.concatenate(gradients, axis=0)

        ## ... or its alternative
        gradients = torch.autograd.grad(
            loss, self.nnmodel.parameters(), create_graph=True, retain_graph=True
        )
        gradients = [gradient.flatten() for gradient in gradients]

        hessian_rows = []
        # Calculate the gradient of the elements of the gradient
        for gradient in gradients:
            for j in range(gradient.size(0)):
                hessian_rows.append(
                    torch.autograd.grad(gradient[j], self.nnmodel.parameters(), retain_graph=True)
                )
        hessian_mat = []
        # Shape the Hessian to a 2D tensor
        for i in range(len(hessian_rows)):
            row_hessian = []
            for gradient in hessian_rows[i]:
                row_hessian.append(gradient.flatten().unsqueeze(0))
            hessian_mat.append(torch.cat(row_hessian, dim=1))
        hessian_mat = torch.cat(hessian_mat, dim=0)
        return hessian_mat.detach().numpy()


    def calc_hess_diag(self, weigths, loss_fn, inputs, targets):
        """Calculates the diagonal hessian approximation of a given loss function with respect to model parameters.

        Args:
            weights (np.ndarray): weights of the model.
            loss_fn (torch.nn.Module): pytorch loss module of signature loss(inputs, targets)
            inputs (np.ndarray): inputs to the model.
            targets (np.ndarray): target outputs that get compared to model outputs.

        Returns:
            np.ndarray: A diagonal Hessian matrix of the loss with respect to the model parameters at inputs.
        """
        inputs = tch(inputs, rgrad=False)
        targets = tch(targets, rgrad=False)
        self.p_unflatten(weigths) # TODO: this is not always necessary if loss_fn already incorporates the weights?

        # Calculate the gradient
        gradient_list = []
        for input_, target_ in zip(inputs, targets):
            loss = loss_fn(input_, target_)

            gradients = torch.autograd.grad(loss, self.nnmodel.parameters(), create_graph=True, retain_graph=True)
            gradient_list.append(torch.cat([gradient.flatten() for gradient in gradients]).unsqueeze(0))
        diag_fim = torch.cat(gradient_list, dim=0).pow(2).mean(0)

        return torch.diag(diag_fim).detach().numpy()

############################################################
############################################################
############################################################

class SNet(MLPBase):
    """A single NN wrapper of a given torch NN module. This is useful as it will inherit all the methods of MLPBase. Written in the spirit of UQ wrapper/solvers.

    Attributes:
        nnmodel (torch.nn.Module): The underlying torch NN module.
    """

    def __init__(self, nnmodel, indim, outdim, device='cpu'):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): The underlying torch NN module.
            indim (int): Input dimensionality.
            outdim (int): Output dimensionality.
            device (str, optional): Device where the computations will be done. Defaults to 'cpu'.
        """
        super().__init__(indim, outdim, device=device)
        self.nnmodel = nnmodel

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.nnmodel(x)

###############################################################
###############################################################
###############################################################

def nnwrapper(x, nnmodel):
    """A simple numpy-ifying wrapper function to any PyTorch NN module :math:`f(x)=\textrm{NN}(x)`.

    Args:
        x (np.ndarray): An input numpy array `x` of size `(N,d)`.
        nnmodel (torch.nn.Module): The underlying PyTorch NN module.

    Returns:
        np.ndarray: An output numpy array of size `(N,o)`.
    """
    try:
        device = nnmodel.device
    except AttributeError:
        device = 'cpu'
    return npy(nnmodel.forward(tch(x,device=device, rgrad=False)))


def nn_surrogate(x, *otherpars):
    r"""A simple wrapper function as a surrogate to a PyTorch NN module :math:`f(x)=\textrm{NN}(x)`.

    Args:
        x (np.ndarray): An input numpy array `x` of size `(N,d)`.
        otherpars (list): List containing one element, the PyTorch NN module of interest.

    Returns:
        np.ndarray: An output numpy array of size `(N,o)`.

    Note:
        This is effectively the same as nnwrapper. It is kept for backward compatibility.
    """
    nnmodule = otherpars[0]

    return nnwrapper(x, nnmodule)

###############################################################
###############################################################
###############################################################

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

###############################################################
###############################################################
###############################################################

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
