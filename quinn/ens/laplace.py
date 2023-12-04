#!/usr/bin/env python
"""Module for Ensemble NN wrapper."""

import numpy as np
import torch
from torch.optim import SGD, Adam
from torch import randperm

from .ens import Ens_NN
from .learner import Learner
from ..nns.tchutils import npy, tch


class LAPLACE_NN(Ens_NN):
    """Laplace Approximation Ensemble NN Wrapper.
    Ritter, H., Izmailov, Botev A., Barber D. A Scalable Laplace Approximation For
    Neural Networks, 2018. ICLR Conference.
    Attributes:
        - loss_func : Loss function over which the mode is optimized. I takes,
        a NN model, x and y data, and requires_grad as input.
        - la_type (string): type of covariance matrix approximation. Default
        is full. Options are full, kfac and diag.
        - means (list of np.ndarrays): list of means of the posterior over each
        member of the ensemble.
        - cov_mats (list of np.ndarrays): list of covariance matrices of the
        posterior over each member of the ensemble.
        - cov_scale (float): approximated covariances are scaled by this
        factor. Defaults to 1.
    """

    def __init__(
        self,
        nnmodel,
        loss_func,
        nens=1,
        dfrac=1.0,
        la_type="full",
        cov_scale=1,
        verbose=False,
    ):
        """Initialization.
        Args:
            nnmodel (torch.nn.Module): NNWrapper class.
            - loss_func : Loss function over which the mode is optimized. I takes,
            a NN model, x and y data, and requires_grad as input.
            - nens (int, optional): Number of ensemble members. Defaults to 1.
            - dfrac (float, optional): Fraction of data for each learner. Defaults
            to 1.0.
            - la_type (string): type of covariance matrix approximation. Default
            is full. Options are full, kfac and diag.
            - cov_scale (float): approximated covariances are scaled by this
            factor. Defaults to 1.
            - verbose (bool, optional): Verbose or not.
        """
        super().__init__(
            nnmodel, nens=nens, dfrac=dfrac, type_ens="la", verbose=verbose
        )
        self.loss_func = loss_func
        self.la_type = la_type
        self.means = []
        self.cov_mats = []
        self.cov_scale = cov_scale
        print(
            "NOTE: the hessian has not been averaged,",
            " i.e., it has not been divided by the number of training data points.",
            "Hence, the hyperparameter hessian scale can be tuned to calibrate uncertainty.",
        )

        if self.verbose:
            self.print_params(names_only=True)

    def la_calc(self, learner, xtrn, ytrn, batch_size=None):
        """
        Given an optimized model, this method stores in the corresponding lists
        the vectors and matrices defining the posterior according to the
        laplace approximation.
        Args:
            - learner (Learner class): Instance of the Learner class including the model
            (torch.nn.Module) being analysed.
            - xtrn (np.ndarray): input part of the training data.
            - ytrn (np.ndarray): target part of the training data.
            - batch_size (int): batch size used in the swag covariance estimation.
            Defaults to None.
        """
        weights_map = learner.nnmodel.p_flatten().detach().squeeze().numpy()
        if self.la_type == "full":
            hessian_func = full_hessian
        elif self.la_type == "kfac":
            hessian_func = kfac_hessian
        elif self.la_type == "diag":
            hessian_func = diag_hessian
        else:
            assert (
                NotImplementedError
            ), "Wrong approximation type given. Only full, kf and diag are accepted."
        if not batch_size:
            hessian_mat = hessian_func(
                learner.nnmodel, weights_map, self.loss_func, xtrn, ytrn
            )
            cov_mat = np.linalg.inv(hessian_mat)
        if batch_size:
            hessian_mat = None
            for i in range(len(xtrn) // batch_size + 1):
                j = min(batch_size, len(xtrn) - i * batch_size)
                if j > 0:
                    x_batch, y_batch = (
                        xtrn[i * batch_size : i * batch_size + j],
                        xtrn[i * batch_size : i * batch_size + j],
                    )
                    hessian_cur = hessian_func(
                        learner.nnmodel, weights_map, self.loss_func, x_batch, y_batch
                    )
                    if i == 0:
                        hessian_mat = hessian_cur
                    else:
                        hessian_mat = hessian_mat + hessian_cur
            cov_mat = np.linalg.inv(hessian_mat * self.cov_scale)
        self.means.append(weights_map)
        self.cov_mats.append(cov_mat)

    def fit_la(self, xtrn, ytrn, **kwargs):
        """
        This methods optimizes the models and then calculates the LA posterior for
        each of them.
        Args:
            - xtrn (np.ndarray): input part of the training data.
            - ytrn (np.ndarray): target part of the training data.
            - kwargs: they must include batch_size. Other arguments are to be used
            by the nnfit function.
        """
        self.fit(xtrn, ytrn, loss_fn="logposterior", loss=self.loss_func, **kwargs)
        for jens in range(self.nens):
            self.la_calc(
                self.learners[jens], xtrn, ytrn, batch_size=kwargs["batch_size"]
            )

    def predict_sample(self, x, jens=None):
        """Predict a single, randomly selected sample.

        Args:
            x (np.ndarray): Input array of size `(N,d)`.
            jens (int): the ensemble index to use.
        Returns:
            np.ndarray: Output array of size `(N,o)`.
        """
        if jens is None:
            jens = np.random.randint(0, self.nens)
        theta = np.random.multivariate_normal(self.means[jens], cov=self.cov_mats[jens])
        return self.learners[jens].nnmodel.predict(x, theta)


def full_hessian(model, weigths_map, loss_func, x_train, y_train):
    """
    Calculates the hessian of the loss with respect to the model parameters by
    first calculating the gradient of the loss and then calculating the gradient
    of each of the elements of the first gradient.
    --------
    Inputs:
    - model (NNWrapp_Torch class instance): model over whose parameters we are
    calculating the posterior.
    - weights_map (torch.Tensor): weights of the MAP of the log_posterior given
    by the image.
    - loss_func: function that calculates the negative of the log posterior
    given the model, the training data (x and y) and requires_grad to indicate that
    gradients will be calculated.
    - x_train (numpy.ndarray or torch.Tensor): input part of the training data.
    - y_train (numpy.ndarray or torch.Tensor): target part of the training data.
    --------
    Outputs:
    - (torch.Tensor) Hessian of the loss with respect to the model parameters.

    """
    model.p_unflatten(weigths_map)
    # Calculate the gradient
    loss = loss_func(model, tch(x_train), tch(y_train), requires_grad=True)
    gradients = torch.autograd.grad(
        loss, model.parameters(), create_graph=True, retain_graph=True
    )
    gradients = [gradient.flatten() for gradient in gradients]
    hessian_rows = []
    # Calculate the gradient of the elements of the gradient
    for gradient in gradients:
        for j in range(gradient.size(0)):
            hessian_rows.append(
                torch.autograd.grad(gradient[j], model.parameters(), retain_graph=True)
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


def diag_hessian(model, weigths_map, loss_func, x_train, y_train):
    """
    Approximates the hessian as the diagonal of the Fisher Information Matrix.
    --------
    Inputs:
    - model (NNWrapp_Torch class instance): model over whose parameters we are
    calculating the posterior.
    - weights_map (torch.Tensor): weights of the MAP of the log_posterior given
    by the image.
    - loss_func: function that calculates the negative of the log posterior
    given the model, the training data (x and y) and requires_grad to indicate that
    gradients will be calculated.
    - x_train (numpy.ndarray or torch.Tensor): input part of the training data.
    - y_train (numpy.ndarray or torch.Tensor): target part of the training data.
    --------
    Outputs:
    - (torch.Tensor) Hessian of the loss with respect to the model parameters.

    """
    model.p_unflatten(weigths_map)
    # Calculate the gradient
    assert hasattr(loss_func, "likelihood_fn"), (
        "The function returning the log posterior must have as attribute a"
        "function that calculate the likelihood called 'likelihood_fn', which"
        "can predict the likelihood pointwise using the method 'predict_pointwise'."
    )
    loss = loss_func.likelihood_fn.forward_pointwise(
        model, tch(x_train), tch(y_train), requires_grad=True
    ).squeeze()
    gradient_list = []
    for i in range(loss.size(0)):
        gradients = torch.autograd.grad(
            loss[i], model.parameters(), create_graph=True, retain_graph=True
        )
        gradient_list.append(
            torch.cat([gradient.flatten() for gradient in gradients]).unsqueeze(0)
        )
    diag_fim = torch.cat(gradient_list, dim=0).pow(2).mean(0)
    return torch.diag(diag_fim).detach().numpy()


def kfac_hessian(model, weigths_map, loss_func, x_train, y_train):
    """
    Calculates the lowrank hessian of the loss. To be implemented taking the
    following papers into account:
    Aleksandaer Botev, Hippolyt Ritter, David Barber. Practrical Gauss-Newton
    Optimisation for Deep Learining. Proceedings of the 34th International
    Conference on Machine Learning.
    James Martens and Roger Grosse. Optimizing Neural Networks with Kronecker-
    factored Approximate Curvature.
    --------
    Inputs:
    - model (NNWrapp_Torch class instance): model over whose parameters we are
    calculating the posterior.
    - weights_map (torch.Tensor): weights of the MAP of the log_posterior given
    by the image.
    - loss_func: function that calculates the negative of the log posterior
    given the model, the training data (x and y) and requires_grad to indicate that
    gradients will be calculated.
    - x_train (numpy.ndarray or torch.Tensor): input part of the training data.
    - y_train (numpy.ndarray or torch.Tensor): target part of the training data.
    --------
    Outputs:
    - (torch.Tensor) KFAC Hessian of the loss with respect to the model parameters.

    """
    return NotImplementedError
