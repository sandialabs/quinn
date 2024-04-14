#!/usr/bin/env python
"""Module for Laplace NN wrapper."""

import torch
import numpy as np

from .nn_ens import NN_Ens
from ..nns.nnwrap import NNWrap
from ..nns.losses import NegLogPost


class NN_Laplace(NN_Ens):
    """Wrapper class for the Laplace method.

    Attributes:
        cov_mats (list): List of covariance matrices.
        cov_scale (TYPE): Covariance scaling factor for prediction.
        datanoise (float): Data noise standard deviation.
        la_type (str): Laplace approximation type ('full' or 'diag').
        means (list): List of MAP centers.
        nparams (int): Number of parameters in the model.
        priorsigma (float): Gaussian prior standard deviation.
    """

    def __init__(self, nnmodel, la_type='full', cov_scale=1.0, datanoise=0.1, priorsigma=1.0, **kwargs):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): NNWrapper class.
            la_type (str, optional): Laplace approximation type ('full' or 'diag'). Dedaults to 'full'.
            cov_scale (float, optional): Covariance scaling factor for prediction. Defaults to 1.0.
            datanoise (float, optional): Data noise standard deviation. Defaults to 0.1.
            priorsigma (float, optional): Gaussian prior standard deviation. Defaults to 1.0.
            **kwargs: Any keyword argument that :meth:`..nns.nnfit.nnfit` takes.
        """
        super().__init__(nnmodel, **kwargs)
        self.la_type = la_type
        self.cov_scale = cov_scale
        print(
            "NOTE: the hessian has not been averaged,",
            " i.e., it has not been divided by the number of training data points.",
            "Hence, the hyperparameter hessian scale can be tuned to calibrate uncertainty.",
        )
        self.datanoise = datanoise
        self.priorsigma = priorsigma
        self.nparams = sum(p.numel() for p in self.nnmodel.parameters())

        self.means = []
        self.cov_mats = []

    def fit(self, xtrn, ytrn, **kwargs):
        """Fitting function for each ensemble member.
        Args:
            xtrn (np.ndarray): Input array of size `(N,d)`.
            ytrn (np.ndarray): Output array of size `(N,o)`.
            **kwargs (dict): Any keyword argument that :meth:`..nns.nnfit.nnfit` takes.
        """
        for jens in range(self.nens):
            print(f"======== Fitting Learner {jens+1}/{self.nens} =======")

            ntrn = ytrn.shape[0]
            permutation = np.random.permutation(ntrn)
            ind_this = permutation[: int(ntrn * self.dfrac)]

            this_learner = self.learners[jens]

            kwargs["lhist_suffix"] = f"_e{jens}"
            kwargs["loss_fn"] = "logpost"
            kwargs["datanoise"] = self.datanoise
            kwargs["priorparams"] = {'sigma': self.priorsigma, 'anchor': torch.randn(size=(self.nparams,)) * self.priorsigma}

            this_learner.fit(xtrn[ind_this], ytrn[ind_this], **kwargs)
            self.la_calc(this_learner, xtrn[ind_this], ytrn[ind_this])


    def la_calc(self, learner, xtrn, ytrn, batch_size=None):
        """Given alearner, this method stores in the corresponding lists
        the vectors and matrices defining the posterior according to the
        laplace approximation.
        Args:
            learner (Learner): Instance of the Learner class including the model
            torch.nn.Module being used.
            xtrn (np.ndarray): input part of the training data.
            ytrn (np.ndarray): target part of the training data.
            batch_size (int): batch size used in the hessian estimation.
            Defaults to None, i.e. single batch.
        """
        model = NNWrap(learner.nnmodel)

        weights_map = model.p_flatten().detach().squeeze().numpy()

        if self.la_type == "full":
            hessian_func = model.calc_hess_full
        elif self.la_type == "diag":
            hessian_func = model.calc_hess_diag
        else:
            assert (
                NotImplementedError
            ), "Wrong approximation type given. Only full and diag are accepted."

        ntrn = len(xtrn)
        loss = NegLogPost(learner.nnmodel, ntrn, 0.1, None) # TODO: hardwired datanoise
        if not batch_size:
            hessian_mat = hessian_func(weights_map, loss, xtrn, ytrn)
        if batch_size:
            hessian_mat = None
            for i in range(ntrn // batch_size + 1):
                j = min(batch_size, ntrn - i * batch_size)
                if j > 0:
                    x_batch, y_batch = (
                        xtrn[i * batch_size : i * batch_size + j],
                        ytrn[i * batch_size : i * batch_size + j],
                    )
                    hessian_cur = hessian_func(weights_map, loss, x_batch, y_batch)
                    if i == 0:
                        hessian_mat = hessian_cur
                    else:
                        hessian_mat = hessian_mat + hessian_cur

        cov_mat = np.linalg.inv(hessian_mat * self.cov_scale)
        self.means.append(weights_map)
        self.cov_mats.append(cov_mat)


    def predict_sample(self, x):
        """Predict a single sample.

        Args:
            x (np.ndarray): Input array `x` of size `(N,d)`.

        Returns:
            np.ndarray: Output array `x` of size `(N,o)`.
        """
        jens = np.random.randint(0, self.nens)
        theta = np.random.multivariate_normal(self.means[jens], cov=self.cov_mats[jens])

        model = NNWrap(self.learners[jens].nnmodel)

        return model.predict(x, theta)

    def predict_ens(self, x, nens=1):
        """Predict an ensemble of results.

        Args:
            x (np.ndarray): `(N,d)` input array.

        Returns:
            list[np.ndarray]: List of `M` arrays of size `(N, o)`, i.e. `M` random samples of `(N,o)` outputs.

        Note:
            This overloads NN_Ens's and QUiNN's base predict_ens function.
        """

        return self.predict_ens_fromsamples(x, nens=nens)



# def kfac_hessian(model, weigths_map, loss_func, x_train, y_train):
#     """
#     Calculates the Kronecker-factor Approximate Curvature hessian of the loss.
#     To be implemented taking the following papers into account:
#     Aleksandaer Botev, Hippolyt Ritter, David Barber. Practrical Gauss-Newton
#     Optimisation for Deep Learining. Proceedings of the 34th International
#     Conference on Machine Learning.
#     James Martens and Roger Grosse. Optimizing Neural Networks with Kronecker-
#     factored Approximate Curvature.
#     --------
#     Inputs:
#     - model (NNWrapp_Torch class instance): model over whose parameters we are
#     calculating the posterior.
#     - weights_map (torch.Tensor): weights of the MAP of the log_posterior given
#     by the image.
#     - loss_func: function that calculates the negative of the log posterior
#     given the model, the training data (x and y) and requires_grad to indicate that
#     gradients will be calculated.
#     - x_train (numpy.ndarray or torch.Tensor): input part of the training data.
#     - y_train (numpy.ndarray or torch.Tensor): target part of the training data.
#     --------
#     Outputs:
#     - (torch.Tensor) KFAC Hessian of the loss with respect to the model parameters.

#     """
#     return NotImplementedError
