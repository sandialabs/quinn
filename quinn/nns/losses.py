#!/usr/bin/env python

import torch
import numpy as np
import torch.autograd.functional as F

from .tchutils import tch

class PeriodicLoss(torch.nn.Module):
    r"""Example of a periodic loss regularization.

    Attributes:
        model (callable): Model evaluator.
        lam (float): Penalty strength.
        bdry1 (torch.Tensor): First boundary.
        bdry2 (torch.Tensor): Second boundary.

    The loss function has a form

    .. math::
        \frac{1}{N}||y_{\text{pred}}-y_{\text{target}}||^2 + \frac{\lambda}{N}||M\text{(boundary1)}-M\text{(boundary2)}||^2.
    """

    def __init__(self, loss_params):
        """Initialization.

        Args:
            loss_params (tuple): A 4-tuple of (model, lam, boundary1, boundary2).
        """
        super().__init__()
        self.model, self.lam, self.bdry1, self.bdry2 = loss_params

    def forward(self, predictions, targets):
        """Forward function.

        Args:
            predictions (torch.Tensor): Predictions tensor.
            targets (torch.Tensor): Targets tensor.

        Returns:
            float: Loss value.
        """

        fit = torch.mean((predictions-targets)**2)

        penalty = self.lam * torch.mean((self.model(self.bdry1)-self.model(self.bdry2))**2)
        loss =  fit + penalty

        return loss


########################################################
########################################################
########################################################

class GradLoss(torch.nn.Module):
    r"""Example of grad loss function, including derivative contraints.

    Attributes:
        lam (float): Penalty strength.
        nnmodel (callable): NN module.

    The loss function has a form

    .. math::
        \frac{1}{N}||M(x_{\text{train}})-y_{\text{train}}||^2 + \frac{\lambda}{Nd}||\nabla M(x_{\text{train}})-G_{\text{train}}||_F^2.
    """

    def __init__(self, nnmodel, lam=0.0, xtrn=None, gtrn=None):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): NN model.
            lam (float, optional): Penalty strength. Defaults to 0.
            xtrn (np.ndarray, optional): Input array of size `(N,d)`. Needs to be user-provided: default produces assertion error.
            gtrn (np.ndarray, optional): Gradient array of size `(N,d)`. Needs to be user-provided: default produces assertion error.
        """
        super().__init__()
        self.nnmodel = nnmodel
        self.lam = lam
        assert(xtrn is not None)
        assert(gtrn is not None)

        self._xtrn = tch(xtrn, rgrad=True)
        self._gtrn = tch(gtrn, rgrad=True)

    def forward(self, inputs, targets):
        """Forward function.

        Args:
            inputs (torch.Tensor): Input tensor.
            targets (torch.Tensor): Target tensor.

        Returns:
            float: Loss value.
        """

        predictions = self.nnmodel(inputs)

        loss =  torch.mean((predictions-targets)**2)


        # outputs = self.nnmodel(self.xtrn)
        # outputs.requires_grad_()
        # der = torch.autograd.grad(outputs=outputs, inputs=self.xtrn,
        #                           grad_outputs=torch.ones_like(outputs),
        #                           create_graph=True, retain_graph=True, allow_unused=True)[0]
        # if der is not None:
        #     #print(der1.shape, self.gtrn.shape, der.shape)
        #     loss += self.lam*torch.mean((der-self.gtrn)**2)

        der = torch.vstack( [ F.jacobian(self.nnmodel, state, create_graph=True, strict=True).squeeze() for state in self.xtrn ] )


        loss += self.lam*torch.mean((der-self.gtrn)**2)

        return loss


########################################################
########################################################
########################################################

class NegLogPost(torch.nn.Module):
    r"""Negative log-posterior loss function.

    Attributes:
        nnmodel (callable): Model evaluator.
        priorparams (float): Dictionary of parameters of prior.
        sigma (float): Likelihood data noise standard deviation.
        fulldatasize (int): Full datasize. Important for weighting in case likelihood is computed on a batch.
        pi (float): 3.1415...

    The negative log-posterior has the form:

    .. math::
        \frac{N}{2}\log{(2\pi\sigma^2)} + \frac{1}{2\sigma^2}||M(x_{\text{train}})-y_{\text{train}}||^2 +
    .. math::
        +\frac{N}{N_{\text{full}}} \left(\frac{1}{2\sigma_{\text{prior}}^2}||w-w_{\text{anchor}}||^2 + \frac{K}{2} \log{(2\pi\sigma_{\text{prior}}^2)}\right).
    """

    def __init__(self, nnmodel, fulldatasize, sigma, priorparams):
        """Initialization.

        Args:
            nnmodel (callable): Model evaluator.
            fulldatasize (int): Full datasize. Important for weighting in case likelihood is computed on a batch.
            sigma (float): Likelihood data noise standard deviation.
            priorparams (float): Dictionary of parameters of prior. If None, there will be no prior.
        """
        super().__init__()
        self.nnmodel = nnmodel
        self.sigma = tch(float(sigma), rgrad=False)
        self.priorparams = priorparams
        self.pi = tch(np.pi, rgrad=False)
        self.fulldatasize = fulldatasize

    def forward(self, inputs, targets):
        """Forward function.

        Args:
            inputs (torch.Tensor): Input tensor.
            targets (torch.Tensor): Target tensor.

        Returns:
            float: Loss value.
        """

        predictions = self.nnmodel(inputs)
        neglogpost = 0.5 * torch.sum(torch.pow(targets - predictions, 2)) / self.sigma**2
        neglogpost += (len(predictions) / 2) * torch.log(2 * self.pi)
        neglogpost += len(predictions) * torch.log(self.sigma)

        if self.priorparams is not None:
            neglogprior_fcn = NegLogPrior(self.priorparams['sigma'], self.priorparams['anchor'])
            neglogpost += len(predictions)*neglogprior_fcn(self.nnmodel)/self.fulldatasize

        return neglogpost

########################################################
########################################################
########################################################

class NegLogPrior(torch.nn.Module):
    r"""Calculates a Gaussian negative log-prior.

    Attributes:
        anchor (torch.Tensor): Anchor, i.e. center vector of the gaussian prior.
        sigma (float): The standard deviation of the gaussian prior (same for all parameters).
        pi (float): 3.1415..

    The negative log-prior has the form:

    .. math::
        \frac{1}{2\sigma_{\text{prior}}^2}||w-w_{\text{anchor}}||^2 + \frac{K}{2} \log{(2\pi\sigma_{\text{prior}}^2)}.
    """

    def __init__(self, sigma, anchor):
        """
        Args:
            sigma (float): The standard deviation of the gaussian prior (same for all parameters).
            anchor (torch.Tensor): Anchor, i.e. center vector of the gaussian prior.
        """
        super().__init__()
        self.sigma = tch(float(sigma), rgrad=False)
        self.pi = tch(np.pi, rgrad=False)
        self.anchor = anchor


    def forward(self, model):
        """Forward evaluator

        Args:
            model (torch.nn.Module): The corresponding NN module.

        Returns:
            float: Negative log-prior value.
        """
        neglogprior = 0
        i = 0
        for p in model.parameters():
            cur_len = p.flatten().size()[0]
            neglogprior += torch.sum(
                torch.pow(p.flatten() - self.anchor[i : i + cur_len], 2) ) / 2 / self.sigma**2

            i += cur_len
        neglogprior += (i / 2) * torch.log(2 * self.pi * self.sigma**2)
        return neglogprior

########################################################
########################################################
########################################################

class CustomLoss(torch.nn.Module):
    r"""Example of custom one-dimensional loss function, including derivative and periodicity contraints. Quite experimental, but a base for developing problem-specific loss functions.

    Attributes:
        model (callable): Model evaluator.
        lam1 (float): Penalty strength for the periodicity constraint.
        lam2 (float): Penalty strength for the derivative constraint.

    The loss function has a form:

    .. math::
        \frac{1}{N}||y_{\text{pred}}-y_{\text{target}}||^2 + \lambda_1 (M(0.5)-M(-0.5))^2 + \lambda_2 (M'(0.5)-M'(-0.5))^2
    """

    def __init__(self, loss_params):
        """Initialization.

        Args:
            loss_params (tuple): (model, penalty1, penalty2) pair.
        """
        super().__init__()
        self.model, self.lam1, self.lam2 = loss_params

    def forward(self, predictions, targets):
        """Forward function.

        Args:
            predictions (torch.Tensor): Input tensor.
            targets (torch.Tensor): Target tensor.

        Returns:
            float: Loss value.
        """
        loss =  torch.mean((predictions-targets)**2)

        loss += self.lam1 * (self.model(torch.Tensor([0.5]))-self.model(torch.Tensor([-0.5])))**2

        x = torch.Tensor([-0.5, 0.5]).view(-1,1)
        x.requires_grad_()

        outputs = self.model(x)
        outputs.requires_grad_()
        der = torch.autograd.grad(outputs=outputs, inputs=x,
                                  grad_outputs=torch.ones_like(outputs),
                                  create_graph=True, retain_graph=True, allow_unused=True)[0]

        if der is not None: # in testing regimes, der is None
            reg = (der[0]-der[1])**2
        else:
            reg = 0.0

        loss += self.lam2*reg

        return loss

