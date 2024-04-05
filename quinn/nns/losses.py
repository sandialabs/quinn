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
        \frac{1}{N}||y_{\text{pred}}-y_{\text{target}}||^2 + \frac{\lambda}{N}||\text{model(boundary1)}-\text{model(boundary2)}||^2.
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
    """Example of custom loss function, including derivative contraints.

    Attributes:
        model (callable): Model evaluator.
        lam (torch.float): Penalty strength.
    """

    def __init__(self, nnmodel, fulldatasize, sigma, priorparams):
        """Initialization

        Args:
            sigma : data noise
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
    """Calculates the probability of the parameters given a Gaussian prior.

    Attributes:
        - sigma: the standard deviation of the gaussian prior (same for all
        parameters). Type: float.
        - n_params: number of parameters being sampled. Type: int.
        - pi: number pi. Type: torch.Tensor.
        - anchor (torch.Tensor): vector parameter sampled from the prior that acts
        as anchor of the prior calculation.
    """

    def __init__(self, sigma, anchor):
        """
        Args:
            - sigma: the standard deviation of the gaussian prior (same for
            all parameters). Type: float.
            - n_params: number of parameters being sampled. Type: int.
        """
        super().__init__()
        self.sigma = tch(float(sigma), rgrad=False)
        self.pi = tch(np.pi, rgrad=False)  # torch.Tensor(tch(np.pi, rgrad=False))
        self.anchor = anchor


    def forward(self, model):

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
    """Example of custom loss function, including derivative contraints.

    Attributes:
        model (callable): Model evaluator.
        lam (torch.float): Penalty strength.
    """

    def __init__(self, loss_params):
        """Initialization

        Args:
            loss_params (tuple): (model, penalty) pair
        """
        super().__init__()
        self.model, self.lam = loss_params

    def forward(self, inputs, targets):
        """Forward function.

        Args:
            inputs (torch.Tensor): Input tensor.
            targets (torch.Tensor): Target tensor.

        Returns:
            float: Loss value.
        """
        tmp = (inputs-targets)**2
        loss =  torch.mean(tmp)+self.lam * (self.model(torch.Tensor([0.5]))-self.model(torch.Tensor([-0.5])))**2

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
        return loss+100.*reg

