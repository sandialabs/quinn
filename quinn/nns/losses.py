#!/usr/bin/env python

import torch
import numpy as np
from .tchutils import tch

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

class Loss_Gaussian(torch.nn.Module):
    """Example of custom loss function, including derivative contraints.

    Attributes:
        model (callable): Model evaluator.
        lam (torch.float): Penalty strength.
    """

    def __init__(self, model, sigma):
        """Initialization

        Args:
            model : torch module
            sigma : data noise
        """
        super().__init__()
        self.model = model
        self.sigma = tch(float(sigma), rgrad=False)
        self.pi = tch(np.pi, rgrad=False)

    def forward(self, inputs, targets):
        """Forward function.

        Args:
            inputs (torch.Tensor): Input tensor.
            targets (torch.Tensor): Target tensor.


        Returns:
            float: Loss value.
        """

        #if requires_grad: TODO: need to differentiate when loss gradient is needed or not, to avoide extra work
        predictions = self.model(inputs)  # , type_="torch")
        part1 = 0.5 * torch.sum(torch.pow(targets - predictions, 2)) / self.sigma**2
        part2 = (len(inputs) / 2) * torch.log(2 * self.pi)
        part2 += len(inputs) * torch.log(self.sigma)
        return part1 + part2
        # else:
        #     with torch.no_grad():




class PeriodicLoss(torch.nn.Module):
    """Example of periodic loss.

    Attributes:
        model (callable): Model evaluator.
        lam (float): Penalty strength.
        bdry1 (torch.Tensor): First boundary.
        bdry2 (torch.Tensor): Second boundary.
    """

    def __init__(self, loss_params):
        """Initialization.

        Args:
            loss_params (tuple): A 4-tuple of (model, penalty, boundary1, boundary2).
        """
        super().__init__()
        self.model, self.lam, self.bdry1, self.bdry2 = loss_params

    def forward(self, inputs, targets):
        """Forward function.

        Args:
            inputs (torch.Tensor): Input tensor.
            targets (torch.Tensor): Target tensor.

        Returns:
            float: Loss value.
        """
        tmp = (inputs-targets)**2
        fit = torch.mean(tmp)
        penalty = self.lam * torch.mean((self.model(self.bdry1)-self.model(self.bdry2))**2)
        loss =  fit + penalty
        #print(fit, penalty)

        return loss
