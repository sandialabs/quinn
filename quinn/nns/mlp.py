#!/usr/bin/env python

import torch

from .nns import Expon, Sine
from .nnbase import MLPBase

class MLP(MLPBase):
    """Multilayer perceptron class.

    Attributes:
        hls (tuple): Tuple of hidden layer widths.
        biasorno (bool): Whether biases are included or not.
        bnorm (bool): Whether batch normalization is implemented or not.
        bnlearn (bool): Whether batch normalization is learnable or not.
        dropout (float): Dropout fraction.
        final_transform (str): Final transformation. Currently only 'exp' is implemented for Exponential.
        nlayers (int): Number of layers.
        nnmodel (torch.nn.Module): Underlying model evaluator.
    """

    def __init__(self, indim, outdim, hls, biasorno=True,
                 activ='relu', bnorm=False, bnlearn=True, dropout=0.0,
                 final_transform=None, device='cpu'):
        """Initialization.

        Args:
            indim (int): Input dimensionality.
            outdim (int): Output dimensionality.
            hls (tuple): Tuple of hidden layer widths.
            biasorno (bool): Whether biases are included or not.
            activ (str, optional): Activation function. Options are 'tanh', 'relu', 'sin' or else identity is used.
            bnorm (bool): Whether batch normalization is implemented or not.
            bnlearn (bool): Whether batch normalization is learnable or not.
            dropout (float, optional): Dropout fraction. Default is 0.0.
            final_transform (str, optional): Final transform, if any (onle 'exp' is implemented). Default is None.
            device (str): It represents where computations are performed and tensors are allocated. Default is cpu.
        """
        super(MLP, self).__init__(indim, outdim, device=device)

        self.nlayers = len(hls)
        assert(self.nlayers > 0)
        self.hls = hls
        self.biasorno = biasorno
        self.dropout = dropout
        self.bnorm = bnorm
        self.bnlearn = bnlearn
        self.final_transform = final_transform

        if activ == 'tanh':
            activ_fcn = torch.nn.Tanh()
        elif activ == 'relu':
            activ_fcn = torch.nn.ReLU()
        elif activ == 'sin':
            activ_fcn = Sine()
        else:
            activ_fcn = torch.nn.Identity()

        modules = []
        modules.append(torch.nn.Linear(self.indim, self.hls[0], self.biasorno))
        if self.dropout > 0.0:
            modules.append(torch.nn.Dropout(p=self.dropout))

        if self.bnorm:
            modules.append(torch.nn.BatchNorm1d(self.hls[0], affine=self.bnlearn))
        for i in range(1, self.nlayers):
            modules.append(activ_fcn)
            modules.append(torch.nn.Linear(self.hls[i - 1], self.hls[i], self.biasorno))
            if self.dropout > 0.0:
                modules.append(torch.nn.Dropout(p=self.dropout))
            if self.bnorm:
                modules.append(torch.nn.BatchNorm1d(self.hls[i], affine=self.bnlearn))


        modules.append(activ_fcn)
        modules.append(torch.nn.Linear(self.hls[-1], self.outdim, bias=self.biasorno))
        if self.dropout > 0.0:
            modules.append(torch.nn.Dropout(p=self.dropout))
        if self.bnorm:
            modules.append(torch.nn.BatchNorm1d(self.outdim, affine=self.bnlearn))

        if self.final_transform=='exp':
            modules.append(Expon())


        self.nnmodel = torch.nn.Sequential(*modules)
        # sync model to device
        self.to(device)



    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.nnmodel(x)

