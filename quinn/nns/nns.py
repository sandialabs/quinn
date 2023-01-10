#!/usr/bin/env python
"""Module containing various simple PyTorch NN modules."""


import math
import torch

class Gaussian(torch.nn.Module):
    r"""Gaussian function. :math:`\textrm{Gaussian}(x) = e^{-x^2}`
    """
    def __init__(self):
        """Initialization."""
        super().__init__()

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor `x`.

        Returns:
            torch.Tensor: Output tensor of same size as input `x`.
        """
        return torch.exp(-x**2)


class Sine(torch.nn.Module):
    r"""Sine function. :math:`\textrm{Sin}(x) = A\sin\left(2\pi x/T\right)`"""
    def __init__(self, A=1.0, T=1.0):
        """Initialization.

        Args:
            A (float, optional): Amplitude `A`. Defaults to 1.
            T (float, optional): Period `T`. Defaults to 1.
        """
        super().__init__()
        self.A = A
        self.T = T

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor `x`.

        Returns:
            torch.Tensor: Output tensor of same size as input `x`.
        """

        return torch.sin(self.A*torch.Tensor(math.pi)*x/self.T)


class Polynomial(torch.nn.Module):
    r"""Polynomial function :math:`\textrm{Polynomial}(x)=\sum_{i=0}^p c_i x^i`.

    Attributes:
        order (int): Order of the polynomial.
        coefs (torch.nn.Parameter): Coefficient array of size `p+1`.
    """

    def __init__(self, order):
        """Initialization.

        Args:
            order (int): Order of the polynomial.
        """
        super().__init__()
        self.order = order

        self.coefs= torch.nn.Parameter(torch.randn((self.order+1,)))

        # Parameter List does not work with quinn.vi.vi
        # self.coefs= torch.nn.ParameterList([torch.nn.Parameter(torch.randn(())) for i in range(self.order+1)])

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor `x`.

        Returns:
            torch.Tensor: Output tensor of same size as input `x`.
        """


        val = torch.zeros_like(x)
        for i, cf in enumerate(self.coefs):
            val += cf*x**i

        return val


class Polynomial3(torch.nn.Module):
    r"""Example 3-rd order polynomial function :math:`\textrm{Polynomial3}(x)=a+bx+cx^2+dx^3`.

    Attributes:
        a (torch.nn.Parameter): Constant coefficient.
        b (torch.nn.Parameter): First-order coefficient.
        c (torch.nn.Parameter): Second-order coefficient.
        d (torch.nn.Parameter): Third-order coefficient.
    """

    def __init__(self):
        """Instantiate four parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor `x`.

        Returns:
            torch.Tensor: Output tensor of same size as input `x`.
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

class Constant(torch.nn.Module):
    r"""Constant function :math:`\textrm{Constant}(x)=C`.

    Attributes:
        constant (torch.nn.Parameter): Constant `C`.
    """

    def __init__(self):
        """Instantiate the constant."""
        super().__init__()
        self.constant = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor `x`.

        Returns:
            torch.Tensor: Output tensor of same size as input `x`.
        """
        return self.constant * torch.ones_like(x)


class SiLU(torch.nn.Module):
    r"""Sigmoid Linear Unit (SiLU) function :math:`\textrm{SiLU}(x) = x \sigma(x) = \frac{x}{1+e^{-x}}`
    """
    def __init__(self):
        """Initialization. """
        super().__init__()

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor `x`.

        Returns:
            torch.Tensor: Output tensor of same size as input `x`.
        """
        return x * torch.sigmoid(x)

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class Expon(torch.nn.Module):
    r"""Exponential function :math:`\textrm{Expon}(x) = e^{x}`
    """
    def __init__(self):
        """Initialization. """
        super().__init__()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor `x`.

        Returns:
            torch.Tensor: Output tensor of same size as input `x`.
        """
        return torch.exp(x)

class TwoLayerNet(torch.nn.Module):
    """Example two-layer function, with a cubic polynomical between layers.

    Attributes:
        linear1 (torch.nn.Linear): First linear layer.
        linear2 (torch.nn.Linear): Second linear layer.
        cubic (torch.nn.Module): Cubic layer in-between the linear ones.
    """

    def __init__(self, D_in, H, D_out):
        r"""Initializes give the input, output dimensions and the hidden width.

        Args:
            D_in (int): Input dimension :math:`d_{in}`.
            H (int): Hidden layer width.
            D_out (int): Output dimension :math:`d_{out}`.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.cubic = Polynomial3()

    def forward(self, x):
        r"""Forward function.

        Args:
            x (torch.Tensor): Input tensor `x` of size :math:`(N,d_{in})`.

        Returns:
            torch.Tensor: Output tensor of size :math:`(N,d_{out})`.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.cubic(h_relu)
        y_pred = self.linear2(y_pred)

        return y_pred


class MLP_simple(torch.nn.Module):
    r"""Simple MLP example.

    Attributes:
        biasorno (bool): Whether to use bias or not.
        hls (tuple[int]): List of layer widths.
        indim (int): Input dimensionality :math:`d_{in}`.
        outdim (int): Output dimensionality :math:`d_{out}`.
        model (torch.nn.Sequential): The PyTorch Sequential model behind the forward function.

    Note:
        Uses :math:`\tanh(x)` as activation function between layers.
    """

    def __init__(self, hls, biasorno=True):
        """Initialization.

        Args:
            hls (tuple[int]): Tuple  of number of units per layer, length  of list if  number of layers
            biasorno (bool, optional): Whether to use bias or not. Defaults to True.
        """
        super().__init__()
        assert(len(hls)>1)
        self.hls = hls[1:-1]
        self.indim = hls[0]
        self.outdim = hls[-1]
        self.biasorno = biasorno

        modules = []
        for j in range(len(hls)-2):
            modules.append(torch.nn.Linear(hls[j], hls[j+1], self.biasorno))
            modules.append(torch.nn.Tanh())
        modules.append(torch.nn.Linear(hls[-2], hls[-1], bias=self.biasorno))

        self.model = torch.nn.Sequential(*modules)

    def forward(self, x):
        r"""Forward function.

        Args:
            x (torch.Tensor): Input tensor `x` of size :math:`(N,d_{in})`.

        Returns:
            torch.Tensor: Output tensor of size :math:`(N,d_{out})`.
        """

        return self.model(x)


