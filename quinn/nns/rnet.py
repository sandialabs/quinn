#!/usr/bin/env python
"""Module containing ResNet class and layer weight parameterization class."""

import math
import torch
import torch.nn.functional as F

from .nnbase import MLPBase


########################################################################
########################################################################
########################################################################


class RNet(MLPBase):
    """Residual Neural Network (ResNet) class.

    Attributes:
        indim (int): Input dimensionality `d`.
        outdim (int): Output dimensionality `o`.
        nlayers (int): Number of layers `L`.
        step_size (float): Time step size, `1/(L+1)`.
        rdim (int): Width of the ResNet `r`, i.e. number of units in each hidden layer.
        activ (torch.nn.Module): Activation function.
        biasorno (bool): Whether or not to include biases in each resnet layer.
        mlp (bool): If True, residual connections are ignored, and this becomes a regular MLP.
        layer_pre (bool): Whether there is a pre-resnet linear layer.
        layer_post (bool): Whether there is a post-resnet linear layer.
        final_layer (str): If there is a final layer function. The only current option is 'exp' for exponential function.
        wp_function (LayerFcn): Weight parameterization function.
        weight_pre (torch.nn.Parameter): Weight matrix of pre-resnet layer, if any.
        bias_pre (torch.nn.Parameter): Bias vector of pre-resnet layer, if any.
        weight_post (torch.nn.Parameter): Weight matrix of post-resnet layer, if any.
        bias_post (torch.nn.Parameter): Bias vector of post-resnet layer, if any.
        paramsw (list[torch.nn.Parameter]): List of Resnet weight matrices.
        paramsb (list[torch.nn.Parameter]): List of Resnet bias vectors.
        device (str): It represents where computations are performed and tensors are allocated. Default to cpu.
    """
    def __init__(self, rdim, nlayers, wp_function=None, indim=None,
                       outdim=None, biasorno=True, nonlin=True, mlp=False,
                       layer_pre=False, layer_post=False,final_layer=None,
                       device='cpu', init_factor=1, sum_dim=1):
        """Instantiate ResNet object.

        Args:
            rdim (int): Width of the ResNet `r`, i.e. number of units in each hidden layer.
            nlayers (int): Number of layers `L`.
            wp_function (LayerFcn, optional): Weight parameterization function. Defaults to a regular ResNet without weight parameterization.
            indim (int, optional): Input dimensionality `d`. Defaults to width `r`.
            outdim (int, optional): Output dimensionality `o`. Defaults to width `r`.
            biasorno (bool, optional): Whether or not to include biases in each resnet layer. Default is True.
            nonlin (bool, optional): Whether to use nonlinear activation function between layers. Defaults to True.
            mlp (bool, optional): If True, residual connections are ignored, and this becomes a regular MLP. Default is False.
            layer_pre (bool, optional): Whether there is a pre-resnet linear layer. Defaults to False.
            layer_post (bool, optional): Whether there is a post-resnet linear layer. Defaults to False.
            final_layer (str, optional): If there is a final layer function. Two options: "exp" for exponential function; "sum" for sum function which will reduce rank of the output tensor. Defaults to no final layer.
            sum_dim (int, optional): If final layer function is sum, it will select i which dimension to perform sum. Defaults to 1  
            device (str): It represents where computations are performed and tensors are allocated. Default to cpu.
            init_factor(int): Multiply initial condition tensors by factor.
        """
        super().__init__(indim, outdim, device=device)
        if self.indim is None:
            self.indim = rdim
        if self.outdim is None:
            self.outdim = rdim

        self.nlayers = nlayers
        self.biasorno = biasorno
        if wp_function is None:
            self.wp_function = NonPar(nlayers+1)
        else:
            assert(isinstance(wp_function, LayerFcn))
            self.wp_function = wp_function
        self.step_size = 1.0 / (nlayers + 1.0)
        self.mlp = mlp
        self.layer_pre = layer_pre
        self.layer_post = layer_post
        self.final_layer = final_layer
        self.init_factor = init_factor
        # only for final_layer=sum
        self.sum_dim = sum_dim

        self.rdim = rdim

        if self.indim != self.rdim:
            assert self.layer_pre
        if self.outdim != self.rdim:
            assert self.layer_post

        if self.layer_pre:
            self.weight_pre = torch.nn.Parameter(self.init_factor*(2. * torch.rand(self.rdim, self.indim) -1.)/math.sqrt(self.indim))
            self.bias_pre = torch.nn.Parameter(self.init_factor*(2. * torch.rand(self.rdim) -1.)/math.sqrt(self.indim))

        if self.layer_post:
            self.weight_post = torch.nn.Parameter(self.init_factor*(2. * torch.rand(self.outdim, self.rdim) -1.)/math.sqrt(self.rdim))
            self.bias_post = torch.nn.Parameter(self.init_factor*(2. * torch.rand(self.outdim) -1.)/math.sqrt(self.rdim))

        pars_w = []
        for ip in range(self.wp_function.npar):
            ww = torch.nn.Parameter(self.init_factor*(2. * torch.rand(self.rdim, self.rdim) -1.)/math.sqrt(self.rdim))
            pars_w.append(ww)
            self.register_parameter(name='ww_'+str(ip), param=ww)
            #pars_w.append(torch.nn.Parameter(torch.randn(rdim, rdim)))
        #self.paramsw = pars_w #torch.nn.ParameterList(pars_w)

        if self.biasorno:
            pars_b = []
            for ip in range(self.wp_function.npar):
                bb = torch.nn.Parameter(self.init_factor*(2.*torch.rand(self.rdim)-1.)/math.sqrt(self.rdim))
                pars_b.append(bb)
                self.register_parameter(name='bb_'+str(ip), param=bb)

                #pars_b.append(torch.nn.Parameter(torch.randn(rdim)))
            #self.paramsb = pars_b #torch.nn.ParameterList(pars_b)


        if nonlin:
            self.activ = torch.nn.Tanh()
        else:
            self.activ = torch.nn.Identity()

        self.to(device)    

    def forward(self, x):
        r"""Forward function.

        Args:
            x (torch.Tensor): Input tensor `x` of size :math:`(N,d)`.

        Returns:
            torch.Tensor: Output tensor of size :math:`(N,o)`.
        """
        out = x+0.0

        # Note that the prelayer has activation, too, to avoid two linear layers in succession
        if self.layer_pre:
            out = self.activ(F.linear(out, self.weight_pre, self.bias_pre))

        paramsw = [getattr(self, 'ww_'+str(ip)) for ip in range(self.wp_function.npar)]
        if self.biasorno:
            paramsb = [getattr(self, 'bb_'+str(ip)) for ip in range(self.wp_function.npar)]

        for i in range(self.nlayers+1):
            weight = self.wp_function(paramsw, self.step_size * i)
            if self.biasorno:
                bias = self.wp_function(paramsb, self.step_size * i)
            else:
                bias = None

            if self.mlp:
                out = self.activ(F.linear(out, weight, bias))
            else:
                out = out + self.step_size * self.activ(F.linear(out, weight, bias))

        if self.layer_post:
            out = F.linear(out, self.weight_post, self.bias_post)
        if self.final_layer == "exp":
            out = torch.exp(out)
        elif self.final_layer == "sum":
            out = torch.sum(out,dim=self.sum_dim)
            
        return out

    # def getParams(self):
    #     """Get parameters of the ResNet.

    #     Returns:
    #         list[torch.nn.Parameter] or (list[torch.nn.Parameter], list[torch.nn.Parameter]): List of weights or a tuple containing list of weights and list of biases.
    #     """
    #     if self.biasorno:
    #         return self.paramsw, self.paramsb
    #     else:
    #         return self.paramsw

    # def setParams(self, paramsw, paramsb=None):
    #     """Setting the parameters.

    #     Args:
    #         paramsw (list[torch.nn.Parameter]): List of weight matrices.
    #         paramsb (list[torch.nn.Parameter], optional): List of bias vectors, if any.
    #     """
    #     if self.biasorno:
    #         self.paramsw = paramsw
    #         assert(paramsb is not None)
    #         self.paramsb = paramsb
    #     else:
    #         self.paramsw = paramsw
    #         assert(paramsb is None)

########################################################################
########################################################################
########################################################################

class LayerFcn():
    """Base class for layer weight parameterization layer functions.

    Attributes:
        npar (int): Number of parameters in the parameterization (parameters can be Tensors).
    """

    def __init__(self):
        """Instantiation."""
        self.npar=None

    def __call__(self, pars, t):
        """Call signature.

        Args:
            pars (list[torch.nn.Parameter]): List of parameters.
            t (float): 'Time', i.e. layer number.
        Raises:
            NotImplementedError: Need to implement it in children.
        """
        raise NotImplementedError

class Const(LayerFcn):
    """Constant weight parameterization.

    Attributes:
        npar (int): Number of parameters. Should be 1.
    """

    def __init__(self):
        """Instantiation."""
        super().__init__()
        self.npar = 1

    def __call__(self, pars, t):
        """Call function.

        Args:
            pars (list[torch.nn.Parameter]): List of parameters.
            t (float): 'Time', i.e. layer number.

        Returns:
            torch.nn.Parameter: Constant (independent of `t`).
        """
        assert(len(pars) == self.npar)
        return pars[0]


class Lin(LayerFcn):
    """Linear weight parameterization.

    Attributes:
        npar (int): Number of parameters. Should be 2.
    """
    def __init__(self):
        """Instantiation."""
        super().__init__()
        self.npar = 2

    def __call__(self, pars, t):
        """Call function.

        Args:
            pars (list[torch.nn.Parameter]): List of parameters.
            t (float): 'Time', i.e. layer number.

        Returns:
            torch.nn.Parameter: Linear in `t`.
        """
        assert(len(pars) == self.npar)
        return pars[0] + pars[1] * t


class Quad(LayerFcn):
    """Quadratic weight parameterization.

    Attributes:
        npar (int): Number of parameters. Should be 3.
    """
    def __init__(self):
        """Instantiation."""
        super().__init__()
        self.npar = 3

    def __call__(self, pars, t):
        """Call function.

        Args:
            pars (list[torch.nn.Parameter]): List of parameters.
            t (float): 'Time', i.e. layer number.

        Returns:
            torch.nn.Parameter: Quadratic in `t`.
        """
        assert(len(pars) == self.npar)
        return pars[0] + pars[1] * t + pars[2] * t**2

class Cubic(LayerFcn):
    """Cubic weight parameterization.

    Attributes:
        npar (int): Number of parameters. Should be 4.
    """
    def __init__(self):
        """Instantiation."""
        super().__init__()
        self.npar = 4

    def __call__(self, pars, t):
        """Call function.

        Args:
            pars (list[torch.nn.Parameter]): List of parameters.
            t (float): 'Time', i.e. layer number.

        Returns:
            torch.nn.Parameter: Cubic in `t`.
        """
        assert(len(pars) == self.npar)
        return pars[0] + pars[1] * t + pars[2] * t**2 + pars[3] * t**3

class Poly(LayerFcn):
    """Polynomial weight parameterization.

    Attributes:
        npar (int): Number of parameters.
    """
    def __init__(self, order):
        """Instantiation.

        Args:
            order (int): Order of the polynomial.
        """
        super().__init__()
        self.npar = order+1

    def __call__(self, pars, t):
        """Call function.

        Args:
            pars (list[torch.nn.Parameter]): List of parameters.
            t (float): 'Time', i.e. layer number.

        Returns:
            torch.nn.Parameter: Polynomial in `t`.
        """
        assert(len(pars) == self.npar)

        val = 0.0
        for i in range(self.npar):
            val += pars[i]*t**i
        return val

class NonPar(LayerFcn):
    """Non-parameteric weight parameterization, i.e. regular ResNet without weight parameterization.

    Attributes:
        npar (int): Number of parameters.
    """
    def __init__(self, npar):
        """Instantiation.

        Args:
            npar (int): Should be one more than the number of layers `L`.
        """
        super().__init__()
        self.npar = npar

    def __call__(self, pars, t):
        """Call function.

        Args:
            pars (list[torch.nn.Parameter]): List of parameters.
            t (float): 'Time', i.e. layer number.

        Returns:
            torch.nn.Parameter: Non-parameteric: a new parameter per `t` value (i.e. per layer).
        """

        #print(len(pars), self.npar, t, t * self.npar, int(t * self.npar))
        assert(len(pars) == self.npar)
        return pars[int(t * self.npar)]

