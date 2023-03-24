#!/usr/bin/env python

import functools
import numpy as np
import matplotlib.pyplot as plt

import torch

from .tchutils import npy, tch, nnfit

from ..utils.stats import get_domain
from ..utils.plotting import plot_dm
from ..utils.maps import scaleDomTo01, scale01ToDom

class MLPBase(torch.nn.Module):
    """Base class for an MLP architecture.

    Attributes:
        best_model (torch.nn.Module): Best trained instance, if any.
        indim (int): Input dimensionality.
        outdim (int): Output dimensionality.
        trained (bool): Whether the NN is already trained.
        history (list of np.ndarray): List containing training history, namely, [fepoch, loss_trn, loss_trn_full, loss_val]
    """

    def __init__(self, indim, outdim):
        """Initialization.

        Args:
            indim (int): Input dimensionality, `d`.
            outdim (int): Output dimensionality, `o`.
        """
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        self.best_model = None
        self.trained = False
        self.history = None


    def forward(self, x):
        """Forward function is not implemented in base class.

        Args:
            x (torch.Tensor): Input of the function.

        Raises:
            NotImplementedError: Need to implement it in children.
        """
        raise NotImplementedError

    def predict(self, x):
        """Prediction of the NN.

        Args:
            x (np.ndarray): Input array of size `(N,d)`.
            trained (bool, optional): Whether to evaluate with the best trained parameters or with the current parameters.

        Returns:
            np.ndarray: Output array of size `(N,o)`.

        Note:
            Both input and outputs are numpy arrays.
        """
        if self.trained:
            return npy(self.best_model(tch(x)))
        else:
            return npy(self.forward(tch(x)))

    def numpar(self):
        """Get the number of parameters of NN.

        Returns:
            int: Number of parameters, trainable or not.
        """
        pdim = sum(p.numel() for p in self.parameters())
        return pdim

    def fit(self, xtrn, ytrn, **kwargs):
        """Fit function.

        Args:
            xtrn (np.ndarray): Input array of size `(N,d)`.
            ytrn (np.ndarray): Output array of size `(N,o)`.
            **kwargs (dict): Keyword arguments.

        Returns:
            torch.nn.Module: Best trained instance.
        """
        #self.fitdict = locals()
        fit_info = nnfit(self, xtrn, ytrn, **kwargs)
        self.best_model = fit_info['best_nnmodel']
        self.history = fit_info['history']
        self.trained = True


        return self.best_model


    def printParams(self):
        """Print parameter names and values."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)


    def printParamNames(self):
        """Print parameter names and shapes."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)


    def predict_plot(self, xx_list, yy_list, labels=None, colors=None, iouts=None):
        """Plots the diagonal comparison figures.

        Args:
            xx_list (list[np.ndarray]): List of `(N,d)` inputs (e.g., training, validation, testing).
            yy_list (list[np.ndarray]): List of `(N,o)` outputs.
            labels (list[str], optional): List of labels. If None, set label internally.
            colors (list[str], optional): List of colors. If None, sets colors internally.
            iouts (list[int], optional): List of outputs to plot. If None, plot all.

        Note:
            There is a similar function for probabilistic NN in ``quinn.quinn``.
        """
        nlist = len(xx_list)
        assert(nlist==len(yy_list))


        yy_pred_list = []
        for xx in xx_list:
            yy_pred = self.predict(xx)
            yy_pred_list.append(yy_pred)

        nout = yy_pred.shape[1]
        if iouts is None:
            iouts = range(nout)

        if labels is None:
            labels = [f'Set {i+1}' for i in range(nlist)]
        assert(len(labels)==nlist)

        if colors is None:
            colors = ['b', 'g', 'r', 'c', 'm', 'y']*nlist
            colors = colors[:nlist]
        assert(len(colors)==nlist)

        for iout in iouts:
            x1 = [yy[:, iout] for yy in yy_list]
            x2 = [yy[:, iout] for yy in yy_pred_list]

            plot_dm(x1, x2, labels=labels, colors=colors,
                    axes_labels=[f'Model output # {iout+1}', f'Fit output # {iout+1}'],
                    figname='fitdiag_o'+str(iout)+'.png',
                    legendpos='in', msize=13)

    def plot_1d_fits(self, xx_list, yy_list, domain=None, ngr=111, true_model=None, labels=None, colors=None):
        """Plotting one-dimensional slices, with the other dimensions at the nominal, of the fit.

        Args:
            xx_list (list[np.ndarray]): List of `(N,d)` inputs (e.g., training, validation, testing).
            yy_list (list[np.ndarray]): List of `(N,o)` outputs.
            domain (np.ndarray, optional): Domain of the function, `(d,2)` array. If None, sets it automatically based on data.
            ngr (int, optional): Number of grid points in the 1d plot.
            true_model (callable, optional): Optionally, plot a function
            labels (list[str], optional): List of labels. If None, set label internally.
            colors (list[str], optional): List of colors. If None, sets colors internally.
            name_postfix (str, optional): Postfix of the filename of the saved fig.

        Note:
            There is a similar function for probabilistic NN in ``quinn.quinn``.
        """

        nlist = len(xx_list)
        assert(nlist==len(yy_list))

        if labels is None:
            labels = [f'Set {i+1}' for i in range(nlist)]
        assert(len(labels)==nlist)

        if colors is None:
            colors = ['b', 'g', 'r', 'c', 'm', 'y']*nlist
            colors = colors[:nlist]
        assert(len(colors)==nlist)

        if domain is None:
            xall = functools.reduce(lambda x,y: np.vstack((x,y)), xx_list)
            domain = get_domain(xall)

        mlabel = 'Mean Pred.'

        ndim = xx_list[0].shape[1]
        nout = yy_list[0].shape[1]
        for idim in range(ndim):
            xgrid_ = 0.5 * np.ones((ngr, ndim))
            xgrid_[:, idim] = np.linspace(0., 1., ngr)

            xgrid = scale01ToDom(xgrid_, domain)
            ygrid_pred = self.predict(xgrid)

            for iout in range(nout):

                for j in range(nlist):
                    xx = xx_list[j]
                    yy = yy_list[j]

                    plt.plot(xx[:, idim], yy[:, iout], colors[j]+'o', markersize=13, markeredgecolor='w', label=labels[j])

                if true_model is not None:
                    true = true_model(xgrid, 0.0)
                    plt.plot(xgrid[:, idim], true[:, iout], 'k-', label='Truth', alpha=0.5)


                p, = plt.plot(xgrid[:, idim], ygrid_pred[:, iout], 'm-', linewidth=5, label=mlabel)


                plt.legend()
                plt.xlabel(f'Input # {idim+1}')
                plt.ylabel(f'Output # {iout+1}')
                plt.savefig('fit_d' + str(idim) + '_o' + str(iout) + '.png')
                plt.clf()

