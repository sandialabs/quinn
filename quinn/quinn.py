#!/usr/bin/env python
"""Module for base QUiNN class."""

import copy
import functools
import numpy as np
import matplotlib.pyplot as plt

from .utils.plotting import plot_dm, lighten_color
from .utils.maps import scale01ToDom
from .utils.stats import get_stats, get_domain
from .nns.tchutils import print_nnparams


class QUiNNBase:
    """Base QUiNN class.

    Attributes:
        nens (int): Number of samples requested, `M`.
        nnmodel (torch.nn.Module): Underlying PyTorch NN model.
    """

    def __init__(self, nnmodel):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): Underlying PyTorch NN model.
        """
        self.nnmodel = copy.deepcopy(nnmodel)
        self.nens = None

    def print_params(self, names_only=False):
        """Print model parameter names and optionally, values.

        Args:
            names_only (bool, optional): Print names only. Default is False.
        """
        print_nnparams(self.nnmodel, names_only=names_only)

    def predict_sample(self, x):
        """Produce a single sample prediction.

        Args:
            x (np.ndarray): `(N,d)` input array.

        Raises:
            NotImplementedError: Not implemented in the base class.
        """
        raise NotImplementedError

    def predict_ens(self, x, nens=None):
        """Produce an ensemble of predictions.

        Args:
            x (np.ndarray): `(N,d)` input array.
            nens (int, optional): Number of samples requested, `M`.

        Returns:
            np.ndarray: Array of size `(M, N, o)`, i.e. `M` random samples of `(N,o)` outputs
        """
        if nens is None:
            nens = self.nens
        y_list = []
        for _ in range(nens):
            yy = self.predict_sample(x)
            y_list.append(yy)

        y = np.array(y_list)  # y.shape is nens, nsam(x.shape[0]), nout

        return y

    def predict_plot(
        self,
        xx_list,
        yy_list,
        nmc=100,
        plot_qt=False,
        labels=None,
        colors=None,
        iouts=None,
        msize=14,
        sigma=1,
        figname=None,
    ):
        """Plots the diagonal comparison figures.

        Args:
            xx_list (list[np.ndarray]): List of `(N,d)` inputs (e.g., training, validation, testing).
            yy_list (list[np.ndarray]): List of `(N,o)` outputs.
            nmc (int, optional): Requested number of samples for computing statistics, `M`.
            plot_qt (bool, optional): Whether to plot quantiles or mean/st.dev.
            labels (list[str], optional): List of labels. If None, set label internally.
            colors (list[str], optional): List of colors. If None, sets colors internally.
            iouts (list[int], optional): List of outputs to plot. If None, plot all.
            figname (str, optional): Name of the figure to be saved.

        Note:
            There is a similar function for deterministic NN in ``quinn.nns.nnbase``.
        """
        nlist = len(xx_list)
        assert nlist == len(yy_list)
        yy_pred_mb_list = []
        yy_pred_lb_list = []
        yy_pred_ub_list = []

        for xx in xx_list:
            yy_pred = self.predict_ens(xx, nens=nmc)
            yy_pred_mb, yy_pred_lb, yy_pred_ub = get_stats(yy_pred, plot_qt)
            # print(yy_pred.shape)
            yy_pred_mb_list.append(yy_pred_mb)
            yy_pred_lb_list.append(yy_pred_lb)
            yy_pred_ub_list.append(yy_pred_ub)

        nout = yy_pred_mb.shape[1]
        if iouts is None:
            iouts = range(nout)

        if labels is None:
            labels = [f"Set {i+1}" for i in range(nlist)]
        assert len(labels) == nlist

        if colors is None:
            colors = ["b", "g", "r", "c", "m", "y"] * nlist
            colors = colors[:nlist]
        assert len(colors) == nlist

        for iout in iouts:
            x1 = [yy[:, iout] for yy in yy_list]
            x2 = [yy[:, iout] for yy in yy_pred_mb_list]
            eel = [yy[:, iout] for yy in yy_pred_lb_list]
            eeu = [yy[:, iout] for yy in yy_pred_ub_list]
            ee = list(zip(eel, eeu))

            if figname is None:
                figname_ = "fitdiag_o" + str(iout) + ".png"
            else:
                figname = figname_.copy()

            plot_dm(
                x1,
                x2,
                errorbars=ee,
                labels=labels,
                colors=colors,
                axes_labels=[f"Model output # {iout+1}", f"Fit output # {iout+1}"],
                figname=figname_,
                legendpos="in",
                msize=msize,
            )

    def plot_1d_fits(
        self,
        xx_list,
        yy_list,
        domain=None,
        ngr=111,
        plot_qt=False,
        nmc=100,
        true_model=None,
        labels=None,
        colors=None,
        show_plot=False,
        save_plot=True,
        name_postfix="",
        ax=None,
    ):
        """Plotting one-dimensional slices, with the other dimensions at the nominal, of the fit.

        Args:
            xx_list (list[np.ndarray]): List of `(N,d)` inputs (e.g., training, validation, testing).
            yy_list (list[np.ndarray]): List of `(N,o)` outputs.
            domain (np.ndarray, optional): Domain of the function, `(d,2)` array. If None, sets it automatically based on data.
            ngr (int, optional): Number of grid points in the 1d plot.
            plot_qt (bool, optional): Whether to plot quantiles or mean/st.dev.
            nmc (int, optional): Requested number of samples for computing statistics, `M`.
            true_model (callable, optional): Optionally, plot a function
            labels (list[str], optional): List of labels. If None, set label internally.
            colors (list[str], optional): List of colors. If None, sets colors internally.
            name_postfix (str, optional): Postfix of the filename of the saved fig.

        Note:
            There is a similar function for deterministic NN in ``quinn.nns.nnbase``.
        """
        nlist = len(xx_list)
        assert nlist == len(yy_list)

        if labels is None:
            labels = [f"Set {i+1}" for i in range(nlist)]
        assert len(labels) == nlist

        if colors is None:
            colors = ["b", "g", "r", "c", "m", "y"] * nlist
            colors = colors[:nlist]
        assert len(colors) == nlist

        if domain is None:
            xall = functools.reduce(lambda x, y: np.vstack((x, y)), xx_list)
            domain = get_domain(xall)
        if ax is None:
            _ = plt.figure(figsize=(12, 8))

        if plot_qt:
            mlabel = "Median Pred."
            slabel = "Qtile"
        else:
            mlabel = "Mean Pred."
            slabel = "St.Dev."

        ndim = xx_list[0].shape[1]
        nout = yy_list[0].shape[1]
        for idim in range(ndim):
            xgrid_ = 0.5 * np.ones((ngr, ndim))
            xgrid_[:, idim] = np.linspace(0.0, 1.0, ngr)

            xgrid = scale01ToDom(xgrid_, domain)
            ygrid_pred = self.predict_ens(xgrid, nens=nmc)
            ygrid_pred_mb, ygrid_pred_lb, ygrid_pred_ub = get_stats(ygrid_pred, plot_qt)

            if ax is None:
                for iout in range(nout):
                    for j in range(nlist):
                        xx = xx_list[j]
                        yy = yy_list[j]

                        plt.plot(
                            xx[:, idim],
                            yy[:, iout],
                            colors[j] + "o",
                            markersize=13,
                            markeredgecolor="w",
                            label=labels[j],
                            zorder=1000,
                        )

                    if true_model is not None:
                        true = true_model(xgrid, 0.0)
                        plt.plot(
                            xgrid[:, idim],
                            true[:, iout],
                            "k-",
                            label="Truth",
                            alpha=0.5,
                        )

                    (p,) = plt.plot(
                        xgrid[:, idim],
                        ygrid_pred_mb[:, iout],
                        "m-",
                        linewidth=5,
                        label=mlabel,
                    )
                    for ygrid_pred_sample in ygrid_pred:
                        (p,) = plt.plot(
                            xgrid[:, idim],
                            ygrid_pred_sample[:, iout],
                            "m--",
                            linewidth=1,
                            zorder=-10000,
                        )
                    lc = lighten_color(p.get_color(), 0.5)
                    plt.fill_between(
                        xgrid[:, idim],
                        ygrid_pred_mb[:, iout] - ygrid_pred_lb[:, iout],
                        ygrid_pred_mb[:, iout] + ygrid_pred_ub[:, iout],
                        color=lc,
                        zorder=-1000,
                        alpha=0.9,
                        label=slabel,
                    )

                    plt.legend()
                    plt.xlabel(f"Input # {idim+1}")
                    plt.ylabel(f"Output # {iout+1}")
                    if show_plot:
                        plt.show()
                    if save_plot:
                        plt.savefig(
                            "fit_d"
                            + str(idim)
                            + "_o"
                            + str(iout)
                            + "_"
                            + name_postfix
                            + ".png"
                        )
                    plt.clf()
            elif ax is not None:
                for iout in range(nout):
                    for j in range(nlist):
                        xx = xx_list[j]
                        yy = yy_list[j]

                        ax.plot(
                            xx[:, idim],
                            yy[:, iout],
                            colors[j] + "o",
                            markersize=11,
                            markeredgecolor="w",
                            label=labels[j],
                            zorder=1000,
                        )

                    if true_model is not None:
                        true = true_model(xgrid, 0.0)
                        ax.plot(
                            xgrid[:, idim],
                            true[:, iout],
                            "k-",
                            label="Truth",
                            alpha=0.5,
                        )

                    (p,) = ax.plot(
                        xgrid[:, idim],
                        ygrid_pred_mb[:, iout],
                        "m-",
                        linewidth=5,
                        label=mlabel,
                    )

                    lc = lighten_color(p.get_color(), 0.5)
                    ax.fill_between(
                        xgrid[:, idim],
                        ygrid_pred_mb[:, iout] - ygrid_pred_lb[:, iout],
                        ygrid_pred_mb[:, iout] + ygrid_pred_ub[:, iout],
                        color=lc,
                        zorder=-1000,
                        alpha=0.9,
                        label=slabel,
                    )
