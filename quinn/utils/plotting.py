#!/usr/bin/env python
"""Module for various plotting functions."""

import os
import sys
import warnings
import numpy as np

import scipy.stats as ss
from scipy.interpolate import interp1d

import colorsys
import matplotlib as mpl
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .xutils import get_pdf, sample_sphere, pick_basis, project, strarr
from .maps import scale01ToDom

#############################################################

warnings.simplefilter(action='ignore', category=FutureWarning)

#############################################################

def myrc():
    """Configure matplotlib common look and feel.

    Returns:
        dict: Dictionary of matplotlib parameter config, not really used
    """
    mpl.rc('legend', loc='best', fontsize=22)
    mpl.rc('lines', linewidth=4, color='r')
    mpl.rc('axes', linewidth=3, grid=True, labelsize=22)
    mpl.rc('xtick', labelsize=15)
    mpl.rc('ytick', labelsize=15)
    mpl.rc('font', size=20)
    mpl.rc('figure', figsize=(12, 9), max_open_warning=200)
    #mpl.rc('lines', markeredgecolor='w')
    # mpl.rc('font', family='serif')

    return mpl.rcParams

#############################################################

def saveplot(figname):
    """Save a figure with warnings ignored.

    Args:
        figname (mpl.Figure): Figure handle.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.savefig(figname)

#############################################################

def set_colors(npar):
    """Sets a list of different colors of requested length, as rgb triples.

    Args:
        npar (int): Number of parameters.
    Returns:
        list[tuple]: List of rgb triples.
    """
    colors = []
    pp=1+int(npar/6)
    for i in range(npar):
        c=1-float(int((i/6))/pp)
        b=np.empty((3))
        for jj in range(3):
            b[jj]=c*int(i%3==jj)
        a=int(int(i%6)/3)
        colors.append(((1-a)*b[2]+a*(c-b[2]),(1-a)*b[1]+a*(c-b[1]),(1-a)*b[0]+a*(c-b[0])))

    return colors

#############################################################

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Args:
        color (str or tuple): Initial color.
        amount (float): How much to lighten: should be between 0 and 1.

    Returns:
        str: lightened color.

    Examples:
        >>> lighten_color('g', 0.3)
        >>> lighten_color('#F034A3', 0.6)
        >>> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    return colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])


#############################################################

def plot_dm(datas, models, errorbars=None, labels=None, colors=None,
            axes_labels=['Model', 'Apprx'], figname='dm.png',
            legendpos='in', msize=4, alpha=1.0):
    """Plots data-vs-model and overlays `y=x`.

    Args:
        datas (list[np.ndarray]): List of `K` 1d data arrays.
        models (list[np.ndarray]): List of `K` 1d model arrays with matching sizes to data arrays.
        errorbars (list[(np.ndarray, np.ndarray)], optional): List of `K` (lower,upper) tuples as errorbars for each model array. Can be None.
        labels (list[str], optional): List of `K` labels. If None, the code uses something generic.
        colors (list[str], optional): List of `K` colors. If None, the code chooses internally.
        axes_labels (list[str], optional): List of two strings, x- and y-axis labels.
        figname (str, optional): Figure file name.
        legendpos (str, optional): Legend position, 'in' or 'out'
        msize (int, optional): Marker size.
        alpha (float, optional): Marker opacity, between `0` and `1`. Defaults to 1.
    """
    if errorbars is None:
        erb = False
    else:
        erb = True

    custom_xlabel = axes_labels[0]
    custom_ylabel = axes_labels[1]

    if legendpos == 'in':
        fig = plt.figure(figsize=(10, 10))
    elif legendpos == 'out':
        fig = plt.figure(figsize=(14, 10))
        fig.add_axes([0.1, 0.1, 0.6, 0.8])

    ncase = len(datas)
    if labels is None:
        labels = [''] * ncase

    # Create colors list
    if colors is None:
        colors = set_colors(ncase)

    yy = np.empty((0, 1))
    for i in range(ncase):
        data = datas[i]
        model = models[i]
        if erb:
            erbl, erbh = errorbars[i]
        npts = data.shape[0]
        neach = 1
        if (data.ndim > 1):
            neach = data.shape[1]

        ddata = data.reshape(npts, neach)

        for j in range(neach):
            yy = np.append(yy, ddata[:, j])
            if (erb):
                plt.errorbar(ddata[:, j], model, yerr=[erbl, erbh],
                             fmt='o', markersize=msize,
                             markeredgecolor='w',
                             color=colors[i],
                             ecolor=colors[i], label=labels[i], alpha=alpha)
            else:
                plt.plot(ddata[:, j], model, 'o', color=colors[i], label=labels[i], markeredgecolor='w', markersize=msize, alpha=alpha)

    delt = 0.03 * (yy.max() - yy.min())
    minmax = [yy.min() - delt, yy.max() + delt]
    plt.plot(minmax, minmax, 'k--', linewidth=2)

    plt.xlabel(custom_xlabel)
    plt.ylabel(custom_ylabel)
    # plt.title('Data vs Model')
    if legendpos == 'in':
        plt.legend()
    elif legendpos == 'out':
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                   ncol=1, fancybox=True, shadow=True)


    # plt.xscale('log')
    # plt.yscale('log')

    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.axis('scaled')
    # plt.axis('equal')

    # Trying to make sure both axis have the same number of ticks
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(7))
    plt.savefig(figname)
    plt.clf()

#############################################################

def plot_xrv(xsam, prefix='xsam'):
    """Plotting samples one at a time, and one dimension vs the other.

    Args:
        xsam (np.ndarray): A `(N,d)` numpy array of samples
        prefix (str, optional): Prefix for filenames of the figures.
    """
    nsam, ndim = xsam.shape
    for i in range(ndim):
        plt.plot(xsam[:,i], 'o', markeredgecolor='w')
        plt.savefig(f"{prefix}_d{i}.png")
        plt.clf()

    for i in range(ndim):
        for j in range(i+1, ndim):
            plt.plot(xsam[:,i], xsam[:,j], 'o', markeredgecolor='w')
            plt.savefig(f"{prefix}_d{i}_d{j}.png")
            plt.clf()

#############################################################

def parallel_coordinates(parnames, values, labels, savefig='pcoord'):
    """Plots parallel coordinates.

    Args:
        parnames (list[str]): list of `d` parameter names
        values (np.ndarray): `(d, N)` array of `N` data points with `d` parameters
        labels (list[str]): list of `N` labels/categories, one per point
        savefig (str, optional): figure name to save
    """

    # Start the figure
    fig=plt.figure(figsize=(14,8))
    fig.add_axes([0.1,0.25,0.8,0.65])
    ax = plt.gca()

    # Categorize
    ulabels = np.unique(labels)
    n_labels = len(ulabels)


    # Plot
    class_id = np.searchsorted(ulabels, labels)
    lines = plt.plot(values[:,:], 'go-',ms=6,linewidth=0.1)

    if n_labels>1:
        # Set colors
        cmap = plt.get_cmap('coolwarm')
        colors = np.array([cmap(j) for j in np.arange(n_labels)/(n_labels-1)])
        for c,l in zip(class_id, lines):
            l.set_color(colors[c])


    # Gridification
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 5))
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('none')

    plt.xticks(np.arange(len(parnames)), parnames, rotation=60, fontsize=11)
    plt.grid(axis='x', ls='-')

    leg_handlers = [ lines[np.where(class_id==id)[0][0]]
                    for id in range(n_labels)]
    ax.legend(leg_handlers, ulabels, frameon=False, loc='upper left',
                    ncol=len(labels),
                    bbox_to_anchor=(0, 1.15, 1, 0))

    # Show or save
    plt.savefig(savefig)
    plt.clf()

#############################################################

def plot_yx(x, y, rowcols=None, ylabel='', xlabels=None,
            log=False, filename='eda.png',
            xpad=0.3, ypad=0.3, gridshow=True, ms=2, labelsize=18):
    """Plots an output vs one input at a time in a matrix of figures.

    Args:
        x (np.ndarray): An `(N,d)` input array.
        y (np.ndarray): An `(N,)` output array.
        rowcols (tuple[int], optional): A pair of integers, rows and columns.
        ylabel (str, optional): Y-axis label.
        xlabels (list[str], optional): List of X-axis labels.
        log (bool, optional): Whether to have log-`y` values or not.
        filename (str, optional): Figure filename to save.
        xpad (float, optional): Padding size between columns.
        ypad (float, optional): Padding size between rows.
        gridshow (bool, optional): Whether to show grid or not.
        ms (int, optional): Marker size.
        labelsize (int, optional): Axes label sizes.
    """
    nsam, ndim = x.shape
    assert(nsam==y.shape[0])

    if rowcols is None:
        rows = 3
        cols = (ndim // 3) + 1
    else:
        rows, cols = rowcols

    fig, axes = plt.subplots(rows, cols, figsize=(8*cols,(4+ypad)*rows),
                             gridspec_kw={'hspace': ypad, 'wspace': xpad})

    if rows * cols > 1:
        axes=axes.reshape(rows, cols)
        axes = axes.T
    else:
        axes = np.array([[axes]])

    if xlabels is None:
        xlabels = ['' for i in range(ndim)]
    #print(axes.shape)
    for i in range(ndim):
        ih = i % cols
        iv = i // cols
        axes[ih, iv].plot(x[:, i], y, 'o', ms=ms, markeredgecolor='w')
        axes[ih, iv].set_xlabel(xlabels[i], size=labelsize)
        axes[ih, iv].set_ylabel(ylabel, size=labelsize)
        #axes[ih, iv].set_ylim(ymin=-0.05, ymax=0.5)
        axes[ih, iv].grid(gridshow)
        if log:
            axes[ih, iv].set_yscale('log')

    for i in range(ndim, cols*rows):
        ih = i % cols
        iv = i // cols
        axes[ih, iv].remove()

    plt.savefig(filename)


#############################################################

def plot_sens(sensdata, pars, cases,
              vis="bar", reverse=False, topsens=None,
              par_labels=None, case_labels=None, colors=None,
              xlbl='', title='', grid_show=True,
              legend_show=2, legend_size=10, ncol=4,
              lbl_size=22, yoffset=0.1,
              xdatatick=None, xticklabel_size=None, xticklabel_rotation=0,
              figname='sens.png'):
    """Plots sensitivities for multiple observables with respect to multiple parameters.

    Args:
        sensdata (np.ndarray): An array of sensitivities of size `(o,d)`.
        pars (list[int]): List of parameter indices to use. All should be less than `d`.
        cases (list[int]): List of output indices to use. All should be less than `o`.
        vis (str, optional): Plot type 'bar' or 'graph' (the latter is not used/tested often).
        reverse (bool, optional): Whether to flip the input data (i.e. parameters and outputs).
        topsens (int, optional): Show only some number of top parameters (default: show all).
        par_labels (list[str], optional): Parameter labels.
        case_labels (list[str], optional): Output labels.
        colors (None, optional): Parameter bar colors.
        xlbl (str, optional): X-label.
        title (str, optional): Title.
        grid_show (bool, optional): Whether to show the grid.
        legend_show (int, optional): Type of legend location, `1` is inside, `2` is below, `3` is above the graph.
        legend_size (int, optional): Legend fontsize.
        ncol (int, optional): Number of columns in legend.
        lbl_size (int, optional): Axes label size.
        yoffset (float, optional): Vertical offset, white space below the figure.
        xdatatick (list[float], optional): X-tick locations. By default, `1, ..., o`.
        xticklabel_size (int, optional): X-tick label size.
        xticklabel_rotation (int, optional): X-tick label rotation angle.
        figname (str, optional): Figure file name.
    """

    ncases=sensdata.shape[0]
    npar=sensdata.shape[1]

    wd=0.6
    ylbl='Sensitivity'

    assert set(pars) <= set(range(npar))
    assert set(cases) <= set(range(ncases))

    # Set up the figure
    if xticklabel_size is None:
        xticklabel_size=min(400//ncases, 20)
    fig = plt.figure(figsize=(20,12))
    fig.add_axes([0.1,0.2+yoffset,0.8,0.6-yoffset])

    # Default parameter names
    if (par_labels is None):
        par_labels = ['par_'+str(i+1) for i in range(npar)]

    # Default case names
    if (case_labels is None):
        case_labels = ['case_'+str(i+1) for i in range(ncases)]

    if(reverse):
        par_labels, case_labels = case_labels, par_labels
        pars, cases = cases, pars
        sensdata=sensdata.T

    npar_=len(pars)
    ncases_=len(cases)

    sensind = np.argsort(np.average(sensdata, axis=0))[::-1]

    if topsens is None:
        topsens=npar_

    # Create colors list
    if colors is None:
        colors_ = set_colors(topsens)
        colors_.extend(set_colors(npar_-topsens))
        colors = [0.0 for i in range(npar_)]
        for i in range(npar_):
            colors[sensind[i]]=colors_[i]

    case_labels_=[]
    for i in range(ncases_):
        case_labels_.append(case_labels[cases[i]])

    if xdatatick is None:
        xflag=False
        xdatatick=np.array(range(1,ncases_+1))
        sc=1.
    else:
        xflag=True
        sc=float(xdatatick[-1]-xdatatick[0])/ncases_


    if (vis=="graph"):
        for i in range(npar_):
            plt.plot(xdatatick,sensdata[cases,i], '-o',color=colors[pars[i]], label=par_labels[pars[i]])
    elif (vis=="bar"):
        curr=np.zeros((ncases_))
        #print pars,colors
        for i in range(npar_):
            plt.bar(xdatatick, sensdata[cases,i], width=wd*sc,color=colors[pars[i]], bottom=curr, label=par_labels[pars[i]])
            curr=sensdata[cases,i]+curr

        if not xflag:
            plt.xticks(np.array(range(1,ncases_+1)),case_labels_,rotation=xticklabel_rotation, fontsize=xticklabel_size)

        plt.xlim(xdatatick[0]-wd*sc/2.-0.1,xdatatick[-1]+wd*sc/2.+0.1)

    plt.ylabel(ylbl,fontsize=lbl_size)
    plt.xlabel(xlbl,fontsize=lbl_size)
    plt.title(title)

    maxsens=max(max(curr),1.0)
    plt.ylim([0,maxsens])
    handles,labels = plt.gca().get_legend_handles_labels()
    handles = [ handles[i] for i in sensind[:topsens]]
    labels = [ labels[i] for i in sensind[:topsens]]
    if legend_show==1:
        plt.legend(handles,labels,fontsize=legend_size)
    elif (legend_show==2):
        plt.legend(handles,labels,loc='upper left',bbox_to_anchor=(0.0, -0.15),fancybox=True, shadow=True,ncol=ncol,labelspacing=-0.1,fontsize=legend_size)
    elif (legend_show==3):
        plt.legend(handles,labels,loc='upper left', bbox_to_anchor=(0.0, 1.2),fancybox=True, shadow=True,ncol=ncol,labelspacing=-0.1,fontsize=legend_size)


    # if not xflag:
    #     print(dir(plt.gca().xaxis.get_major_ticks()[0]))
    #     zed = [tick.label.set_fontsize(xticklabel_size) for tick in plt.gca().xaxis.get_major_ticks()]

    plt.grid(grid_show)
    plt.savefig(figname)
    plt.clf()


#############################################################

def plot_jsens(msens,jsens,varname='', inpar_names=None,figname='senscirc.png'):
    """Plotting circular joint sensitivities.

    Args:
        msens (np.ndarray): Main sensitivities, a 1d array.
        jsens (np.ndarray): Joint sensitivities. A 2d square array.
        varname (str, optional): Variable name.
        inpar_names (list, optional): List of names for input parameters. Defaults to something generic.
        figname (str, optional): Saving figure file name.
    """
    Nmain=min(len(np.nonzero(msens)[0]),6)
    Nsec=Nmain-1
    lwMax=10
    lwCut=0.2
    radMain=50
    radOut=15
    lext=0.4
    verbose=1

    nx,ny=jsens.shape
    # assert nx=ny? compare with msens shape?

    if inpar_names is None:
        inpar_names = [f'Par{j}' for j in range(nx)]


    #jsens=np.log10(jsens);
    #print msens
    ind=msens.argsort()[::-1];
    msensShort=msens[ind[0:Nmain]]
    if verbose > 0:
        for i in range(Nmain):
            print("Input %d, main sensitivity %lg" % (ind[i],msens[ind[i]]))
    fig = plt.figure(figsize=(10,8))
    ax=fig.add_axes([0.05, 0.05, 0.9, 0.9],aspect='equal')
    #circ=pylab.Circle((0,0),radius=0.5,color='r')
    circ=mpl.patches.Wedge((0.0,0.0),1.01, 0, 360, width=0.02,color='r')
    ax.add_patch(circ)
    maxJfr=-1.e10;
    for i in range(Nmain):
        jfr_i=np.array(np.zeros(nx))
        iord=ind[i]
        for j in range(iord):
            jfr_i[j]=jsens[j,iord]
        for j in range(iord+1,nx):
            jfr_i[j]=jsens[iord,j]
        ind_j=jfr_i.argsort()[::-1];
        if jfr_i[ind_j[0]] > maxJfr: maxJfr = jfr_i[ind_j[0]];
        if verbose > 1:
            for j in range(Nsec):
                print("%d  %d %d" % (iord,ind_j[j],jfr_i[ind_j[j]]))
    if verbose > 1:
        print("Maximum joint sensitivity %lg:" % maxJfr)
    gopar=[]
    for i in range(Nmain):
        jfr_i=np.array(np.zeros(nx))
        iord=ind[i]
        for j in range(iord):
            jfr_i[j]=jsens[j,iord]
        for j in range(iord+1,nx):
            jfr_i[j]=jsens[iord,j]
        ind_j=jfr_i.argsort()[::-1];
        elst=[]
        for j in range(Nsec):
            if maxJfr>1.e-16 and jfr_i[ind_j[j]]/maxJfr >= lwCut:
                posj=[k for k,x in enumerate(ind[:Nmain]) if x == ind_j[j]]
                if verbose > 2:
                    print(j,posj)
                if len(posj) > 0 :
                    x1=np.cos(0.5*np.pi+(2.0*np.pi*posj[0])/Nmain)
                    x2=np.cos(0.5*np.pi+(2.0*np.pi*i      )/Nmain)
                    y1=np.sin(0.5*np.pi+(2.0*np.pi*posj[0])/Nmain)
                    y2=np.sin(0.5*np.pi+(2.0*np.pi*i      )/Nmain)
                    lw=lwMax*jfr_i[ind_j[j]]/maxJfr
                    plt.plot([x1,x2],[y1,y2],'g-',linewidth=lw)
                    if ( verbose > 2 ):
                        print(iord,ind[posj[0]])
                else:
                    elst.append(j)
        if len(elst) > 0:
            asft=[0,-1,1]
            for k in range(min(len(elst),3)):
                ang=0.5*np.pi+(2.0*np.pi*i)/Nmain+2*np.pi/12*asft[k]
                x2=np.cos(0.5*np.pi+(2.0*np.pi*i)/Nmain)
                y2=np.sin(0.5*np.pi+(2.0*np.pi*i)/Nmain)
                x1=x2+lext*np.cos(ang)
                y1=y2+lext*np.sin(ang)
                lw=lwMax*jfr_i[ind_j[elst[k]]]/maxJfr
                plt.plot([x1,x2],[y1,y2],'g-',linewidth=lw)
                plt.plot([x1],[y1],"wo",markersize=radOut,markeredgecolor='k',
                         markeredgewidth=2)
                if ( ind_j[elst[k]] > 32 ):
                    ltext=str(ind_j[elst[k]]+3)
                elif ( ind_j[elst[k]] > 30 ):
                    ltext=str(ind_j[elst[k]]+2)
                else:
                    ltext=str(ind_j[elst[k]]+1)
                plt.text(x1+(0.15)*np.cos(ang),y1+(0.15)*np.sin(ang),ltext,
                            ha='center',va='center',fontsize=16)
                posj=[k1 for k1,x in enumerate(gopar) if x == ind_j[elst[k]]]
                if len(posj)==0:
                    gopar.append(ind_j[elst[k]])
        if ( verbose > 2 ):
            print("------------------------")
    for i in range(Nmain):
        angl=0.5*np.pi+(2.0*np.pi*i)/Nmain
        xc=np.cos(angl);
        yc=np.sin(angl);
        msize=radMain*msens[ind[i]]/msens[ind[0]]
        plt.plot([xc],[yc],"bo",markersize=msize,markeredgecolor='k',markeredgewidth=2)
        da=1.0
        lab=0.2
        llab=lab*msens[ind[i]]/msens[ind[0]]

        ltext=str(ind[i]+1)
        lleg=ltext+" - "+inpar_names[ind[i]]
        plt.text(xc+(0.08+llab)*np.cos(angl+da),yc+(0.08+llab)*np.sin(angl+da),ltext,
                 ha='center',va='center',fontsize=16)
        plt.text(1.6,1.2-0.15*i,lleg,fontsize=16)
    for k in range(len(gopar)):
        lleg=str(gopar[k]+1)+" - "+inpar_names[gopar[k]]
        plt.text(1.6,1.2-0.15*Nmain-0.15*k,lleg,fontsize=16)

    plt.text(0.9,-1.2,varname,fontsize=27)

    ax.set_xlim([-1-1.6*lext,1.8+1.6*lext])
    ax.set_ylim([-1-1.6*lext,1+1.6*lext])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(figname)
    plt.clf()

#############################################################

def plot_tri(xi, yy=None, names=None, msize=3, figname='xsam_tri.png'):
    """Plots multidimensional samples in a triangular pairwise way.

    Args:
        xi (np.ndarray): `(N,d)` array to plot.
        yy (None, optional): `(N,)` array. Color code the dots with y values. Defaults to None, i.e. no color coding.
        names (list[str], optional): List of `d` names.
        msize (int, optional): Markersize of the 2d plots.
        figname (str, optional): Figure file name.
    """
    nsam, npar = xi.shape
    figs, axarr = plt.subplots(npar, npar, figsize=(15, 15))
    if npar==1: axarr=[[axarr]]

    if names is None:
        names = ['p'+str(j) for j in range(npar)]
    assert(len(names)==npar)

    for i in range(npar):
        thisax = axarr[i][i]
        thisax.plot(np.arange(nsam), xi[:, i], linewidth=1)

        if i == 0:
            thisax.set_ylabel(names[i])
        if i == npar - 1:
            thisax.set_xlabel(names[i])
        if i > 0:
            thisax.yaxis.set_ticks_position("right")
        # thisax.yaxis.set_label_coords(-0.12, 0.5)


        for j in range(i):
            thisax = axarr[i][j]
            axarr[j][i].axis('off')

            if yy is not None:
                thisax.scatter(xi[:, j], xi[:, i],c=yy, s=msize, alpha=0.8)
            else:
                thisax.plot(xi[:, j], xi[:, i], 'o', markersize=msize)

            # x0, x1 = thisax.get_xlim()
            # y0, y1 = thisax.get_ylim()
            # #thisax.set_aspect((x1 - x0) / (y1 - y0))

            if j == 0:
                thisax.set_ylabel(names[i])
            if i == npar - 1:
                thisax.set_xlabel(names[j])
            if j > 0:
                thisax.yaxis.set_ticklabels([])

    plt.savefig(figname)

#############################################################

def plot_pdf1d(sams, pltype='hist', color='b',
               lw=1.0, nom_height_factor=10., histalpha=1.0, label='', ax=None):
    """Plotting 1d PDFs of samples.

    Args:
        sams (np.ndarray): The `(N,)` samples of interest.
        pltype (str, optional): Plot type. Options are 'kde' (Kernel Density Estimation), 'hist' (Histogram), 'sam' (plot samples as dots on x-axis), 'nom' (plot a nominal vertical line)
        color (str, optional): Color.
        lw (float, optional): Line width, when relevant.
        nom_height_factor (float, optional): Controls the height of the nominal vertical bar.
        histalpha (float, optional): Opacity of histogram, between `0` and `1`.
        label (str, optional): Label for legend.
        ax (plt.Axes, optional): Axis handle. If None, use the current axis.
    Note:
        ax is changed as a result of this function. Further beautification and figure saving should be done outside this function.
    """
    if ax is None:
        ax = plt.gca()

    if pltype == 'kde':
        ngrid = 111
        a = 3.
        pnom = np.mean(sams)
        pdel = np.std(sams)
        pgrid = np.linspace(pnom - a * pdel, pnom + a * pdel, ngrid)
        if (np.var(sams, axis=0)>0.0):
            pdf = get_pdf(sams, pgrid)
            ax.plot(pgrid, pdf, color=color, linewidth=lw, label=label)
        else:
            y1, y2 = ax.get_ylim()
            ax.plot([sams[0], sams[0]], [0, y2], color, linewidth=lw, label=label)
    elif pltype == 'hist':
        n, bins, patches = ax.hist(sams, bins=20,
                                   density=True,
                                   facecolor=color, alpha=histalpha)
    elif pltype == 'sam':
        ax.plot(sams, np.zeros_like(sams), 'ro', ms=3, markerfacecolor='None', label=label)
    elif pltype == 'nom':
        y1, y2 = ax.get_ylim()
        y2f = nom_height_factor
        for sam in sams:
            ax.plot([sam, sam], [y1, y2 / y2f], '--', color=color, linewidth=lw, label=label)

    else:
        print("Plot type is not recognized. Exiting")
        sys.exit()

    #ax.set_ylim(bottom=0)
    ax.grid(False)


#############################################################

def plot_pdf2d(samsx, samsy, pltype='kde', ncont=10,
               color=None, lwidth=1.0, mstyle='o', ax=None):
    """Plot 2d contour plot of a PDF given two sets of samples.

    Args:
        samsx (np.ndarray): First `(N,)` samples of interest.
        samsy (np.ndarray): Second `(N,)` samples of interest.
        pltype (str, optional): Plot type. Options are Options are 'kde' (Kernel Density Estimation), 'sam' (plot samples only).
        ncont (int, optional): Number of contours.
        color (str, optional): Color. If None, uses the multicolor default of matplotlib.
        lwidth (float, optional): Line width.
        mstyle (str, optional): Marker stile.
        ax (plt.Axes, optional): Axis handle. If None, use the current axis.
    Note:
        ax is changed as a result of this function. Further beautification and figure saving should be done outside this function.
    """
    if ax is None:
        ax = plt.gca()

    if pltype == 'kde':
        ngrid = 100
        a = 3.
        pnomx = np.mean(samsx)
        pdelx = np.std(samsx)
        pnomy = np.mean(samsy)
        pdely = np.std(samsy)

        x = np.linspace(pnomx - a * pdelx, pnomx + a * pdelx, ngrid)
        y = np.linspace(pnomy - a * pdely, pnomy + a * pdely, ngrid)
        X, Y = np.meshgrid(x, y)
        pgrid = np.vstack((X.flatten(), Y.flatten())).T  # pgrid.shape is (33^2,2)

        if (np.var(samsx, axis=0)>0.0 and np.var(samsy, axis=0)>0.0):
            pdf = get_pdf(np.vstack((samsx, samsy)).T, pgrid)

            if color is None:
                ax.contour(X, Y, pdf.reshape(X.shape), ncont, linewidths=lwidth)
            else:
                ax.contour(X, Y, pdf.reshape(X.shape), ncont, colors=color, linewidths=lwidth)
        else:
            print("Skipping 2d PDF due to degeneracy")

    elif pltype == 'sam':
        ax.plot(samsx, samsy, color=color, marker=mstyle, linestyle='None')
    else:
        print("Plot type is not recognized. Exiting")
        sys.exit()

    ax.grid(False)


#############################################################

def plot_pdfs(ind_show=None, plot_type='tri', pdf_type='hist',
              samples_='chain.txt', burnin=100, every=1,
              names_=None, nominal_=None, prange_=None,
              show_2dsamples=False,
              lsize=13, zsize=13, xpad=0.3, ypad=0.3):
    """Computing and plotting set of 1d and/or 2d PDFs given a sample set.

    Args:
        ind_show (list[int], optional): Indices of dimensions (columns of samples) to show.
        plot_type (str, optional): Plot type. Options are 'tri' (trianguar plot of 1d/2d marginals), 'inds' (1d marginals in a single figure), 'ind' (individual files for 1d and 2d marginal PDFs)
        pdf_type (str, optional): 1d PDF type. Options are 'hist' (histogram) or 'kde' (Kernel Density Estimation)
        samples_ (str or np.ndarray, optional): Filename or `(N,d)` numpy array of samples.
        burnin (int, optional): Number of samples to throw away from the beginning of samples.
        every (int, optional): Stratification: use every `k`-th sample.
        names_ (str or list[str], optional): Filename containing the names of dimensions or list of names. Default is None, that leads to generic names.
        nominal_ (str or np.ndarray, optional): Filename or `(d,)` numpy array of nominals to be shown as vertical bars on top of 1d PDFs, and dots on top of 2d PDFs.
        prange_ (str or np.ndarray, optional): Filename or `(d,2)` numpy array of dimensional ranges to be shown as 'box' priors.
        show_2dsamples (bool, optional): Whether or not to display the samples on top of 2d contour plots.
        lsize (int or float, optional): Title size and X- and Y- label size.
        zsize (int or float, optional): X- and Y- tick label size.
        xpad (float, optional): Horizontal padding for multiplot figures ('tri' and 'inds').
        ypad (float, optional): Vertical padding for multiplot figures ('tri' and 'inds').

    Returns:
        tuple: Figure and axes array handles for further edits if needed.
    """
    if isinstance(samples_, str):
        samples_all = np.loadtxt(samples_, ndmin=2)
    else:
        samples_all = samples_.copy()

    npar_all = samples_all.shape[1]

    if ind_show is None:
        ind_show = range(npar_all)

    samples = samples_all[burnin::every, ind_show]
    npar = len(ind_show)

    if isinstance(names_, str):
        if os.path.exists(names_):
            with open(names_) as f:
                names = f.read().splitlines()
                assert(len(names) == npar_all)
        else:
            print(f"File {names_} does not exist. Exiting")
            sys.exit()
    elif names_ is None:
        names = ['p' + str(i) for i in range(npar_all)]
    else:
        names = names_.copy()

    if nominal_ is not None:
        if isinstance(nominal_, str):
            if os.path.exists(nominal_):
                nominals = np.loadtxt(nominal_, ndmin=1)
            else:
                print(f"File {nominal_} does not exist. Exiting")
                sys.exit()
        else:
            nominals = nominal_.copy()

    show_range = False
    if prange_ is not None:
        show_range = True
        if isinstance(prange_, str):
            if os.path.exists(prange_):
                prange = np.loadtxt(prange_, ndmin=2)
            else:
                print(f"File {prange_} does not exist. Exiting")
                sys.exit()
        else:
            prange = prange_.copy()

    if plot_type == 'tri':
        figs, axarr = plt.subplots(npar, npar,
                                   figsize=(6*npar,(5+ypad)*npar),
                                   gridspec_kw={'hspace': ypad, 'wspace': xpad})
        if npar==1: axarr=[[axarr]]

    elif plot_type == 'inds':
        ncol = min(npar, 4)
        nrow = (npar-1)//ncol+1
        figs, axarr = plt.subplots(nrow, ncol,
                                   figsize=(6*ncol,(5+ypad)*nrow),
                                   gridspec_kw={'hspace': ypad, 'wspace': xpad})
        if nrow==1: axarr=[axarr]
        if ncol==1: axarr=[axarr]
        for i in range(npar, ncol*nrow):
            k = i % ncol
            j = (i-k)//ncol
            axarr[j][k].axis('off')
    elif plot_type == 'ind':
        axarr = []
        figs = []
    else:
        print(f"Plot type {plot_type} is not recognized. Exiting.")
        sys.exit()

    for i in range(npar):
        print(f"Plotting PDFs for {names[ind_show[i]]}")
        if plot_type == 'tri':
            thisax = axarr[i][i]
        elif plot_type == 'inds':
            k = i % ncol
            j = (i-k)//ncol
            thisax = axarr[j][k]


        elif plot_type == 'ind':
            fig = plt.figure(figsize=(8, 8))
            thisax = plt.gca()
            axarr.append(thisax)
            figs.append(fig)

        plot_pdf1d(samples[:, i], pltype=pdf_type, ax=thisax)
        thisax.set_ylim(bottom=0)

        if nominal_ is not None:
            plot_pdf1d(np.array(nominals[ind_show[i]], ndmin=2), pltype='nom',
                       nom_height_factor=1., color='r', ax=thisax)
        if show_range:
            dr = np.abs(prange[ind_show[i], 1] - prange[ind_show[i], 0])
            thisax.plot([prange[ind_show[i], 0], prange[ind_show[i], 0],
                         prange[ind_show[i], 1], prange[ind_show[i], 1]],
                        [0.0, 1.0 / dr, 1.0 / dr, 0.0], 'g--', linewidth=0.6, label='Prior', zorder=1000)

        x0, x1 = thisax.get_xlim()
        y0, y1 = thisax.get_ylim()
        # thisax.set_aspect((x1 - x0) / (y1 - y0))
        thisax.set_title('PDF of ' + names[ind_show[i]], fontsize=lsize)

        if plot_type == 'tri':
            if i == 0:
                thisax.set_ylabel(names[ind_show[i]], size=lsize)
            if i == npar - 1:
                thisax.set_xlabel(names[ind_show[i]], size=lsize)
                thisax.tick_params(axis='x', which='major', labelsize=zsize)
            if i > 0:
                thisax.yaxis.set_ticks_position("right")

            thisax.tick_params(axis='y', which='major', labelsize=zsize)
            # thisax.yaxis.set_label_coords(-0.12, 0.5)
        elif plot_type == 'inds':
            thisax.set_xlabel(names[ind_show[i]], size=lsize)
            thisax.tick_params(axis='both', which='major', labelsize=zsize)
        elif plot_type == 'ind':
            thisax.set_xlabel(names[ind_show[i]], size=lsize)
            thisax.tick_params(axis='both', which='major', labelsize=zsize)
            plt.savefig('pdf_' + names[ind_show[i]] + '.png')
            # plt.clf()

        # Plot 2d contours
        for j in range(i):
            if plot_type == 'tri':
                thisax = axarr[i][j]
                axarr[j][i].axis('off')
            elif plot_type == 'ind':
                plt.figure(figsize=(8, 8))
                thisax = plt.gca()
            elif plot_type == 'inds':
                break

            plot_pdf2d(samples[:, j], samples[:, i],
                       pltype='kde', color='b', ncont=15, lwidth=1, ax=thisax)
            if show_2dsamples:
                thisax.plot(samples[:, j], samples[:, i], 'ko', markeredgecolor='white', markersize=5, zorder=-1000)

            if nominal_ is not None:
                plot_pdf2d(nominals[ind_show[j]], nominals[ind_show[i]],
                           pltype='sam', color='r', mstyle='x', ax=thisax)

            x0, x1 = thisax.get_xlim()
            y0, y1 = thisax.get_ylim()
            #thisax.set_aspect((x1 - x0) / (y1 - y0))

            if plot_type == 'tri':
                if j == 0:
                    thisax.set_ylabel(names[ind_show[i]], size=lsize)
                    thisax.tick_params(axis='y', which='major', labelsize=zsize)
                if i == npar - 1:
                    thisax.set_xlabel(names[ind_show[j]], size=lsize)
                    thisax.tick_params(axis='x', which='major', labelsize=zsize)
            elif plot_type == 'ind':
                thisax.set_ylabel(names[ind_show[i]], size=lsize)
                thisax.set_xlabel(names[ind_show[j]], size=lsize)
                plt.savefig('pdf_' + names[ind_show[j]] + '_' + names[ind_show[i]] + '.png')
                # plt.clf()


    if plot_type == 'tri':
        plt.savefig('pdf_tri.png')
    elif plot_type == 'inds':
        plt.savefig('pdf_inds.png')

    return figs, axarr

#############################################################

def plot_ens(xdata, ydata,
             color='b', lw=2.0, ms=1,
             grid_show=True, label='', mec='k',
             connected=True, interp=True,
             offset=(None, None), ax=None):
    """Plotting an ensemble of `y` values versus input `x`.

    Args:
        xdata (np.ndarray): Input values, an 1d array of size `(N,)`.
        ydata (np.ndarray): Output values, a 2d array of size `(N,M)`.
        color (str, optional): Plot color.
        lw (int or float, optional): Linewidth.
        ms (int or float, optional): Markersize.
        grid_show (bool, optional): Whether to show the grid or not.
        label (str, optional): Label for legends down the road.
        mec (str, optional): Marker edge color
        connected (bool, optional): Whether to connect the data dots or not.
        interp (bool, optional): Whether to have smooth interpolation or not.
        offset (tuple, optional): Tuple of (shift, scale) to preprocess y-data, if needed, both shift and scale are either None or 1d arrays of size `(d,)`.
        ax (plt.Axes, optional): Axis handle. If None, use the current axis.
    Note:
        ax is changed as a result of this function. Further beautification and figure saving should be done outside this function.
    """
    if ax is None:
        ax = plt.gca()

    nx = ydata.shape[0]
    nsam = ydata.shape[1]

    shift, scale = offset
    if shift is None:
        shift = np.zeros_like(ydata)
    else:
        shift = shift.reshape((nx, 1))

    if scale is None:
        scale = np.ones_like(ydata)
    else:
        scale = scale.reshape((nx, 1))


    ydata_ = (ydata - shift) / scale

    if interp is not None:
        connected = True
        xmin = xdata.min()
        xmax = xdata.max()
        xdel = xmax - xmin
        ngr = 100
        xdata_ = np.linspace(xmin - 0.0 * xdel, xmax + 0.0 * xdel, ngr)

        tmp = np.empty((ngr, nsam))
        for i in range(nsam):
            interp_fcn = interp1d(xdata, ydata_[:, i], kind=interp)
            tmp[:, i] = interp_fcn(xdata_)
        ydata_ = tmp.copy()

    else:
        xdata_ = xdata.copy()

    if connected:
        for i in range(nsam):
            ax.plot(xdata_, ydata_[:, i],
                       color=color, linewidth=lw)
    else:
        for i in range(nsam):
            ax.plot(xdata_, ydata_[:, i], color=color,
                       linestyle='None',
                       marker='o', markeredgecolor=mec, markersize=ms,
                       label=label, zorder=1000)

    ax.grid(grid_show)

#############################################################

def plot_vars(xdata, ydata, variances=None, ysam=None, stdfactor=1.,
              varlabels=None, varcolors=None, grid_show=True,
              connected=True, interp=None, offset=(None, None), ax=None):
    """Plotting mean predictions with multicolor variances as error bars.

    Args:
        xdata (np.ndarray): Input values, an 1d array of size `(N,)`.
        ydata (np.ndarray): Output values, a 1d array of size `(N,)`.
        variances (None, optional): Variances, a 2d array of size `(N,K)`.
        ysam (None, optional): True samples, a 2d array of size `(N,M)`.
        stdfactor (float, optional): Factor in front of st.dev. for plotting (e.g. `1.0` or `3.0`).
        varlabels (list[str], optional): List of `K` labels for each variance. If None, the code comes up with generic names.
        varcolors (list[str], optional): List of `K` variance colors. If None, the code uses shades of grey (not fifty).
        grid_show (bool, optional): Whether or not to show the grid.
        connected (bool, optional): Whether to connect the data dots or not.
        interp (bool, optional): Whether to have smooth interpolation or not.
        offset (tuple, optional): Tuple of (shift, scale) to preprocess y-data, if needed, both shift and scale are either None or 1d arrays of size `(d,)`.
        ax (plt.Axes, optional): Axis handle. If None, use the current axis.
    Note:
        ax is changed as a result of this function. Further beautification and figure saving should be done outside this function.
    """
    if ax is None:
        ax = plt.gca()

    if varcolors is None:
        cmap = mpl.cm.Greys

    if ysam is not None:
        plot_ens(xdata, ysam,
                 connected=connected, interp=interp,
                 color='r', lw=0.3, offset=offset, ax=ax)

    shift, scale = offset
    if shift is None:
        shift = np.zeros_like(ydata)
    if scale is None:
        scale = np.ones_like(ydata)

    ydata_ = (ydata - shift) / scale


    if variances is None:
        variances = np.zeros((xdata.shape[1], 1))
    if len(variances.shape)==1:
        variances = variances.reshape(-1, 1)

    variances_ = variances / (scale.reshape(-1, 1) * scale.reshape(-1, 1))
    nvariances = variances_.shape[1]

    if interp is not None:
        connected = True
        xmin = xdata.min()
        xmax = xdata.max()
        xdel = xmax - xmin
        ngr = 100
        xdata_ = np.linspace(xmin - 0.0 * xdel, xmax + 0.0 * xdel, ngr)

        interp_fcn = interp1d(xdata, ydata_, kind=interp)
        ydata_ = interp_fcn(xdata_)

        tmp = np.zeros((ngr, nvariances))
        for i in range(nvariances):
            interp_fcn = interp1d(xdata, variances_[:, i], kind=interp)
            tmp[:, i] = interp_fcn(xdata_)
        variances_ = tmp.copy()

    else:
        xdata_ = xdata.copy()

    cvars = np.cumsum(variances_, axis=1)
    normalize = mpl.colors.Normalize(vmin=0.0, vmax=0.5)
    if varlabels is None:
        varlabels = ['Var ' + str(i) for i in range(1, nvariances + 1)]

    if connected:
        ax.plot(xdata_, ydata_, color='orange',
                   marker='None', linestyle='-',
                   label='Mean prediction', zorder=10000)
        for ii in range(nvariances):
            if varcolors is None:
                varcolor = cmap(normalize(0.1 + ii * 0.3 / nvariances))
            else:
                varcolor = varcolors[ii]
            ax.fill_between(xdata_,
                               ydata_ - stdfactor * np.sqrt(cvars[:, ii]),
                               ydata_ + stdfactor * np.sqrt(cvars[:, ii]),
                               color=varcolor,
                               label=varlabels[ii], zorder=1000 - ii)
    else:
        ax.plot(xdata_, ydata_, color='orange',
                   marker='o', linestyle='None',
                   label='Mean prediction', zorder=100000)
        for ii in range(nvariances):
            if varcolors is None:
                varcolor = cmap(normalize(0.1 + ii * 0.3 / nvariances))
            else:
                varcolor = varcolors[ii]
            ax.errorbar(xdata_, ydata_,
                           yerr=stdfactor * np.sqrt(cvars[:, ii]),
                           color=varcolor,
                           fmt='o', elinewidth=13,
                           label=varlabels[ii], zorder=10000 - ii)

    ax.grid(grid_show)


#############################################################

def plot_shade(xdata, ydata, nq=51, cmap=mpl.cm.BuGn,
               bounds_show=False, grid_show=True, ax=None):
    """Plotting quantile-shaded predictions given dataset.

    Args:
        xdata (np.ndarray): Input values, an 1d array of size `(N,)`.
        ydata (np.ndarray): Output values, a 2d array of size `(N,M)`.
        nq (int, optional): Number of quantiles.
        cmap (mpl.Cm, optional): Colormap. Defaults to BuGn.
        bounds_show (bool, optional): Whether to highlight the bounds.
        grid_show (bool, optional): Whether to show the grid or not.
        ax (plt.Axes, optional): Axis handle. If None, use the current axis.
    Note:
        ax is changed as a result of this function. Further beautification and figure saving should be done outside this function.
    """
    if ax is None:
        ax = plt.gca()

    nx = xdata.shape[0]
    assert(nx == ydata.shape[0])

    mq = ss.mstats.mquantiles(ydata, prob=[float(i + 1) / float(nq)
                                 for i in range(nq - 1)], axis=1)

    normalize = mpl.colors.Normalize(vmin=0.01, vmax=0.5)

    for k in range(int(nq / 2)):
        ax.fill_between(xdata, mq[:, k], mq[:, k + 1],
                         color=cmap(normalize(0.01 + k * .02)))
    for k in range(int(nq / 2), nq - 2):
        ax.fill_between(xdata, mq[:, k], mq[:, k + 1],
                         color=cmap(normalize(0.5 - (k - nq / 2) * 0.02)))
    if bounds_show:
        ax.plot(xdata, mq[:, 0], linewidth=2, color="grey")
        ax.plot(xdata, mq[:, -1], linewidth=2, color="grey")

    ax.grid(grid_show)


#############################################################

def plot_1d_anchored_single(models, modelpars,
                            anchor1, anchor2=None,
                            pad=0.5, scale=1., ngr=111,
                            modellabels=None, clearax=False,
                            verbose_labels=False,
                            legend_show=True, ax=None, figname=None):
    """Plots 1d slices of a list of models going through one or two anchor points.

    Args:
        models (list[callable]): List of model evaluators.
        modelpars (list[tuple]): List of model parameter tuples, one for each.
        anchor1 (np.ndarray): 1d array of the first anchor point.
        anchor2 (np.ndarray, optional): 1d array of the second anchor point. Defaults to None, which means a randomly selected second anchor a given distance away from the first.
        pad (float, optional): Padding on both sides of the interval, so the slice goes beyond the anchors.
        scale (float, optional): The distance of the second anchor from the first, if randomly selected.
        ngr (int, optional): Number of grid points for plotting.
        modellabels (None, optional): Labels/names of the models for legend.
        clearax (bool, optional): Clear axes ticks and labels for less busy plotting.
        verbose_labels (bool, optional): Optionally, annotates the points showing their coordinates. Makes sense for low-dim cases.
        legend_show (bool, optional): Whether to show the legend or not.
        ax (None, optional): Axis handle. Default to None, i.e. current axis.
        figname (None, optional): Optionally, save to a figure with a given name.
    """
    if modellabels is None:
        modellabels = [f'Model {i+1}' for i in range(len(models))]

    if ax is None:
        ax = plt.gca()
    if anchor2 is None:
        anchor2 = sample_sphere(center=anchor1, rad=scale, nsam=1).reshape(-1,)
        origin, e1 = anchor1, anchor2-anchor1
        ticklabels = [rf"A$-\Delta$", f'A', rf'A$+\Delta$']
    else:
        origin, e1 = (anchor1+anchor2)/2., (anchor2-anchor1)/2.
        ticklabels = [r'A$_1$', '', r'A$_2$']

    if verbose_labels:
        ticklabels2 = [f"\n{strarr(origin-e1)}", f"\n{strarr(origin)}", f"\n{strarr(origin+e1)}"]
        ticklabels = [a+b for a,b in zip(ticklabels, ticklabels2)]

    tgr = np.linspace(-1.-pad, 1.+pad, ngr)

    xgr = np.array([origin + e1*tt for tt in tgr])

    for model, modelpar, modellabel in zip(models, modelpars, modellabels):
        ygr = model(xgr, modelpar)
        ax.plot(tgr, ygr, label=modellabel)


    #plt.annotate('Something', (-1,0))


    if clearax:
        #ax.set_frame_on(False)
        #ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.grid(False)
        # xmin, xmax = ax.get_xaxis().get_view_interval()
        # ymin, ymax = ax.get_yaxis().get_view_interval()
        # ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
        ax.xaxis.set_ticks([0])
        ax.xaxis.set_ticklabels([''])
        ax.yaxis.set_ticks([0])
        ax.yaxis.set_ticklabels([''])
    else:
        ax.xaxis.set_ticks([-1, 0, 1])
        ax.xaxis.set_ticklabels(ticklabels)
    if legend_show:
        ax.legend()

    if figname is not None:
        plt.savefig(figname)

#############################################################

def plot_1d_anchored(models, modelpars,
                     anchor1,
                     pad=0.5, scale=1., ngr=111,
                     modellabels=None, legend_show=False,
                     clearax=False,
                     ncolrow=(3,5)):
    """Plot multiple 1d slices of models all going through a given anchor point.

    Args:
        models (list[callable]): List of model evaluators.
        modelpars (list[tuple]): List of model parameter tuples, one for each.
        anchor1 (np.ndarray): The anchor point, 1d array.
        pad (float, optional): Padding on both sides of the interval, so the slice goes beyond the anchors.
        scale (float, optional): The distance of the second anchor from the first, if randomly selected.
        ngr (int, optional): Number of grid points for plotting.
        modellabels (None, optional): Labels/names of the models for legend.
        legend_show (bool, optional): Whether to show the legend or not.
        clearax (bool, optional): Clear axes ticks and labels for less busy plotting.
        verbose_labels (bool, optional): Optionally, annotates the points showing their coordinates. Makes sense for low-dim cases.
        ncolrow (tuple, optional): Number of columns and rows in a tuple. Defaults to (3, 5).
    """
    ncol, nrow = ncolrow
    ntot = ncol * nrow


    figs, axarr = plt.subplots(nrow, ncol, figsize=(ncol*4, nrow*4))
    if ntot==1: axarr=[[axarr]]

    for i in range(ntot):
        print(f"Plotting slice {i+1} / {ntot}")
        irow, icol = divmod(i, ncol)

        thisax = axarr[irow][icol]

        plot_1d_anchored_single(models, modelpars, anchor1, anchor2=None,
                     pad=pad, ngr=ngr, scale=scale,
                     modellabels=modellabels, clearax=clearax,
                     verbose_labels=False, legend_show=False,
                     ax=thisax, figname=None)

    if legend_show:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.gca().legend(handles, labels, loc='lower left',
                         ncol=6, bbox_to_anchor=[-2, -0.5])
    plt.savefig('fcn_1dslices.png')

#############################################################

def plot_2d_anchored_single(models, modelpars,
                            anchor1, anchor2=None, anchor3=None,
                            squished=True, pad=0.5, scale=1., ngr=111,
                            modellabels=None, colorful=False, clearax=False,
                            legend_show=True, modelcolors=None, ax=None, figname=None):
    """Plots 2d slices of a list of models going through one or two anchor points.

    Args:
        models (list[callable]): List of model evaluators.
        modelpars (list[tuple]): List of model parameter tuples, one for each.
        anchor1 (np.ndarray): 1d array of the first anchor point.
        anchor2 (np.ndarray, optional): 1d array of the second anchor point. Defaults to None, which means a randomly selected second anchor a given distance away from the first.
        anchor3 (np.ndarray, optional): 1d array of the third anchor point. Defaults to None, which means a randomly selected second anchor a given distance away from the first.
        squished (bool, optional): If squished, the bases in the plane are not orthogonal.
        pad (float, optional): Padding on both sides of the domain, so the slice goes beyond the anchors.
        scale (float, optional): The distance of the second anchor from the first, if randomly selected.
        ngr (int, optional): Number of grid points for plotting.
        modellabels (None, optional): Labels/names of the models for legend.
        colorful (bool, optional): Whether printing with colored surface or simply contours.
        clearax (bool, optional): Clear axes ticks and labels for less busy plotting.
        legend_show (bool, optional): Whether to show the legend or not.
        modelcolors (list, optional): List of model colors.
        ax (None, optional): Axis handle. Default to None, i.e. current axis.
        figname (None, optional): Optionally, save to a figure with a given name.
    """

    # TODO: need to make this cyclic and remove the assertion
    if modelcolors is None:
        modelcolors = ['r', 'g', 'b', 'm', 'y', 'k']
        assert(len(models)<=6)

    if modellabels is None:
        modellabels = [f'Model {i+1}' for i in range(len(models))]


    if ax is None:
        ax = plt.gca()
    if anchor2 is None:
        anchor2 = sample_sphere(center=anchor1, rad=scale, nsam=1).reshape(-1,)
    if anchor3 is None:
        anchor3 = sample_sphere(center=anchor1, rad=scale, nsam=1).reshape(-1,)

    if squished:
        origin, e1, e2 = anchor1, anchor2-anchor1, anchor3-anchor1
            # ticklabelsx = [f"A$_x-\Delta_x$", f'A$_x$', f'A$_x+\Delta_x$']
            # ticklabelsy = [f"A$_y-\Delta_y$", f'A$_y$', f'A$_y+\Delta_y$']
    else:
        print(anchor1.shape, anchor2.shape, anchor3.shape)
        origin, e1, e2 = pick_basis(anchor1, anchor2, anchor3)


    ticklabelsx = []
    ticklabelsy = []


    tgr = np.linspace(-1.-pad, 1.+pad, ngr)

    tx, ty = np.meshgrid(tgr, tgr)
    tgr2 = np.vstack((tx.flatten(), ty.flatten())).T #np.dstack((xp, yp))
    xgr = np.array([origin + e1*tt[0] + e2*tt[1] for tt in tgr2])

    j=0
    for model, modelpar, modellabel, modelcolor in zip(models, modelpars, modellabels, modelcolors):
        zgr = model(xgr, modelpar)
        if colorful:
            cs = ax.contourf(tx, ty, zgr.reshape(tx.shape), 22, cmap='RdYlGn_r')
            ax.contour(cs, colors=modelcolor, linewidths=0.5)
        else:
            ax.contour(tx, ty, zgr.reshape(tx.shape), levels=22, linewidths=1, colors=modelcolor)
        j+=1


    # ax.plot([0], [0], 'ko')
    # plt.annotate('Something', (-1,0))


    if clearax:
        #ax.set_frame_on(False)
        #ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(False)
        # xmin, xmax = ax.get_xaxis().get_view_interval()
        # ymin, ymax = ax.get_yaxis().get_view_interval()
        # thisax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    else:
        ax.xaxis.set_ticks([-1, 0, 1])
        ax.xaxis.set_ticklabels(ticklabelsx)
        ax.yaxis.set_ticks([-1, 0, 1])
        ax.yaxis.set_ticklabels(ticklabelsy)

    if legend_show:
        ax.legend()

    if figname is not None:
        plt.savefig(figname)

#############################################################

def plot_2d_anchored(models, modelpars, anchor1, anchor2=None,
                  pad=0.5, scale=1., ngr=111,
                  modellabels=None, squished=False, colorful=False,
                  legend_show=False, modelcolors=None,
                  clearax=False, ncolrow=(3,5)):
    """Plot multiple 1d slices of models all going through a given anchor point or given two anchor points.

    Args:
        models (list[callable]): List of model evaluators.
        modelpars (list[tuple]): List of model parameter tuples, one for each.
        anchor1 (np.ndarray): The first anchor point, 1d array.
        anchor2 (np.ndarray, optional): The second anchor point. Defaults to None, in which case it is selected randomly.
        pad (float, optional): Padding on both sides of the interval, so the slice goes beyond the anchors.
        scale (float, optional): The distance of the second anchor from the first, if randomly selected.
        ngr (int, optional): Number of grid points for plotting.
        modellabels (None, optional): Labels/names of the models for legend.
        squished (bool, optional): If squished, the bases in the plane are not orthogonal.
        colorful (bool, optional): Whether printing with colored surface or simply contours.
        legend_show (bool, optional): Whether to show the legend or not.
        modelcolors (list, optional): List of model colors.
        clearax (bool, optional): Clear axes ticks and labels for less busy plotting.
        ncolrow (tuple, optional): Number of columns and rows in a tuple. Defaults to (3, 5).
    """

    # TODO: need to make this cyclic and remove the assertion
    if modelcolors is None:
        modelcolors = ['r', 'g', 'b', 'm', 'y', 'k']
        assert(len(models)<=6)

    ncol, nrow = ncolrow
    ntot = ncol * nrow


    figs, axarr = plt.subplots(nrow, ncol, figsize=(ncol*4, nrow*4))
    if ntot==1: axarr=[[axarr]]

    if anchor2 is None:
        anchor2 = sample_sphere(center=anchor1, rad=scale, nsam=1).reshape(-1,)

    for i in range(ntot):
        print(f"Plotting slice {i+1} / {ntot}")
        irow, icol = divmod(i, ncol)

        thisax = axarr[irow][icol]

        plot_2d_anchored_single(models, modelpars, anchor1, anchor2=anchor2,
                     pad=pad, ngr=ngr, squished=squished, scale=scale,
                     modellabels=modellabels, modelcolors=modelcolors,
                     clearax=clearax, colorful=colorful,
                     legend_show=False,
                     ax=thisax, figname=None)

    if legend_show:
        custom_lines = [Line2D([0], [0], color=cc, linestyle='-', lw=2) for cc in modelcolors]
        leg = plt.legend(custom_lines, modellabels, bbox_to_anchor=(-2.0,-0.3),
                         ncol=6, loc="lower left")

    plt.savefig('fcn_2dslices.png')

#############################################################

#############################################################

def plot_fcn_1d_slice(fcn, domain, idim=0, nom=None, ngr=133, color='b', lw=2, ax=None):
    """Plotting 1d slice of a function, keeping the rest of the inputs at a given nominal.

    Args:
        fcn (callable): Function evaluator.
        domain (np.ndarray): Domain of evaluation, a 2d array of size `(d, 2)`.
        idim (int, optional): Dimension, with respect to which the slice is plotted.
        nom (None, optional): Nominal values, an 1d array of size `(d,)`.
        ngr (int, optional): Number of grid points, i.e. resolution.
        color (str, optional): Color of the plot.
        lw (int, optional): Line width.
        ax (plt.Axes, optional): Axis handle. If None, use the current axis.
    Note:
        ax is changed as a result of this function. Further beautification and figure saving should be done outside this function.
    """
    if ax is None:
        ax = plt.gca()

    if nom is None:
        nom = np.mean(domain, axis=1)

    xg1 = np.linspace(domain[idim, 0], domain[idim, 1], ngr)
    xg = np.tile(nom, (ngr*ngr, 1))
    xg[:, idim] = xg1

    yg = fcn(xg)

    ax.plot(xg1, yg, '-', color=color, linewidth=lw)


#############################################################

def plot_fcn_2d_slice(fcn, domain, idim=0, jdim=1, nom=None, ngr=33, ax=None):
    """Plotting 2d slice of a function, keeping the rest of the inputs at a given nominal.

    Args:
        fcn (callable): Function evaluator.
        domain (np.ndarray): Domain of evaluation, a 2d array of size `(d, 2)`.
        idim (int, optional): First dimension, with respect to which the slice is plotted.
        jdim (int, optional): Second dimension, with respect to which the slice is plotted.
        nom (None, optional): Nominal values, an 1d array of size `(d,)`.
        ngr (int, optional): Number of grid points per dimension, i.e. resolution.
        color (str, optional): Color of the plot.
        ax (plt.Axes, optional): Axis handle. If None, use the current axis.
    Note:
        ax is changed as a result of this function. Further beautification and figure saving should be done outside this function.
    """
    if ax is None:
        ax = plt.gca()

    if nom is None:
        nom = np.mean(domain, axis=1)

    xg1 = np.linspace(domain[idim, 0], domain[idim, 1], ngr)
    xg2 = np.linspace(domain[jdim, 0], domain[jdim, 1], ngr)

    xg = np.tile(nom, (ngr*ngr, 1))

    X, Y = np.meshgrid(xg1, xg2)
    xx = np.vstack((X.flatten(), Y.flatten())).T  # xx.shape is (ngr^2,2)

    xg[:, idim] = xx[:, 0]
    xg[:, jdim] = xx[:, 1]

    yg = fcn(xg).reshape(X.shape)

    cs = ax.contourf(X, Y, yg, 22, cmap='RdYlGn_r')
    ax.contour(cs, colors='k', linewidths=0.5)
    plt.colorbar(cs, ax=ax)


#############################################################

def plot_uc_sample(pred_sam, data, nqt=111, label='', ax=None):
    """Plotting uncertainty calibration figure given samples.

    Args:
        pred_sam (np.ndarray): Samples, in a 2d array of size `(M,N)`.
        data (np.ndarray): Data, in a 1d array of size `(N,)`.
        nqt (int, optional): Number of quantiles used. Essentially, the resolution.
        label (str, optional): Custom label.
        ax (plt.Axes, optional): Axis handle. If None, use the current axis.
    Returns:
        tuple: Data fractions and quantile values corresponding to these fractions.
    Note:
        ax is changed as a result of this function. Further beautification and figure saving should be done outside this function.
    """
    nsam, nx_ = pred_sam.shape
    nx = data.shape[0]
    assert(nx==nx_)
    qq = np.linspace(0.0, 1.0, num=nqt, endpoint=True)
    pred_qt = np.quantile(pred_sam, qq, axis=0)
    frac_qq = np.zeros_like(qq)
    for i in range(nqt):
        frac_qq[i] = np.sum(np.array([int(j) for j in data<pred_qt[i,:]]))/nx

    if ax is None:
        plt.figure(figsize=(9,9))
        plt.plot(qq, frac_qq, 'o-', markersize=1, label=label)
        plt.plot([0,1], [0,1], '--', linewidth=2)
        plt.xlabel('Prediction Quantile')
        plt.ylabel('Data Fraction')
        plt.xlim([0.,1.])
        plt.ylim([0.,1.])
        plt.savefig('uc_sample.png')
    else:
        ax.plot(qq, frac_qq, 'o-', markersize=1, label=label)
        ax.plot([0,1], [0,1], '--', linewidth=2)


    return frac_qq, qq

#############################################################

def plot_uc_exact(pred_mean, pred_std, data, nqt=111, label='', ax=None):
    """Plotting uncertainty calibration figure given mean and standard deviation of predictions.

    Args:
        pred_mean (np.ndarray): Prediction mean, in a 1d array of size `(N,)`.
        pred_std (np.ndarray): Prediction standard deviation, in a 1d array of size `(N,)`.
        data (np.ndarray): Data, in a 1d array of size `(N,)`.
        nqt (int, optional): Number of quantiles used. Essentially, the resolution.
        label (str, optional): Custom label.
        ax (plt.Axes, optional): Axis handle. If None, use the current axis.
    Returns:
        tuple: Data fractions and quantile values corresponding to these fractions.
    Note:
        ax is changed as a result of this function. Further beautification and figure saving should be done outside this function.
    """
    nx = pred_mean.shape[0]
    nx_ = pred_std.shape[0]
    assert(nx==nx_)
    qq = np.linspace(0.0, 1.0, num=nqt, endpoint=True)

    pred_qt_exact = np.zeros((nqt, nx))
    for i in range(nqt):
        for j in range(nx):
            pred_qt_exact[i, j] = ss.norm.ppf(qq[i], loc=pred_mean[j], scale=pred_std[j])

    frac_qq_exact = np.zeros_like(qq)
    for i in range(nqt):
        frac_qq_exact[i] = np.sum(np.array([int(j) for j in data<pred_qt_exact[i,:]]))/nx

    if ax is None:
        plt.figure(figsize=(9,9))
        plt.plot(qq, frac_qq_exact, 'o-', markersize=1, label=label)
        plt.plot([0,1], [0,1], '--', linewidth=2)
        plt.xlabel('Prediction Quantile')
        plt.ylabel('Data Fraction')
        plt.xlim([0.,1.])
        plt.ylim([0.,1.])
        plt.savefig('uc_exact.png')
    else:
        ax.plot(qq, frac_qq_exact, 'o-', markersize=1, label=label)
        ax.plot([0,1], [0,1], '--', linewidth=2)


    return frac_qq_exact, qq


def plot_samples_pdfs(xx_list, legends=None, colors=None, file_prefix='x', title=''):
    """Plots multiple pdfs given list of samples

    Args:
        xx_list (list[np.ndarray]): List of samples.
        legends (list[str], optional): List of legends. Defaults to generic text.
        colors (list[str], optional): List of colors. Defaults to generic color cycle.
        file_prefix (str, optional): Figure file prefix.
        title (str, optional): Figure title. Default is no title.
    """
    npdfs = len(xx_list)
    ndim = xx_list[0].shape[1]

    if colors is None:
        colors = ['b', 'r', 'g']*npdfs #overkill
    if legends is None:
        legends = ['Data #'+str(i) for i in range(npdfs)]
    for idim in range(ndim):
        ff = plt.figure(figsize=(10,9))
        for i in range(npdfs):
            plot_pdf1d(xx_list[i][:, idim], pltype='kde', color=colors[i], lw=1, label=legends[i], ax=plt.gca())
        h, l = plt.gca().get_legend_handles_labels()
        plt.xlabel(f'x$_{idim+1}$')
        plt.gca().legend()
        plt.title(title)
        plt.savefig(file_prefix+f'_d{idim}.png')
        plt.clf()
        for jdim in range(idim+1, ndim):
            ff = plt.figure(figsize=(10,9))
            for i in range(npdfs):
                plot_pdf2d(xx_list[i][:, idim], xx_list[i][:, jdim], pltype='kde', ncont=10, color=colors[i], lwidth=1, ax=plt.gca())

            plt.xlabel(f'x$_{idim+1}$')
            plt.ylabel(f'x$_{jdim+1}$')
            plt.gca().legend(h, l)
            plt.title(title)
            plt.savefig(file_prefix+f'_d{idim}_d{jdim}.png')
            plt.clf()

    return

def plot_1d(func, domain, ax=None, idim=0, odim=0, nom=None, ngr=100, color='orange', label='', lstyle='-', figname='func1d.png'):
    """Plotting 1d slice of a function.

    Args:
        func (callable): The callable function of interest.
        domain (np.ndarray): A dx2 array indicating the domain of the function.
        ax (None, optional): Axis object to plot on. If None, plots on current axis.
        idim (int, optional): Input dimension to plot against.
        odim (int, optional): Output QoI to plot against. Useful for multioutput funtions.
        nom (np.ndarray, optional): Nominal value to fix non-plotted dimensions at. An array of size d. If None, uses the domain center.
        ngr (int, optional): Number of grid points.
        color (str, optional): Color of the graph.
        label (str, optional): Label of the graph.
        lstyle (str, optional): Linestyle of the graph.
        figname (str, optional): Figure name to save.
    """
    if ax is None:
        ax = plt.gca()

    if nom is None:
        nom = np.mean(domain, axis=1)

    xg1 = np.linspace(domain[idim, 0], domain[idim, 1], ngr)

    xg = np.tile(nom, (ngr, 1))


    xg[:, idim] = xg1
    yg = func(xg)[:, odim]

    ax.plot(xg[:, idim], yg, color=color, label=label, linestyle=lstyle)
    plt.savefig(figname)

    return

def plot_2d(func, domain, ax=None, idim=0, jdim=1, odim=0, nom=None, ngr=33, figname='func2d.png'):
    """Plotting 2d slice of a function.

    Args:
        func (callable): The callable function of interest.
        domain (np.ndarray): A dx2 array indicating the domain of the function.
        ax (None, optional): Axis object to plot on. If None, plots on current axis.
        idim (int, optional): First input dimension to plot against.
        jdim (int, optional): Second input dimension to plot against.
        odim (int, optional): Output QoI to plot against. Useful for multioutput funtions.
        nom (np.ndarray, optional): Nominal value to fix non-plotted dimensions at. An array of size d. If None, uses the domain center.
        ngr (int, optional): Number of grid points.
        figname (str, optional): Figure name to save.
    """
    if ax is None:
        ax = plt.gca()

    if nom is None:
        nom = np.mean(domain, axis=1)

    xg1 = np.linspace(domain[idim, 0], domain[idim, 1], ngr)
    xg2 = np.linspace(domain[jdim, 0], domain[jdim, 1], ngr)

    xg = np.tile(nom, (ngr*ngr, 1))

    X, Y = np.meshgrid(xg1, xg2)
    xx = np.vstack((X.flatten(), Y.flatten())).T  # xx.shape is (ngr^2,2)

    xg[:, idim] = xx[:, 0]
    xg[:, jdim] = xx[:, 1]

    yg = func(xg)[:, odim].reshape(X.shape)

    cs = ax.contourf(X, Y, yg, 22, cmap='RdYlGn_r')
    ax.contour(cs, colors='k', linewidths=0.5)
    plt.colorbar(cs, ax=ax)
    plt.savefig(figname)

    return

########################################################################

def plot_parity(y1, y2, labels=['y1', 'y2'], filename='parity.png'):
    """A minimal parity plot.

    Args:
        y1 (np.ndarray): The 1d array on the x-axis.
        y2 (np.ndarray): The 1d array on the y-axis.
        labels (list, optional): List of length two for the axes labels.
        filename (str, optional): Figure filename to save.
    """
    plt.figure(figsize=(6,6))
    plt.plot(y1, y2,'o')
    llim=min(y1.min(), y2.min())
    ulim=max(y1.max(), y2.max())
    delt = ulim - llim
    plt.xlim((llim-0.1*delt, ulim+0.1*delt))
    plt.ylim((llim-0.1*delt, ulim+0.1*delt))
    plt.plot([llim, ulim], [llim, ulim])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(filename)


#############################################################

def plot_cov(mm, cc, ngr=100, f=3., pnames=None, ax=None, savefig=False):
    """Plotting covariance contour given mean and covariance matrix.

    Args:
        mm (np.ndarray): Mean, an 1d array of size `(2,)`.
        cc (np.ndarray): Covariance matrix, a 2d array of size `(2,2)`.
        ngr (int, optional): Number of grid points per dimension, i.e. resolution.
        f (float, optional): Factor for the plotting range in terms of standard deviations.
        pnames (list, optional): List of parameter names. If None, generic names are used.
        ax (plt.Axes, optional): Axis handle. If None, use the current axis.
        savefig (bool, optional): Whether to save the figure or not.
    """
    if ax is None:
        ax = plt.gca()

    if pnames is None:
        pnames = ['p1', 'p2']

    x = np.linspace(mm[0]-f*np.sqrt(cc[0, 0]), mm[0]+f*np.sqrt(cc[0,0]), ngr)
    y = np.linspace(mm[1]-f*np.sqrt(cc[1, 1]), mm[1]+f*np.sqrt(cc[1,1]), ngr)
    X, Y = np.meshgrid(x, y)

    try:
        rv = ss.multivariate_normal(mm, cc, allow_singular=True)
        XY = np.dstack((X, Y))

        Z = rv.pdf(XY)
        ax.contour(X,Y,Z)

        if savefig:
            ax.set_xlabel(pnames[0])
            ax.set_ylabel(pnames[1])
            plt.savefig(f'cov_{pnames[0]}_{pnames[1]}.png')

    except ValueError:
        print(f"Covariance for pair ({i},{j}) is not positive-semidefinite.")


def plot_cov_tri(mean, cov, names=None, figname='cov_tri.png'):
    """Plots covariance in a triangular pairwise way.

    Args:
        mean (np.ndarray): Mean, an 1d array of size `(npar,)`.
        cov (np.ndarray): Covariance matrix, a 2d array of size `(npar,npar)`.
        names (list, optional): List of parameter names. If None, generic names are used.
        figname (str, optional): Figure filename to save.
    """
    npar = mean.shape[0]
    figs, axarr = plt.subplots(npar, npar, figsize=(15, 15))
    if npar==1: axarr=[[axarr]]

    if names is None:
        names = ['p'+str(j) for j in range(npar)]
    assert(len(names)==npar)

    for i in range(npar):
        thisax = axarr[i][i]
        x = np.linspace(mean[i]-3.0*np.sqrt(cov[i,i]), mean[i]+3.0*np.sqrt(cov[i,i]), 100)
        rv = ss.norm(mean[i], np.sqrt(cov[i,i]))
        z = rv.pdf(x)
        thisax.plot(x, z, 'b-')

        if i == 0:
            thisax.set_ylabel(names[i])
        if i == npar - 1:
            thisax.set_xlabel(names[i])
        if i > 0:
            thisax.yaxis.set_ticks_position("right")
        # thisax.yaxis.set_label_coords(-0.12, 0.5)


        for j in range(i):
            thisax = axarr[i][j]
            axarr[j][i].axis('off')

            mm = np.array([mean[j], mean[i]])
            cc = np.array([[cov[j,j], cov[i,j]],[cov[j,i], cov[i,i]]])
            plot_cov(mm, cc, f=3., pnames=[f'p{i}', f'p{j}'], ngr=100, ax=thisax, savefig=False)

            # x0, x1 = thisax.get_xlim()
            # y0, y1 = thisax.get_ylim()
            # #thisax.set_aspect((x1 - x0) / (y1 - y0))

            if j == 0:
                thisax.set_ylabel(names[i])
            if i == npar - 1:
                thisax.set_xlabel(names[j])
            if j > 0:
                thisax.yaxis.set_ticklabels([])

    plt.savefig(figname)

####################################################################################


def plot_sensmat(sensdata,pars,cases,par_labels=[],case_labels=[],cutoff=-1000., figname='sensmat.png'):
    r"""Plot sensitivity matrix as a heatmap or bar plot.

    Args:
        sensdata (np.ndarray): 2d array of sensitivity data, size (ncases, npars).
        pars (list): List of parameter names.
        cases (list): List of case names.
        par_labels (list, optional): List of parameter labels for plotting. Defaults to None, which uses generic names.
        case_labels (list, optional): List of case labels for plotting. Defaults to None, which uses generic names.
        cutoff (float, optional): Cutoff value for sensitivity inclusion. Defaults to -1000.
        figname (str, optional): Figure filename to save. Defaults to 'sensmat.png'.
    """

    cdict = mpl.cm.jet._segmentdata.copy()
    cdict['red']=tuple([tuple([0.0,  1,   1  ]),
                        tuple([0.01, 0,   0  ]),
                        tuple([0.35, 0,   0  ]),
                        tuple([0.66, 1,   1  ]),
                        tuple([0.89, 1,   1  ]),
                        tuple([1,    0.5, 0.5])
                        ]
                       )
    cdict['green']=tuple([tuple([0.0,   1, 1]),
                          tuple([0.01,  0, 0]),
                          tuple([0.125, 0, 0]),
                          tuple([0.375, 1, 1]),
                          tuple([0.64,  1, 1]),
                          tuple([0.91,  0, 0]),
                          tuple([1,     0, 0])
                          ]
                         )
    cdict['blue']=tuple([tuple([0,    1.0,1.0]),
                         tuple([0.01, 0.5,0.5]),
                         tuple([0.11, 1,  1  ]),
                         tuple([0.34, 1,  1  ]),
                         tuple([0.65, 0,  0  ]),
                         tuple([1,    0,  0  ])
                         ]
                        )


    cp=mpl.colors.LinearSegmentedColormap('colormap',cdict,64)

    # Read varfrac files and retain indices of important params
    vlst=[]
    allSens=[]
    for nm in range(len(cases)):
        #vfr=np.array(column(readfile("varfrac."+nm+".dat")[0],0))
        vfr=sensdata[nm,:] #np.array(column(readfile(nm+".vf.dat")[0],0))
        allSens.append(vfr)
        vlst.append([ n for n,i in enumerate(vfr) if i>=cutoff ])
    # Get union
    allV=[]
    for i in range(len(vlst)):
        allV=list(set(allV) | set(vlst[i]))
    allV=np.sort(allV)
    # Create matrix, populate, and rescale
    nobs=len(cases);
    npar=len(allV);
    print("Number of observables plotted = %d" % nobs)
    print("Number of parameters plotted = %d" % npar)

    if par_labels is None:
        par_labels = ['p'+str(j) for j in range(npar)]
    if case_labels is None:
        case_labels = ['out'+str(j) for j in range(nobs)]


    jsens=np.array(np.zeros([nobs,npar]));
    for i in range(nobs):
        for j in range(npar):
            jsens[i,j]=allSens[i][allV[j]];
    #for i in range(nobs):
    #    jsens[i]=jsens[i]/jsens[i].max();
    jsens[np.where(jsens==0)]=0.5*jsens[np.where(jsens>0)].min();
    #for i in range(nobs):
     #   for j in range(npar):
      #      jsens[i,j]=np.log10(jsens[i,j]);

    par_labels_sorted=[];
    for i in allV:
        par_labels_sorted.append(par_labels[i]);
    # make fig
    fs1=13;
    fig = plt.figure(figsize=(10,3.9));
    ax=fig.add_axes([0.12, 0.27, 0.88, 0.68]);
    cs=ax.pcolor(jsens,cmap=cp);
    #cs=ax.pcolor(jsens,cmap=mpl.cm.jet)
    ax.set_xlim([0,npar]);
    ax.set_ylim([0,nobs]);
    ax.set_xticks([0.5+i for i in range(npar)]);
    ax.set_yticks([0.4+i for i in range(nobs)]);
    ax.set_yticklabels([case_labels[i] for i in range(nobs)],fontsize=fs1);
    ax.set_xticklabels([par_labels_sorted[i] for i in range(npar)],rotation=45,fontsize=fs1);
    ax.tick_params(length=0.0)
    cbar=plt.colorbar(cs)
    #cbar.set_ticks(range(-13,1,1))
    #cbar.set_ticklabels(['$10^{'+str(i)+'}$' for i in range(-13,1,1)])

    ax.grid(False);
    plt.savefig(figname)


def plot_joy(sams_list, xcond, outnames, color_list, nominal=None, offset_factor=1.0, ax=None, figname='joyplot.png'):
    r"""Plots a joyplot of multiple sample sets along given output conditions.

    Args:
        sams_list (list[np.ndarray]): List of sample sets, each in a 2d array of size `(M,N)`.
        xcond (list[float]): List of output condition values, length N.
        outnames (list[str]): List of output condition names, length N.
        color_list (list[str]): List of colors for each sample set.
        nominal (np.ndarray, optional): Nominal values for each output condition, an 1d array of size `(N,)`. Defaults to None.
        offset_factor (float, optional): Factor to scale the pdf heights. Defaults to 1.0.
        ax (plt.Axes, optional): Axis handle. If None, use the current axis.
        figname (str, optional): Figure filename to save.
    """
    if ax is None:
        ax = plt.gca()

    for ic, xc in enumerate(xcond):
        # plot gray horizontal lines where bottom of each pdf will sit
        offset = xc
        for isam, sams in enumerate(sams_list):
            a = 4.
            pnom = np.mean(sams[:, ic])
            pdel = np.std(sams[:, ic])
            pgrid = np.linspace(pnom - a * pdel, pnom + a * pdel, 111)
            kde_py = ss.gaussian_kde(sams[:, ic], 'silverman')
            pdf = kde_py(pgrid)
            ff = offset_factor
            ax.fill_between(pgrid, xc * np.ones_like(pgrid), ff * pdf + offset,
                            fc=color_list[isam], ec='black',
                            lw=1, label='', alpha=0.4)
        if nominal is not None:
            ax.plot([nominal[ic], nominal[ic]], [
                    offset, offset + 1.2 * ff * np.max(pdf)], 'k--', lw=1)
    ax = plt.gca()
    ax.grid(False, axis='x')
    ax.set_yticks(xcond)
    ax.set_yticklabels(outnames)
    plt.tight_layout()
    plt.savefig(figname)
