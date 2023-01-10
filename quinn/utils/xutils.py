#!/usr/bin/env python
"""Collection of various useful utilities."""

import sys
import os
import itertools
import numpy as np
try:
    import dill as pk
except ModuleNotFoundError:
    import pickle as pk
import matplotlib as mpl

from scipy import stats
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d

def idt(x):
    """Identity function.

    Args:
        x (any type): input

    Returns:
        any type: output
    """
    return x

####################################################################
####################################################################

def savepk(sobj, nameprefix='savestate'):
    """Pickle a python object.

    Args:
        sobj (any type): Object to be pickled.
        nameprefix (str, optional): Name prefix.
    """
    pk.dump(sobj, open(nameprefix + '.pk', 'wb'), -1)


def loadpk(nameprefix='savestate'):
    """Unpickle a python object from a pickle file.

    Args:
        nameprefix (str, optional): Filename prefix

    Returns:
        any type: Unpickled object
    """
    return pk.load(open(nameprefix + '.pk', 'rb'))

####################################################################
####################################################################

def cartes_list(somelists):
    """Generate a list of all combination of elements in given lists.

    Args:
        somelists (list): List of lists
    Returns:
        list[tuple]: List of all combinations of elements in lists that make up somelists
    Example:
        >>> cartes_list([['a', 'b'], [3, 4, 2]])
        [('a', 3), ('a', 4), ('a', 2), ('b', 3), ('b', 4), ('b', 2)]

    """
    final_list = []
    for element in itertools.product(*somelists):
        final_list.append(element)

    return final_list

####################################################################
####################################################################

def read_textlist(filename, nsize, names_prefix=''):
    """Read a textfile into a list containing the rows.

    Args:
        filename (str): File name
        nsize (int): Number of rows in the file
        names_prefix (str, optional): Prefix of a dummy list entry names if the file is not present.

    Returns:
        list[str]: List of elements that are rows of the file
    """
    if os.path.exists(filename):
        with open(filename) as f:
            names = f.read().splitlines()
            assert(len(names) == nsize)
    else:
        names = [names_prefix + '_' + str(i) for i in range(1, nsize + 1)]

    return names

####################################################################
####################################################################

def sample_sphere(center=None, rad=1.0, nsam=100):
    """Sample on a hypersphere of a given radius.

    Args:
        center (np.ndarray, optional): Center of the sphere. Defaults to origin.
        rad (float, optional): Radius of the sphere. Defaults to 1.0.
        nsam (int, optional): Number of samples requested. Defaults to 100.

    Returns:
        np.ndarray: Array of size `(N,d)`
    """
    if center is None:
        center = np.zeros((3,))
    dim = center.shape[0]

    samples = np.random.randn(nsam, dim)
    samples /= np.linalg.norm(samples, axis=1).reshape(-1,1)
    samples *= rad
    samples += center

    return samples

####################################################################
####################################################################

def get_npc(ord, dim):
    """Get number of polynomial basis with a total-degree truncation.

    Args:
        ord (int): Order `p`
        dim (int): Dimension `d`

    Returns:
        int: Number of bases with order up to `p` and dimension `d`, i.e. `(p+d)!/p!d!`.
    """
    npc = 1

    for i in range(ord):
        npc = npc * (dim + i + 1)
    for i in range(ord):
        npc = npc / (i + 1)

    assert(npc==int(npc))

    return int(npc)


def get_mi(ord, dim):
    """Get multiindex array with a total-degree truncation.

    Args:
        ord (int): Order `p`
        dim (int): Dimension `d`

    Returns:
        int np.ndarray: Multiindex array of size `(K,d)`, where `K=(p+d)!/p!d!`.
    """

    assert(dim>0)
    npc = get_npc(ord, dim)
    ic = np.ones(dim, dtype='int')
    iup = 0
    mi = np.zeros((npc, dim), dtype='int')
    if (ord > 0):
        #: first order terms
        for idim in range(dim):
            iup += 1
            mi[iup, idim] = 1
    if (ord > 1):
        #: higher order terms
        for iord in range(2, ord + 1):
            lessiord = iup
            for idim in range(dim):
                for ii in range(idim + 1, dim):
                    ic[idim] += ic[ii]
            for idimm in range(dim):
                for ii in range(lessiord - ic[idimm] + 1, lessiord + 1):
                    iup += 1
                    mi[iup] = mi[ii].copy()
                    mi[iup, idimm] += 1
    return mi

####################################################################
####################################################################

def get_opt_bw(xsam):
    """Get the rule-of-thumb optimal bandwidth for kernel density estimation.

    Args:
        xsam (np.ndarray): Data array, `(N,d)`

    Returns:
        np.ndarray: Array of length `d`, the optimal per-dimension bandwidth
    """
    nsam, ndim = xsam.shape
    xstd = np.std(xsam, axis=0)
    bw=xstd
    bw *= np.power(4./(ndim+2),1./(ndim+4.))
    bw *= np.power(nsam,-1./(ndim+4.))

    #xmin, xmax = np.min(xsam, axis=0), np.max(xsam, axis=0)

    return bw

####################################################################
####################################################################

def get_pdf(data, target):
    """Compute PDF given data at target points.

    Args:
        data (np.ndarray): an `(N,d)` array of `N` samples in `d` dimensions
        target np.ndarray): an `(M,d)` array of target points

    Returns:
        np.ndarray: PDF values at target
    """
    assert(np.prod(np.var(data, axis=0))>0.0)

    # Python Scipy built-in method of KDE
    kde_py=stats.kde.gaussian_kde(data.T)
    dens=kde_py(target.T)

    # Return the target points and the probability density
    return dens
