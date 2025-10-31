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
    pk.dump(sobj, open(nameprefix + '.pk', 'wb'), protocol=pk.HIGHEST_PROTOCOL)


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

    # final_list = []
    # for element in itertools.product(*somelists):
    #     final_list.append(element)

    final_list = list(itertools.product(*somelists))

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

####################################################################
####################################################################

def get_opt_bw(xsam, bwf=1.0):
    """Get the rule-of-thumb optimal bandwidth for kernel density estimation.

    Args:
        xsam (np.ndarray): Data array, `(N,d)`
        bwf (float): Factor behind the scaling optimal rule
    Returns:
        np.ndarray: Array of length `d`, the optimal per-dimension bandwidth
    """
    nsam, ndim = xsam.shape
    xstd = np.std(xsam, axis=0)
    bw=xstd
    bw *= np.power(4./(ndim+2),1./(ndim+4.))
    bw *= np.power(nsam,-1./(ndim+4.))

    bw *= bwf

    #xmin, xmax = np.min(xsam, axis=0), np.max(xsam, axis=0)

    # in case standard deviation is 0
    bw[bw<1.e-16] = 0.5
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

####################################################################
####################################################################

def strarr(array):
    """Turn an array into a neatly formatted one for annotating figures.

    Args:
        array (np.ndarray): 1d array

    Returns:
        list: list of floats with two decimal digits
    """
    return [float("{:0.2f}".format(i)) for i in array]


####################################################################
####################################################################

def project(a, b):
    """Project a vector onto another vector in high-d space.

    Args:
        a (np.ndarray): The 1d array to be projected.
        b (np.ndarray): The array to project onto.

    Returns:
        tuple(np.ndarray, np.ndarray): tuple (projection, residual) where projection+residual=a, and projection is orthogonal to residual, and colinear with b.
    """
    assert(a.shape[0]==b.shape[0])
    proj = (np.dot(a, b)/ np.dot(b, b))*b
    resid = a - proj
    return proj, resid

####################################################################
####################################################################

def pick_basis(x1, x2, x3, x0=None, random_direction_in_plane=None):
    """Given three points in a high-d space, picks a basis in a plane that goes through these points.

    Args:
        x1 (np.ndarray): 1d array, the first point
        x2 (np.ndarray): 1d array, the second point
        x3 (np.ndarray): 1d array, the third point
        x0 (np.ndarray, optional): 1d array, the central point of basis. Defaults to None, in which case the center-of-mass is selected.
        random_direction_in_plane (np.ndarray, optional): Direction aligned with the first basis. Has to be in the plane already. Defaults to None, in which case a random direction is selected.

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray): tuple(origin, e1, e2) of the origin and two basis directions.
    """
    assert(x1.shape==x2.shape and x1.shape==x3.shape)
    if x0 is None:
        x0 = (x1+x2+x3)/3.

    assert(x0.shape==x1.shape)


    # random direction in that plane
    x1230 = np.vstack((x1-x0, x2-x0, x3-x0))
    assert(np.linalg.matrix_rank(x1230)==2)
    if random_direction_in_plane is None:
        random_direction_in_plane = np.dot(np.random.rand(1, 3), x1230)[0]
    random_direction_in_plane /= np.linalg.norm(random_direction_in_plane)
    # TODO: this assertion occasionally fails, e.g. when running all examples in bulk
    #assert(np.linalg.matrix_rank(np.vstack((x1230, random_direction_in_plane)))==2)

    proj_norms = np.empty(3,)
    resid_norms = np.empty(3,)
    for i in range(3):
        proj, resid = project(x1230[i], random_direction_in_plane)
        proj_norms[i] = np.linalg.norm(proj)
        resid_norms[i] = np.linalg.norm(resid)

    pm = np.argmax(proj_norms)
    rm = np.argmax(resid_norms)

    origin = x0
    e1, _ = project(x1230[pm], random_direction_in_plane)
    _, e2 = project(x1230[rm], random_direction_in_plane)

    return origin, e1, e2

####################################################################
####################################################################

def safe_cholesky(cov):
    r"""Cholesky decomposition with some error handlers, and using SVD+QR trick in case the covariance is degenerate.

    Args:
        cov (np.ndarray): Positive-definite or zero-determinant symmetric matrix `C`.

    Returns:
        np.ndarray: Lower-triangular factor `L` such that `C=L L^T`.
    """

    dim, dim_ = cov.shape
    assert(dim_==dim)
    assert(np.linalg.norm(cov-cov.T)<1.e-14)

    if np.min(np.linalg.eigvals(cov))<0:
        print("The matrix is not a covariance matrix (negative eigenvalues). Exiting.")
        sys.exit()
    elif np.min(np.linalg.eigvals(cov))<1e-14:
        print("Small/near-zero eigenvalue: replacing Cholesky with SVD+QR.")
        u, s, vd = np.linalg.svd(cov, hermitian=True)
        lower = np.linalg.qr(np.dot(np.diag(np.sqrt(s)),vd))[1].T
        signs = np.sign(np.diag(lower))
        lower = np.dot(lower, np.diag(signs))
    else:
        lower = np.linalg.cholesky(cov)

    assert(np.linalg.norm(cov - np.dot(lower, lower.T)) < 1.e-12)

    return lower
