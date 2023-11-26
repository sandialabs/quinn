#!/usr/bin/env python
"""Module for various mapping functions."""

import numpy as np


def scale01ToDom(xx, dom):
    """Scaling an array to a given domain, assuming \
       the inputs are in [0,1]^d.

    Args:
        xx (np.ndarray): Nxd input array.
        dom (np.ndarray): dx2 domain.
    Returns:
        np.ndarray: Nxd scaled array.
    Note:
        If input is outside [0,1]^d, a warning is given, but the scaling will happen nevertheless.
    """
    if np.any(xx<0.0) or np.any(xx>1.0):
        print("Warning: some elements are outside the [0,1] range.")

    return xx*np.abs(dom[:,1]-dom[:,0])+np.min(dom, axis=1)

def scaleDomTo01(xx, dom):
    """Scaling an array from a given domain to [0,1]^d.

    Args:
        xx (np.ndarray): Nxd input array.
        dom (np.ndarray): dx2 domain.
    Returns:
        np.ndarray: Nxd scaled array.
    Note:
        If input is outside domain, a warning is given, but the scaling will happen nevertheless.
    """
    xxsc = (xx-np.min(dom, axis=1)) / np.abs(dom[:,1]-dom[:,0])
    if np.any(xxsc<0.0) or np.any(xxsc>1.0):
        print("Warning: some elements are outside the [0,1] range.")

    return xxsc

def scaleTo01(xx):
    """Scale an array to [0,1], using dimension-wise min and max.

    Args:
        xx (np.ndarray): Initial 2d array

    Returns:
        np.ndarray: Scaled array.
    """
    return (xx - np.min(xx, axis=0)) / (np.max(xx, axis=0) - np.min(xx, axis=0))

def standardize(xx):
    """Normalize an array, i.e. map it to zero mean and unit variance.

    Args:
        xx (np.ndarray): Initial 2d array

    Returns:
        np.ndarray: Normalized array.
    """
    return (xx - np.mean(xx)) / np.std(xx)

class XMap():
    """Base class for a map."""

    def __init__(self):
        """Initialization."""
        ...

    def __call__(self, x):
        raise NotImplementedError("Base XMap call is not implemented")

    def forw(self, x):
        """Forward map.

        Args:
            x (np.ndarray): 2d numpy input array.

        Returns:
            np.ndarray: 2d numpy output array.
        """
        return self.__call__(x)

    def inv(self, xs):
        """Inverse of the map.

        Args:
            xs (np.ndarray): 2d numpy array.
        Returns:
            np.ndarray: if implemented, 2d numpy array.
        """
        raise NotImplementedError("Base XMap inverse is not implemented")

class Expon(XMap):
    """Exponential map."""

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return np.exp(x)

    def inv(self, xs):
        return np.log(xs)

class Logar(XMap):
    """Logarithmic map."""

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return np.log(x)

    def inv(self, xs):
        return np.exp(xs)

class ComposeMap(XMap):
    """Composition of two maps."""

    def __init__(self, map1, map2):
        """Initialize with the two maps to be composed.

        Args:
            map1 (XMap): Inner map
            map2 (XMap): Outer map
        """
        super().__init__()
        self.map1 = map1
        self.map2 = map2

    def __repr__(self):
        return f"ComposeMap({self.map1=}, {self.map2=}"

    def __call__(self, x):
        return self.map2(self.map1(x))

    def inv(self, xs):
        return self.map1.inv(self.map2.inv(xs))


class LinearScaler(XMap):
    """Linear scaler map."""

    def __init__(self, shift=None, scale=None):
        """Initialize with shift and scale.

        Args:
            shift (np.ndarray, optional): Shift array, broadcast-friendly
            scale (np.ndarray, optional): Scale array, broadcast-friendly
        """
        super().__init__()
        self.shift = shift
        self.scale = scale
        return

    def __repr__(self):
        return f"Scaler({self.shift=}, {self.scale=}"

    def __call__(self, x):
        if self.shift is None:
            xs = x - 0.0
        else:
            xs = x - self.shift

        if self.scale is None:
            xs /= 1.0
        else:
            xs /= self.scale

        return xs

    def inv(self, xs):
        if self.scale is None:
            x = xs * 1.0
        else:
            x = xs * self.scale#.reshape(1,-1)

        if self.shift is None:
            x += 0.0
        else:
            x += self.shift

        return x

class Standardizer(LinearScaler):
    """Standardizer map, linearly scaling data to zero mean and unit variance."""

    def __init__(self, x):
        """Initialize with a given 2d array.

        Args:
            x (np.ndarray): Data according to which the standardization happens.
        Note:
            This also can be accomplished by function `normalize`
        """
        super().__init__(shift=np.mean(x, axis=0), scale=np.std(x, axis=0))
        return

class Normalizer(LinearScaler):
    """Normalizer map, linearly scaling data to [0,1]."""

    def __init__(self, x, nugget=0.0):
        """Initialize with a given 2d array and a nugget to keep slightly above zero.

        Args:
            x (np.ndarray): Data according to which the normalization happens.
            nugget (float, optional): Small value to keep data above zero if needed.
        Note:
            When nugget is 0, this also can be accomplished by function `scaleTo01`
        """
        super().__init__(shift=np.min(x, axis=0)-nugget,
                         scale=np.max(x, axis=0)-np.min(x, axis=0))
        return

class Domainizer(LinearScaler):
    """Domainizer map, linearly scaling data (assumed to be in [0,1]) to a given domain.

    Note:
        This also can be accomplished by functions `scaleDomTo01` and its inverse `scale01ToDom`.
    """

    def __init__(self, dom):
        """Initialize with a given domain.

        Args:
            dom (np.ndarray): Domain of size `(d,2)` according to which the normalization happens.
        """
        super().__init__(shift=dom[:,0], scale=dom[:,1]-dom[:,0])
        return

class Affine(XMap):
    """Affine map."""

    def __init__(self, weight=None, bias=None):
        """Initializes with weight and bias arrays.

        Args:
            weight (np.ndarray, optional): 2d array
            bias (np.ndarray, optional): 1d array
        """
        super().__init__()
        self.weight = weight
        self.bias = bias
        return

    def __repr__(self):
        return f"Scaler({self.weight=}, {self.bias=}"


    def __call__(self, x):
        if self.weight is None:
            xs = x * 1.0
        else:
            xs = x @ self.W.T

        if self.bias is None:
            xs += 0.0
        else:
            xs += self.bias

        return xs

    def inv(self, xs):
        if self.bias is None:
            x = xs - 0.0
        else:
            x = xs - self.bias

        if self.weight is None:
            x *= 1.0
        else:
            x = x @ np.linalg.inv(self.W.T)

        return x


