#!/usr/bin/env python
"""Module for Learner class."""

import math
import copy

from ..nns.tchutils import npy, tch, nnfit, print_nnparams

class Learner():
    """A learner class that holds PyTorch NN module and helps train it.

    Attributes:
        nnmodel (torch.nn.Module): Main PyTorch NN module.
        best_model (torch.nn.Module): The best trained PyTorch NN module.
        trained (bool): Whether the module is trained or not.
        verbose (bool): Whether to be verbose or not.
    """

    def __init__(self, nnmodel, verbose=False):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): Main PyTorch NN module.
            verbose (bool): Whether to be verbose or not.
        """
        super().__init__()
        self.nnmodel = copy.deepcopy(nnmodel)
        self.trained = False
        self.verbose = verbose
        self.best_model = None

        if self.verbose:
            self.print_params(names_only=True)

    def print_params(self, names_only=False):
        """Print parameters of the learner's model.

        Args:
            names_only (bool, optional): Whether to print the parameter names only or not.
        """
        if self.trained:
            print_nnparams(self.best_model, names_only=names_only)
        else:
            print_nnparams(self.nnmodel, names_only=names_only)

    def init_params(self):
        """An example of random initialization of parameters.
        """
        for p in self.nnmodel.parameters():
            try:
                stdv = 1. / math.sqrt(p.size(1))
            except IndexError:
                stdv = 1.
            p.data.uniform_(-stdv, stdv)

    def fit(self, xtrn, ytrn, **kwargs):
        """Fitting function for this learner.

        Args:
            xtrn (np.ndarray): Input array of size `(N,d)`.
            ytrn (np.ndarray): Output array of size `(N,o)`.
            **kwargs (dict): Keyword arguments.
        """
        if hasattr(self.nnmodel, 'fit') and callable(getattr(self.nnmodel, 'fit')):
            self.best_model = self.nnmodel.fit(xtrn, ytrn, **kwargs)
        else:
            fit_info = nnfit(self.nnmodel, xtrn, ytrn, **kwargs)
            self.best_model = fit_info['best_nnmodel']
        self.trained = True

    def predict(self, x):
        """Prediction of the learner.

        Args:
            x (np.ndarray): Input array of size `(N,d)`.

        Returns:
            np.ndarray: Output array of size `(N,o)`.
        """
        assert(self.trained)
        device = self.best_model.device
        y = self.best_model(tch(x, rgrad=False, device=device))
        return npy(y)
