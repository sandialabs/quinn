#!/usr/bin/env python
"""Module for Ensemble NN wrapper."""

import numpy as np

from .learner import Learner
from ..quinn import QUiNNBase


class Ens_NN(QUiNNBase):
    """Deep Ensemble NN Wrapper.

    Attributes:
        nens (int): Number of ensemble members.
        learners (list[Learner]): List of learners.
        dfrac (float): Fraction of data each learner sees.
        verbose (bool): Verbose or not.
    """

    def __init__(self, nnmodel, nens=1, dfrac=1.0, type_ens="ens", verbose=False):
        """Initialization.

        Args:
            nnmodel (torch.nn.Module): PyTorch NN model.
            nens (int, optional): Number of ensemble members. Defaults to 1.
            dfrac (float, optional): Fraction of data for each learner. Defaults to 1.0.
            verbose (bool, optional): Verbose or not.
        """
        super().__init__(nnmodel)
        self.verbose = verbose
        self.nens = nens
        self.dfrac = dfrac
        self.type_ens = type_ens
        self.learners = []
        for i in range(nens):
            self.learners.append(Learner(nnmodel))

        if self.verbose:
            self.print_params(names_only=True)

    def print_params(self, names_only=False):
        """Print model parameter names and optionally, values.

        Args:
            names_only (bool, optional): Print names only. Default is False.
        """
        for i, learner in enumerate(self.learners):
            print(f"==========  Learner {i+1}/{self.nens}  ============")
            learner.print_params(names_only=names_only)

    def fit(self, xtrn, ytrn, **kwargs):
        """Fitting function for each ensemble member.

        Args:
            xtrn (np.ndarray): Input array of size `(N,d)`.
            ytrn (np.ndarray): Output array of size `(N,o)`.
            **kwargs (dict): Keyword arguments.
        """
        for jens in range(self.nens):
            print(f"======== Fitting Learner {jens+1}/{self.nens} =======")

            ntrn = ytrn.shape[0]
            permutation = np.random.permutation(ntrn)
            ind_this = permutation[: int(ntrn * self.dfrac)]

            this_learner = self.learners[jens]

            kwargs["lhist_suffix"] = f"_e{jens}"
            this_learner.fit(xtrn[ind_this], ytrn[ind_this], **kwargs)

    def predict_sample(self, x):
        """Predict a single, randomly selected sample.

        Args:
            x (np.ndarray): Input array of size `(N,d)`.

        Returns:
            np.ndarray: Output array of size `(N,o)`.
        """
        jens = np.random.randint(0, self.nens)
        return self.learners[jens].predict(x)

    def predict_ens(self, x, nens=None):
        """Predict from all ensemble members.

        Args:
            x (np.ndarray): `(N,d)` input array.

        Returns:
            np.ndarray: Array of size `(M, N, o)`, i.e. `M` random samples of `(N,o)` outputs.

        Note:
            This overloads QUiNN's base predict_ens function.
        """
        if nens is None:
            nens = self.nens
        if nens > self.nens and self.type_ens == "ens":
            print(
                f"Warning: Requested {nens} but only {self.nens} ensemble members available."
            )
            nens = self.nens

        permuted_inds = np.random.permutation(nens)

        y_all = []
        for jens in range(nens):
            if self.type_ens == "ens":
                y = self.learners[permuted_inds[jens]].predict(x)
            else:
                y = self.predict_sample(x)
            y_all.append(y)

        return y_all
