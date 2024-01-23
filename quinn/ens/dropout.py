#!/usr/bin/env python
"""Module for Ensemble NN wrapper."""

import numpy as np
import torch

from .learner import Learner
from ..quinn import QUiNNBase
from .ens import Ens_NN


class Dropout_NN(Ens_NN):
    """Deep Ensemble NN Wrapper where the models are trained and
    evaluated using dropout.
    """

    def __init__(self, nnmodel, nens=1, dfrac=1.0, verbose=False):
        """Initialization.

        Args:
            - nnmodel (torch.nn.Module): PyTorch NN model.
            - nens (int, optional): Number of ensemble members. Defaults to 1.
            - dfrac (float, optional): Fraction of data for each learner.
                Defaults to 1.0.
            - verbose (bool, optional): Verbose or not.
        """
        super().__init__(
            nnmodel,
            nens=nens,
            dfrac=dfrac,
            type_ens="dropout",
            verbose=verbose,
        )

    def predict_sample(self, x, jens=None):
        """Predict a single, randomly selected sample.

        Args:
            x (np.ndarray): Input array of size `(N,d)`.
            jens (int): the ensemble index to use.
        Returns:
            np.ndarray: Output array of size `(N,o)`.
        """
        dim = x.shape[0]
        seed = torch.randint(0, int(1e6), (1,))
        if jens is None:
            jens = np.random.randint(0, self.nens)
        output = []
        for i in range(dim):
            torch.manual_seed(seed)
            output.append(self.learners[jens].predict(x[i]))
        return np.array(output)
