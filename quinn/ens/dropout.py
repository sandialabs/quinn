#!/usr/bin/env python
"""Module for Ensemble NN wrapper."""

import numpy as np

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
