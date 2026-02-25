#!/usr/bin/env python
"""Test script for Deep Ensemble solver."""

import numpy as np
import torch
from quinn.nns.mlp import MLP
from quinn.solvers.nn_ens import NN_Ens


def test_nn_ens_creation():
    # NN_Ens should be creatable
    net = MLP(1, 1, (8, 8), activ='tanh')
    ens = NN_Ens(net, nens=3)

    assert ens.nens == 3
    assert len(ens.learners) == 3


def test_nn_ens_fit_predict():
    # NN_Ens fit and predict
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (8, 8), activ='tanh')
    ens = NN_Ens(net, nens=2)

    N = 30
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    ens.fit(xtrn, ytrn, nepochs=100, lrate=0.01, freq_out=1000)

    ypred = ens.predict(xtrn)
    assert ypred.shape == (N, 1)


def test_nn_ens_predict_sample():
    # predict_sample should return single prediction
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (8,), activ='tanh')
    ens = NN_Ens(net, nens=2)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    ens.fit(xtrn, ytrn, nepochs=100, lrate=0.01, freq_out=1000)

    y_sample = ens.predict_sample(xtrn)
    assert y_sample.shape == (N, 1)


def test_nn_ens_predict_ens():
    # predict_ens should return ensemble of predictions
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (8,), activ='tanh')
    nens = 3
    ens = NN_Ens(net, nens=nens)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    ens.fit(xtrn, ytrn, nepochs=100, lrate=0.01, freq_out=1000)

    y_ens = ens.predict_ens(xtrn)
    assert y_ens.shape == (nens, N, 1)


def test_nn_ens_predict_mom():
    # predict_mom_sample should return mean, var, cov
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (8,), activ='tanh')
    nens = 3
    ens = NN_Ens(net, nens=nens)

    N = 20
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    ens.fit(xtrn, ytrn, nepochs=100, lrate=0.01, freq_out=1000)

    # msc=0: mean only
    ymean, yvar, ycov = ens.predict_mom_sample(xtrn, msc=0, nsam=nens)
    assert ymean.shape == (N, 1)
    assert yvar is None
    assert ycov is None


def test_nn_ens_dfrac():
    # NN_Ens with data fraction < 1 should still work
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(1, 1, (8,), activ='tanh')
    ens = NN_Ens(net, nens=2, dfrac=0.8)

    N = 40
    xtrn = np.linspace(-1, 1, N).reshape(-1, 1)
    ytrn = np.sin(xtrn)

    ens.fit(xtrn, ytrn, nepochs=100, lrate=0.01, freq_out=1000)

    ypred = ens.predict(xtrn)
    assert ypred.shape == (N, 1)


def test_nn_ens_multioutput():
    # NN_Ens with multi-output network
    np.random.seed(42)
    torch.manual_seed(42)

    net = MLP(2, 2, (8,), activ='tanh')
    ens = NN_Ens(net, nens=2)

    N = 30
    xtrn = np.random.rand(N, 2)
    ytrn = np.column_stack([np.sum(xtrn, axis=1), np.prod(xtrn, axis=1)])

    ens.fit(xtrn, ytrn, nepochs=100, lrate=0.01, freq_out=1000)

    ypred = ens.predict(xtrn)
    assert ypred.shape == (N, 2)


if __name__ == '__main__':
    test_nn_ens_creation()
    test_nn_ens_fit_predict()
    test_nn_ens_predict_sample()
    test_nn_ens_predict_ens()
    test_nn_ens_predict_mom()
    test_nn_ens_dfrac()
    test_nn_ens_multioutput()
