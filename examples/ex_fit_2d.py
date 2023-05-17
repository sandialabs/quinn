#!/usr/bin/env python
"""An example of a 2d function approximation, and an example of a periodic boundary regularization."""

import os
import sys
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt


from quinn.func.funcs import Ackley, Sine, Sine10, blundell
from quinn.nns.mlp import MLP
from quinn.nns.tchutils import tch
from quinn.nns.nnwrap import NNWrap
from quinn.nns.losses import CustomLoss, PeriodicLoss

from quinn.utils.plotting import myrc, plot_fcn_2d_slice
from quinn.utils.maps import scale01ToDom

def main():
    torch.set_default_dtype(torch.double)
    myrc()

    # defaults to cuda:0
    device_id='cuda:0'
    # use: ./ex_ufit uq_method device_id, where uq_method: 'mcmc' 'vi' 'ens', and 
    # device_id: 0, 1,... depending on number of gpus
    if len(sys.argv) > 1:
        device_id=sys.argv[1]
    device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
    print("Using device",device)

    ###########################################################################
    ###########################################################################

    ## Set up
    ndim = 2 # input dimensionality
    nall = 55 # total number of points
    trn_factor = 0.8 # which fraction of nall goes to training
    ntst = 13 # separate test set
    datanoise = 0.02 # Noise in the generated data

    # True model is Ackley function with one output
    true_model, nout = Ackley, 1

    # Function domain
    domain = np.tile(np.array([-1.5, 1.5]), (ndim, 1))

    # Get x data
    xall = scale01ToDom(np.random.rand(nall, ndim), domain)
    if true_model is not None:
        yall = true_model(xall, datanoise=datanoise)

    # Sample test
    if ntst > 0:
        xtst = scale01ToDom(np.random.rand(ntst, ndim), domain)
        if true_model is not None:
            ytst = true_model(xtst, datanoise=datanoise)

    # Model to fit
    nnet = MLP(ndim, nout, (11,11,11), biasorno=True,
               activ='tanh', bnorm=False, bnlearn=True, dropout=0.0,
               device=device)

    # Data split to training and validation
    ntrn = int(trn_factor * nall)
    indperm = np.random.permutation(range(nall))
    indtrn = indperm[:ntrn]
    indval = indperm[ntrn:]
    xtrn, xval = xall[indtrn, :], xall[indval, :]
    ytrn, yval = yall[indtrn, :], yall[indval, :]


    # Set up a periodic boundary
    ngr = 11
    bdry1 = -1.5*np.ones((ngr, ndim))
    bdry1[:,1] = np.linspace(-1.5, 1.5, ngr)
    bdry2 = 1.5*np.ones((ngr, ndim))
    bdry2[:,1] = np.linspace(-1.5, 1.5, ngr)
    # pass input tensors with device 
    loss = PeriodicLoss([nnet.nnmodel, 10.1, tch(bdry1,device=device), tch(bdry2,device=device)]) #None #CustomLoss([nnet.nnmodel, 1.0])
    nnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=10, nepochs=1000, loss=loss)
    print("=======================================")


    # Plot the true model and the NN approximation
    figs, axarr = plt.subplots(1, 2, figsize=(18, 7))
    plot_fcn_2d_slice(true_model, domain, idim=0, jdim=1, nom=None, ngr=33, ax=axarr[0])
    plot_fcn_2d_slice(NNWrap(nnet), domain, idim=0, jdim=1, nom=None, ngr=33, ax=axarr[1])
    axarr[0].plot(xtrn[:,0], xtrn[:,1], 'ko', ms=11, markeredgecolor='w')
    for ax in axarr:
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
    axarr[0].set_title('True Model')
    axarr[1].set_title('NN Apprx.')
    plt.savefig('fcn2d.png')

    # Prepare lists of inputs and outputs for plotting
    xx_list = [xtrn, xval]
    yy_list = [ytrn, yval]
    ll_list = ['Training', 'Validation']
    if ntst > 0:
        xx_list.append(xtst)
        yy_list.append(ytst)
        ll_list.append('Testing')
    # A diagonal plot to check the approximation
    nnet.predict_plot(xx_list, yy_list, labels=ll_list)


if __name__ == '__main__':
    main()
