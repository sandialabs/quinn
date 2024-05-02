#!/usr/bin/env python
"""An example of a 1d function approximation."""

import torch
import numpy as np

from quinn.nns.mlp import MLP
from quinn.utils.plotting import myrc
from quinn.utils.maps import scale01ToDom
from quinn.func.funcs import Sine, Sine10, blundell



def main():

    torch.set_default_dtype(torch.double)
    myrc()
    # defaults to cuda:0 if available
    device_id='cuda:0'
    device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
    print("Using device",device)


    ########################################################################################
    #################################################################################


    nall = 22 # total number of points
    trn_factor = 0.8 # which fraction of nall goes to training
    ntst = 13 # separate test set
    ndim = 1 # input dimensionality
    datanoise = 0.00 # Noise in the generated data

    # Plot 1d fits or not
    plot_1d = (ndim==1)


    true_model, nout = blundell, ndim
    #true_model, nout = Sine10, 10

    # Function domain
    domain = np.tile(np.array([-0.5, 0.5]), (ndim, 1))


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
               activ='tanh',
               bnorm=False, bnlearn=True,
               dropout=0.0,
               device=device)

    # Data split to training and validation
    ntrn = int(trn_factor * nall)
    indperm = np.random.permutation(range(nall))
    indtrn = indperm[:ntrn]
    indval = indperm[ntrn:]
    xtrn, xval = xall[indtrn, :], xall[indval, :]
    ytrn, yval = yall[indtrn, :], yall[indval, :]
    
    nnet.fit(xtrn, ytrn, val=[xval, yval],
             lrate=0.01,
             batch_size=None, nepochs=2000)

    print("=======================================")
    # print("Best Parameters : ")
    # uqnet.print_params()

    # Prepare lists of inputs and outputs for plotting
    xx_list = [xtrn, xval]
    yy_list = [ytrn, yval]
    ll_list = ['Training', 'Validation']
    if ntst > 0:
        xx_list.append(xtst)
        yy_list.append(ytst)
        ll_list.append('Testing')

    if plot_1d:
        nnet.plot_1d_fits(xx_list, yy_list, labels=ll_list, true_model=true_model)
    nnet.predict_plot(xx_list, yy_list, labels=ll_list)

if __name__ == '__main__':
    main()
