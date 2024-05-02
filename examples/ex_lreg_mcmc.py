#!/usr/bin/env python
"""An example of linear regression via MCMC."""

import torch
import numpy as np

from quinn.func.funcs import Sine
from quinn.utils.maps import scale01ToDom
from quinn.solvers.nn_mcmc import NN_MCMC
from quinn.utils.plotting import myrc
from quinn.utils.plotting import plot_xrv, plot_yx
from quinn.utils.plotting import plot_tri, plot_pdfs


def main():
    """Main function."""
    torch.set_default_dtype(torch.double)
    myrc()

    #################################################################################
    #################################################################################

    # defaults to cuda:0
    device='cpu'
    print("Using device",device)


    nall = 12 # total number of points
    trn_factor = 0.9 # which fraction of nall goes to training
    ntst = 13 # separate test set
    ndim = 1 # input dimensionality
    datanoise = 0.1 # Noise in the generated data

    # One output example
    true_model, nout = Sine, 1

    # Function domain
    domain = np.tile(np.array([-2., 2.]), (ndim, 1))

    # Get x data
    xall = scale01ToDom(np.random.rand(nall, ndim), domain)
    if true_model is not None:
        yall = true_model(xall, datanoise=datanoise)

    # Sample test
    if ntst > 0:
        np.random.seed(100)
        xtst = scale01ToDom(np.random.rand(ntst, ndim), domain)
        if true_model is not None:
            ytst = true_model(xtst, datanoise=datanoise)

    # Neural net is a linear function
    nnet = torch.nn.Linear(1,1, bias=True)
    ## Polynomial example
    #nnet = Polynomial(4)



    # Data split to training and validation
    ntrn = int(trn_factor * nall)
    indperm = range(nall)# np.random.permutation(range(nall))
    indtrn = indperm[:ntrn]
    indval = indperm[ntrn:]
    xtrn, xval = xall[indtrn, :], xall[indval, :]
    ytrn, yval = yall[indtrn, :], yall[indval, :]


    nmcmc = 10000
    uqnet = NN_MCMC(nnet, verbose=True)
    #sampler, sampler_params = 'hmc', {'L': 3, 'epsilon': 0.0025}
    sampler, sampler_params = 'amcmc', {'gamma': 0.1}
    uqnet.fit(xtrn, ytrn, zflag=False, datanoise=datanoise, nmcmc=nmcmc, sampler=sampler, sampler_params=sampler_params)


    # Prepare lists of inputs and outputs for plotting
    xx_list = [xtrn, xval]
    yy_list = [ytrn, yval]
    ll_list = ['Training', 'Validation']
    if ntst > 0:
        xx_list.append(xtst)
        yy_list.append(ytst)
        ll_list.append('Testing')

    uqnet.plot_1d_fits(xx_list, yy_list, nmc=100, labels=ll_list, true_model=true_model, name_postfix='mcmc')
    uqnet.predict_plot(xx_list, yy_list, nmc=100, plot_qt=False, labels=ll_list)
    np.savetxt('chain.txt', uqnet.samples)
    plot_xrv(uqnet.samples, prefix='chain')
    plot_yx(np.arange(uqnet.samples.shape[0])[:,np.newaxis],
            uqnet.samples,
            rowcols=(1,1), ylabel='', xlabels='Chain Id',
            log=False, filename='chain.png',
            xpad=0.3, ypad=0.3, gridshow=False, ms=4, labelsize=18)
    plot_tri(uqnet.samples, names=None, msize=3, figname='chain_tri.png')
    plot_pdfs(plot_type='tri', pdf_type='kde',
              samples_=uqnet.samples, burnin=nmcmc//10, every=10,
              names_=None, nominal_=None, prange_=None,
              show_2dsamples=True,
              lsize=13, zsize=13, xpad=0.3, ypad=0.3)

if __name__ == '__main__':
    main()
