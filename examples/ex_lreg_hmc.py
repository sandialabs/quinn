#!/usr/bin/env python

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from quinn.mcmc.mcmc import MCMC_NN
from quinn.mcmc.hmc import HMC_NN
from quinn.nns.nnwrap import NNWrap_Torch
from quinn.nns.posterior_funcs import (
    Gaussian_likelihood_assumed_var,
    Gaussian_prior,
    Log_Posterior,
    NegLogPosterior,
)

from quinn.func.funcs import Sine

from quinn.utils.plotting import myrc, plot_xrv, plot_yx, plot_tri, plot_pdfs
from quinn.utils.maps import scale01ToDom


def main():
    """Summary"""
    torch.set_default_dtype(torch.double)
    myrc()

    #################################################################################
    #################################################################################

    # defaults to cuda:0
    device = "cpu"
    print("Using device", device)

    nall = 12  # total number of points
    trn_factor = 0.9  # which fraction of nall goes to training
    ntst = 13  # separate test set
    ndim = 1  # input dimensionality
    datanoise = 0.2  # Noise in the generated data

    # Plot 1d fits or not
    plot_1d = ndim == 1
    # Plot quantiles or st.dev.
    plot_qt = False

    # One output example
    true_model, nout = Sine, 1

    # Function domain
    domain = np.tile(np.array([-2.0, 2.0]), (ndim, 1))

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
    nnet = torch.nn.Linear(1, 1, bias=True)
    param_dim = sum(p.numel() for p in nnet.parameters())
    # The model is defined in a wrapper class.
    NNmodel = NNWrap_Torch(nnet)

    # Define likelihood and prior modules

    likelihood = Gaussian_likelihood_assumed_var(sigma=datanoise)
    prior = Gaussian_prior(sigma=10, n_params=param_dim)
    log_posterior = Log_Posterior(likelihood, prior)
    u_hmc = NegLogPosterior(likelihood, prior)

    # Define HMC sampler

    sampling_params = {"epsilon": 0.05, "L_steps": 3}
    nmcmc = 3000
    sampler_hmc = HMC_NN(u_hmc, sampling_params, nmcmc=nmcmc, nburning=0)

    # Data split to training and validation
    ntrn = int(trn_factor * nall)
    indperm = range(nall)  # np.random.permutation(range(nall))
    indtrn = indperm[:ntrn]
    indval = indperm[ntrn:]
    xtrn, xval = xall[indtrn, :], xall[indval, :]
    ytrn, yval = yall[indtrn, :], yall[indval, :]

    uqnet = MCMC_NN(NNmodel, verbose=True, sampler=sampler_hmc, log_post=log_posterior)
    uqnet.fit(xtrn, ytrn, zflag=True, datanoise=datanoise)

    # Prepare lists of inputs and outputs for plotting
    xx_list = [xtrn, xval]
    yy_list = [ytrn, yval]
    ll_list = ["Training", "Validation"]
    if ntst > 0:
        xx_list.append(xtst)
        yy_list.append(ytst)
        ll_list.append("Testing")

    uqnet.plot_1d_fits(xx_list, yy_list, nmc=100, labels=ll_list, true_model=true_model)
    uqnet.predict_plot(xx_list, yy_list, nmc=100, plot_qt=False, labels=ll_list)
    np.savetxt("chain.txt", uqnet.samples)
    plot_xrv(uqnet.samples, prefix="chain")
    plot_yx(
        np.arange(uqnet.samples.shape[0])[:, np.newaxis],
        uqnet.samples,
        rowcols=(1, 1),
        ylabel="",
        xlabels="Chain Id",
        log=False,
        filename="chain.png",
        xpad=0.3,
        ypad=0.3,
        gridshow=False,
        ms=4,
        labelsize=18,
    )
    plot_tri(uqnet.samples, names=None, msize=3, figname="chain_tri.png")
    plot_pdfs(
        plot_type="tri",
        pdf_type="kde",
        samples_=uqnet.samples,
        burnin=nmcmc // 10,
        every=10,
        names_=None,
        nominal_=None,
        prange_=None,
        show_2dsamples=True,
        lsize=13,
        zsize=13,
        xpad=0.3,
        ypad=0.3,
    )


if __name__ == "__main__":
    main()
