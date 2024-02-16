#!/usr/bin/env python

import sys
import torch
import numpy as np

from quinn.mcmc.mcmc import MCMC_NN
from quinn.mcmc.admcmc import AMCMC
from quinn.mcmc.hmc import HMC_NN
from quinn.vi.vi import VI_NN
from quinn.ens.ens import Ens_NN
from quinn.ens.swag import SWAG_NN
from quinn.ens.laplace import LAPLACE_NN
from quinn.ens.rms import RMS_NN
from quinn.ens.dropout import Dropout_NN
from quinn.nns.posterior_funcs import (
    Gaussian_likelihood_assumed_var,
    Gaussian_prior,
    RMS_gaussian_prior,
    Log_Posterior_Assumed_Variance,
    NegLogPosterior_Assumed_Variance,
)
from quinn.nns.nnwrap import NNWrap_Torch, NNWrap

from quinn.func.funcs import Ackley, Sine, Sine10, blundell

from quinn.nns.mlp import MLP
from quinn.nns.losses import CustomLoss
from quinn.nns.nns import MLP_simple, Constant, Polynomial, Polynomial3, TwoLayerNet
from quinn.nns.rnet import RNet, Const, Lin, Quad, Cubic, NonPar, Poly
from quinn.nns.tchutils import print_nnparams

from quinn.utils.plotting import myrc
from quinn.utils.maps import scale01ToDom


def main():
    """Summary"""
    torch.set_default_dtype(torch.double)
    myrc()

    #################################################################################
    #################################################################################
    meth = sys.argv[1]  #'mcmc' #'vi' #'ens'

    # defaults to cuda:0
    device_id = "cuda:0"
    # use: ./ex_ufit uq_method device_id, where uq_method: 'mcmc' 'vi' 'ens', and
    # device_id: cuda:0, cuda:1,... depending on number of gpus
    if len(sys.argv) > 2:
        device_id = sys.argv[2]
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    nall = 12  # total number of points
    trn_factor = 0.9  # which fraction of nall goes to training
    ntst = 13  # separate test set
    ndim = 1  # input dimensionality
    datanoise = 0.02  # Noise in the generated data

    # Plot 1d fits or not
    plot_1d = ndim == 1
    # Plot quantiles or st.dev.
    plot_qt = False

    # One output example
    true_model, nout = Sine, 1
    # 10 output example
    # true_model, nout = Sine10, 10

    # Function domain
    domain = np.tile(np.array([-np.pi, np.pi]), (ndim, 1))
    np.random.seed(1)

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

    # Model to fit
    # nnet = TwoLayerNet(1, 4, 1) #Constant() #MLP_simple((ndim, 5, 5, 5, nout)) #Polynomial(4) #Polynomial3() #TwoLayerNet(1, 4, 1)  #torch.nn.Linear(1,1, bias=False)
    nnet = RNet(
        3,
        3,
        wp_function=Poly(0),
        indim=ndim,
        outdim=nout,
        layer_pre=True,
        layer_post=True,
        biasorno=True,
        nonlin=True,
        mlp=False,
        final_layer=None,
        device=device,
    )
    param_dim = sum(p.numel() for p in nnet.parameters())
    # nnet = Polynomial(4)

    # nnet = MLP(ndim, nout, (11,11,11), biasorno=True,
    #                  activ='relu', bnorm=False, bnlearn=True, dropout=0.0)

    # Data split to training and validation
    ntrn = int(trn_factor * nall)
    indperm = range(nall)  # np.random.permutation(range(nall))
    indtrn = indperm[:ntrn]
    indval = indperm[ntrn:]
    xtrn, xval = xall[indtrn, :], xall[indval, :]
    ytrn, yval = yall[indtrn, :], yall[indval, :]

    if meth == "mcmc":
        nmc = 100
        nnet = NNWrap_Torch(nnet)
        likelihood = Gaussian_likelihood_assumed_var(sigma=datanoise)
        prior = Gaussian_prior(sigma=100, n_params=param_dim)
        log_posterior = Log_Posterior_Assumed_Variance(likelihood, prior)
        amcmc = AMCMC()
        amcmc.setParams(log_posterior, cov_ini=None, nmcmc=20000, gamma=0.01)
        uqnet = MCMC_NN(nnet, verbose=True, sampler=amcmc, log_post=log_posterior)
        uqnet.fit(xtrn, ytrn, zflag=False, datanoise=datanoise)
    elif meth == "hmc":
        nmc = 100
        nnet = NNWrap_Torch(nnet)
        likelihood = Gaussian_likelihood_assumed_var(sigma=datanoise)
        prior = Gaussian_prior(sigma=100, n_params=param_dim)
        log_posterior = Log_Posterior_Assumed_Variance(likelihood, prior)
        u_hmc = NegLogPosterior_Assumed_Variance(likelihood, prior)
        sampling_params = {"epsilon": 0.0025, "L_steps": 3}
        hmc = HMC_NN(u_hmc, sampling_params, nmcmc=20000, nburning=0)
        uqnet = MCMC_NN(nnet, verbose=True, sampler=hmc, log_post=log_posterior)
        uqnet.fit(xtrn, ytrn, zflag=False, datanoise=datanoise)
    elif meth == "vi":
        nmc = 111
        uqnet = VI_NN(nnet, verbose=True)
        uqnet.fit(
            xtrn,
            ytrn,
            val=[xval, yval],
            datanoise=datanoise,
            lrate=0.01,
            batch_size=None,
            nsam=1,
            nepochs=5000,
        )
    elif meth == "ens":
        nmc = 10
        uqnet = Ens_NN(nnet, nens=nmc, dfrac=0.8, verbose=True)
        uqnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=2, nepochs=500)
    elif meth == "swag":
        nens = 4
        nnet = NNWrap_Torch(nnet)
        likelihood = Gaussian_likelihood_assumed_var(sigma=datanoise)
        prior = Gaussian_prior(sigma=3, n_params=param_dim)
        neg_logposterior = NegLogPosterior_Assumed_Variance(likelihood, prior)
        uqnet = SWAG_NN(
            nnet,
            neg_logposterior,
            param_dim,
            nens=nens,
            learn_rate_swag=2e-6,
            verbose=True,
            k=10,
            n_steps=12,
        )
        uqnet.fit_swag(
            xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=20, nepochs=500
        )

    elif meth == "la":
        nnet = NNWrap_Torch(nnet)
        likelihood = Gaussian_likelihood_assumed_var(sigma=datanoise)
        prior = Gaussian_prior(sigma=3, n_params=param_dim)
        neg_logposterior = NegLogPosterior_Assumed_Variance(likelihood, prior)
        uqnet = LAPLACE_NN(
            nnet,
            neg_logposterior,
            nens=4,
            dfrac=0.8,
            verbose=False,
            la_type="diag",
        )
        uqnet.fit_la(
            xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=5, nepochs=500
        )
    elif meth == "rms":
        nnet = NNWrap_Torch(nnet)
        likelihood = Gaussian_likelihood_assumed_var(sigma=datanoise)
        prior = RMS_gaussian_prior(sigma=3, n_params=param_dim)
        neg_logposterior = NegLogPosterior_Assumed_Variance(likelihood, prior)
        uqnet = RMS_NN(
            nnet,
            neg_logposterior,
            nens=10,
            dfrac=0.8,
            verbose=False,
        )
        uqnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=5, nepochs=1000)
    elif meth == "dropout":
        nnet = RNet(
            100,
            5,
            wp_function=Poly(0),
            indim=ndim,
            outdim=nout,
            layer_pre=True,
            layer_post=True,
            device=device,
            dropout=0.02,
        )
        nmc = 1
        likelihood = Gaussian_likelihood_assumed_var(sigma=datanoise)
        prior = Gaussian_prior(sigma=2, n_params=param_dim)
        neg_logposterior = NegLogPosterior_Assumed_Variance(likelihood, prior)
        uqnet = Dropout_NN(nnet, nens=nmc, dfrac=0.8, verbose=True)
        uqnet.fit(
            xtrn,
            ytrn,
            val=[xval, yval],
            lrate=0.01,
            batch_size=2,
            nepochs=1000,
            loss_fn="logposterior",
            loss=neg_logposterior,
        )
    else:
        print(f"UQ Method {meth} is unknown. Exiting.")
        sys.exit()

    # Prepare lists of inputs and outputs for plotting
    xx_list = [xtrn, xval]
    yy_list = [ytrn, yval]
    ll_list = ["Training", "Validation"]
    if ntst > 0:
        xx_list.append(xtst)
        yy_list.append(ytst)
        ll_list.append("Testing")

    if plot_1d:
        uqnet.plot_1d_fits(
            xx_list,
            yy_list,
            nmc=100,
            labels=ll_list,
            true_model=true_model,
            name_postfix=str(meth),
        )
    uqnet.predict_plot(xx_list, yy_list, nmc=100, plot_qt=False, labels=ll_list)


if __name__ == "__main__":
    main()
