#!/usr/bin/env python
"""An example of running all the available UQ-for-NN methods."""

import sys
import torch
import numpy as np

from quinn.solvers.nn_vi import NN_VI
from quinn.solvers.nn_ens import NN_Ens
from quinn.solvers.nn_rms import NN_RMS
from quinn.solvers.nn_mcmc import NN_MCMC
from quinn.solvers.nn_swag import NN_SWAG
from quinn.solvers.nn_laplace import NN_Laplace


from quinn.nns.rnet import RNet, Poly
from quinn.utils.plotting import myrc
from quinn.utils.maps import scale01ToDom
from quinn.func.funcs import Sine, Sine10, blundell


def main():
    """Main function."""
    torch.set_default_dtype(torch.double)
    myrc()

    #################################################################################
    #################################################################################
    meth = sys.argv[1]
    all_uq_options = ['amcmc', 'hmc', 'vi', 'ens', 'rms', 'laplace', 'swag']
    assert meth in all_uq_options, f'Pick among {all_uq_options}'

    # defaults to cuda:0 if available
    device_id='cuda:0'
    device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
    print("Using device",device)


    nall = 15 # total number of points
    trn_factor = 0.9 # which fraction of nall goes to training
    ntst = 13 # separate test set
    ndim = 1 # input dimensionality
    datanoise = 0.02 # Noise in the generated data

    # Plot 1d fits or not
    plot_1d = (ndim==1)
    # Plot quantiles or st.dev.
    plot_qt = False

    # One output example
    true_model, nout = Sine, 1
    # 10 output example
    # true_model, nout = Sine10, 10

    # Function domain
    domain = np.tile(np.array([-np.pi, np.pi]), (ndim, 1))
    #np.random.seed(111)

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
    #nnet = TwoLayerNet(1, 4, 1) #Constant() #MLP_simple((ndim, 5, 5, 5, nout)) #Polynomial(4) #Polynomial3() #TwoLayerNet(1, 4, 1)  #torch.nn.Linear(1,1, bias=False)
    nnet = RNet(3, 3, wp_function=Poly(0),
                indim=ndim, outdim=nout,
                layer_pre=True, layer_post=True,
                biasorno=True, nonlin=True,
                mlp=False, final_layer=None,
                device=device)
    # nnet = Polynomial(4)

    # nnet = MLP(ndim, nout, (11,11,11), biasorno=True,
    #                  activ='relu', bnorm=False, bnlearn=True, dropout=0.0)



    # Data split to training and validation
    ntrn = int(trn_factor * nall)
    indperm = range(nall)# np.random.permutation(range(nall))
    indtrn = indperm[:ntrn]
    indval = indperm[ntrn:]
    xtrn, xval = xall[indtrn, :], xall[indval, :]
    ytrn, yval = yall[indtrn, :], yall[indval, :]




    if meth == 'amcmc':
        nmc = 100
        uqnet = NN_MCMC(nnet, verbose=True)
        sampler_params = {'gamma': 0.01}
        uqnet.fit(xtrn, ytrn, zflag=False, datanoise=datanoise, nmcmc=10000, sampler='amcmc', sampler_params=sampler_params)
    elif meth == 'hmc':
        nmc = 100
        uqnet = NN_MCMC(nnet, verbose=True)
        sampler_params = {'L': 3, 'epsilon': 0.0025}
        uqnet.fit(xtrn, ytrn, zflag=False, datanoise=datanoise, nmcmc=10000, sampler='hmc', sampler_params=sampler_params)
    elif meth == 'vi':
        nmc = 111
        uqnet = NN_VI(nnet, verbose=True)
        uqnet.fit(xtrn, ytrn, val=[xval, yval], datanoise=datanoise, lrate=0.01, batch_size=None, nsam=1, nepochs=5000)
    elif meth == 'ens':
        nmc = 3
        uqnet = NN_Ens(nnet, nens=nmc, dfrac=0.8, verbose=True)
        uqnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=2, nepochs=1000)
    elif meth == 'rms':
        nmc = 7
        uqnet = NN_RMS(nnet, nens=nmc, dfrac=1.0, verbose=True, datanoise=datanoise, priorsigma=0.1)
        uqnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=2, nepochs=1000)
    elif meth == 'laplace':
        nmc = 3
        uqnet = NN_Laplace(nnet, nens=nmc, dfrac=1.0, verbose=True, la_type='full')
        uqnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=2, nepochs=1000)
    elif meth == 'swag':
        nmc = 3
        uqnet = NN_SWAG(nnet, nens=nmc, dfrac=1.0, verbose=True, k=10,
            n_steps=12, c=1, cov_type="lowrank", lr_swag=0.01)
        uqnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=2, nepochs=1000)
    else:
        print(f"UQ Method {meth} is unknown. Exiting.")
        sys.exit()


    # Prepare lists of inputs and outputs for plotting
    xx_list = [xtrn, xval]
    yy_list = [ytrn, yval]
    ll_list = ['Training', 'Validation']
    if ntst > 0:
        xx_list.append(xtst)
        yy_list.append(ytst)
        ll_list.append('Testing')

    if plot_1d:
        uqnet.plot_1d_fits(xx_list, yy_list, nmc=100, labels=ll_list, true_model=true_model, name_postfix=str(meth))
    uqnet.predict_plot(xx_list, yy_list, nmc=100, plot_qt=False, labels=ll_list)

if __name__ == '__main__':
    main()
