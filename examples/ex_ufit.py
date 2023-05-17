#!/usr/bin/env python

import sys
import torch
import numpy as np

from quinn.mcmc.mcmc import MCMC_NN
from quinn.vi.vi import VI_NN
from quinn.ens.ens import Ens_NN

from quinn.func.funcs import Ackley, Sine, Sine10, blundell

from quinn.nns.mlp import MLP
from quinn.nns.losses import CustomLoss
from quinn.nns.nns import MLP_simple, Constant, Polynomial,Polynomial3, TwoLayerNet
from quinn.nns.rnet import RNet, Const, Lin, Quad, Cubic, NonPar, Poly
from quinn.nns.tchutils import print_nnparams

from quinn.utils.plotting import myrc
from quinn.utils.maps import scale01ToDom


def main():
    """Summary
    """
    torch.set_default_dtype(torch.double)
    myrc()

    #################################################################################
    #################################################################################
    meth = sys.argv[1] #'mcmc' #'vi' #'ens'

    # defaults to cuda:0
    device_id='cuda:0'
    # use: ./ex_ufit uq_method device_id, where uq_method: 'mcmc' 'vi' 'ens', and 
    # device_id: cuda:0, cuda:1,... depending on number of gpus
    if len(sys.argv) > 2:
        device_id=sys.argv[2]
    device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
    print("Using device",device)


    nall = 12 # total number of points
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
    #nnet = TwoLayerNet(1, 4, 1) #Constant() #MLP_simple((ndim, 5, 5, 5, nout)) #Polynomial(4) #Polynomial3() #TwoLayerNet(1, 4, 1)  #torch.nn.Linear(1,1, bias=False)
    nnet = RNet(3, 3, wp_function=Poly(0),
                indim=ndim, outdim=nout,
                layer_pre=True, layer_post=True,
                biasorno=True, nonlin=True,
                mlp=False, final_layer=None,
                device=device)

    # nnet = MLP(ndim, nout, (11,11,11), biasorno=True,
    #                  activ='relu', bnorm=False, bnlearn=True, dropout=0.0)



    # Data split to training and validation
    ntrn = int(trn_factor * nall)
    indperm = np.random.permutation(range(nall))
    indtrn = indperm[:ntrn]
    indval = indperm[ntrn:]
    xtrn, xval = xall[indtrn, :], xall[indval, :]
    ytrn, yval = yall[indtrn, :], yall[indval, :]


    if meth == 'mcmc':
        nmc = 100
        uqnet = MCMC_NN(nnet, verbose=True)
        uqnet.fit(xtrn, ytrn, zflag=False, datanoise=datanoise, gamma=0.01, nmcmc=10000)
    elif meth == 'vi':
        nmc = 111
        uqnet = VI_NN(nnet, verbose=True)
        uqnet.fit(xtrn, ytrn, val=[xval, yval], datanoise=datanoise, lrate=0.01, batch_size=None, nsam=1, nepochs=5000)
    elif meth == 'ens':
        nmc = 3
        uqnet = Ens_NN(nnet, nens=nmc, dfrac=0.8, verbose=True)
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
