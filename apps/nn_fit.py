#!/usr/bin/env python
"""Command line utility for NN fit."""
import sys
import torch
import argparse
import numpy as np


from quinn.solvers.nn_vi import NN_VI
from quinn.solvers.nn_ens import NN_Ens
from quinn.solvers.nn_rms import NN_RMS
from quinn.solvers.nn_mcmc import NN_MCMC
from quinn.solvers.nn_swag import NN_SWAG
from quinn.solvers.nn_laplace import NN_Laplace

from quinn.utils.plotting import myrc
from quinn.utils.stats import get_domain
from quinn.utils.maps import scale01ToDom
from quinn.utils.xutils import read_textlist

from quinn.nns.rnet import RNet, Poly

torch.set_default_dtype(torch.double)

myrc()


usage_str = 'A command line app to build NN surrogate given x-y training dataset.'
parser = argparse.ArgumentParser(description=usage_str)
#parser.add_argument('ind_show', type=int, nargs='*',
#                    help="indices of requested parameters (count from 0)")
parser.add_argument("-x", "--xdata", dest="xdata", type=str, default='ptrain.txt',
                    help="Xdata file")
parser.add_argument("-y", "--ydata", dest="ydata", type=str, default='ytrain.txt',
                    help="Ydata file")
# parser.add_argument("-r", "--xrange", dest="xrange", type=str, default=None,
#                     help="Xrange file")
parser.add_argument("-q", "--outnames_file", dest="outnames_file", type=str, default='outnames.txt',
                    help="Output names file")
parser.add_argument("-p", "--pnames_file", dest="pnames_file", type=str, default='pnames.txt',                    help="Param names file")
parser.add_argument("-m", "--method", dest="method", type=str, default='ens',                    help="Method")
parser.add_argument("-t", "--trnfactor", dest="trnfactor", type=float, default=0.8,
                    help="Factor of data used for training")
parser.add_argument("-v", "--valfactor", dest="valfactor", type=float, default=0.1,
                    help="Factor of data used for validation")

args = parser.parse_args()

method = args.method
all_uq_options = ['amcmc', 'hmc', 'vi', 'ens', 'rms', 'laplace', 'swag']
assert method in all_uq_options, f'Pick among {all_uq_options}'

trnfactor = args.trnfactor
valfactor = args.valfactor
assert(trnfactor+valfactor<=1.0)

x = np.loadtxt(args.xdata)
y = np.loadtxt(args.ydata)



if len(x.shape)==1:
    x = x[:, np.newaxis]
if len(y.shape)==1:
    y = y[:, np.newaxis]

nsam, ndim = x.shape
nsam_, nout = y.shape

assert(nsam == nsam_)

outnames = read_textlist(args.outnames_file, nout, names_prefix='out')
pnames = read_textlist(args.pnames_file, ndim, names_prefix='par')

ntrn = int(trnfactor * nsam)
nval = int(valfactor * nsam)
ntst = nsam - ntrn - nval
assert(ntst>=0)
print(f"Number of training points   : {ntrn}")
print(f"Number of validation points : {nval}")
print(f"Number of testing points    : {ntst}")

rperm = np.random.permutation(nsam)
indtrn = rperm[:ntrn]
indval = rperm[ntrn:ntrn+nval]
indtst = rperm[ntrn+nval:]

################################################################################
################################################################################


# Plot quantiles or st.dev.
plot_qt = False


# Function domain
domain = get_domain(x)

# Get x data
xsc = scale01ToDom(x, domain)


xtrn, ytrn = xsc[indtrn], y[indtrn]
xval, yval = xsc[indval], y[indval]
xtst, ytst = xsc[indtst], y[indtst]


# Model to fit
#nnet = TwoLayerNet(1, 4, 1) #Constant() #MLP_simple((ndim, 5, 5, 5, nout)) #Polynomial(4) #Polynomial3() #TwoLayerNet(1, 4, 1)  #torch.nn.Linear(1,1, bias=False)
nnet = RNet(3, 3, wp_function=Poly(0),
            indim=ndim, outdim=nout,
            layer_pre=True, layer_post=True,
            biasorno=True, nonlin=True,
            mlp=False, final_layer=None)

# nnet = MLP(ndim, nout, (11,11,11), biasorno=True,
#                  activ='relu', bnorm=False, bnlearn=True, dropout=0.0)


if method == 'amcmc':
    nmc = 100
    uqnet = NN_MCMC(nnet, verbose=True)
    sampler_params = {'gamma': 0.01}
    uqnet.fit(xtrn, ytrn, zflag=False, datanoise=0.01, nmcmc=10000, sampler='amcmc', sampler_params=sampler_params)
elif method == 'hmc':
    nmc = 100
    uqnet = NN_MCMC(nnet, verbose=True)
    sampler_params = {'L': 3, 'epsilon': 0.0001}
    uqnet.fit(xtrn, ytrn, zflag=False, datanoise=0.01, nmcmc=10000, sampler='hmc', sampler_params=sampler_params)
elif method == 'vi':
    nmc = 111
    uqnet = NN_VI(nnet, verbose=True)
    uqnet.fit(xtrn, ytrn, val=[xval, yval], datanoise=0.01, lrate=0.01, batch_size=None, nsam=1, nepochs=5000)
elif method == 'ens':
    nmc = 13
    uqnet = NN_Ens(nnet, nens=nmc, dfrac=0.8, verbose=True)
    uqnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=2, nepochs=1000)
elif method == 'rms':
    nmc = 7
    uqnet = NN_RMS(nnet, nens=nmc, dfrac=0.8, verbose=True, datanoise=0.1, priorsigma=0.1)
    uqnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=2, nepochs=1000)
elif method == 'laplace':
    nmc = 3
    uqnet = NN_Laplace(nnet, nens=nmc, dfrac=1.0, verbose=True, la_type='full')
    uqnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=2, nepochs=1000)
elif method == 'swag':
        nmc = 3
        uqnet = NN_SWAG(nnet, nens=nmc, dfrac=1.0, verbose=True, k=10, n_steps=12, c=1, cov_type="lowrank", lr_swag=0.01)
        uqnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, batch_size=2, nepochs=1000)
else:
    print(f"UQ Method {method} is unknown. Exiting.")
    sys.exit()


# Prepare lists of inputs and outputs for plotting
xx_list = [xtrn, xval]
yy_list = [ytrn, yval]
ll_list = ['Training', 'Validation']
if ntst > 0:
    xx_list.append(xtst)
    yy_list.append(ytst)
    ll_list.append('Testing')

if ndim==1:
    uqnet.plot_1d_fits(xx_list, yy_list, nmc=100, labels=ll_list, name_postfix=str(method))
uqnet.predict_plot(xx_list, yy_list, nmc=100, plot_qt=False, labels=ll_list)



## This can go inside uqnet and be improved
# for isam in range(nsam):
#     f = plt.figure(figsize=(12,4))
#     plt.plot(range(nout), y[isam,:], 'b-', label='Data')
#     plt.plot(range(nout), ypred[isam,:], 'g-', label='NN apprx.')
#     plt.title(f'Sample #{isam+1}')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'fit_s{str(isam).zfill(3)}.png')
#     plt.close()




