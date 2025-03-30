#!/usr/bin/env python

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from quinn.nns.nnwrap import NNWrap, nn_p, SNet
from quinn.nns.tchutils import npy, tch
from quinn.nns.rnet import RNet

from quinn.utils.stats import get_domain
from quinn.utils.plotting import myrc, plot_1d_anchored, plot_2d_anchored, plot_2d_anchored_single, plot_1d_anchored_single
from quinn.utils.maps import standardize, scaleDomTo01


torch.set_default_dtype(torch.double)

#####################################################################
#####################################################################
#####################################################################


def nnloss(p, modelpars):
    nnmodel, loss_fn, xtrn, ytrn = modelpars

    npt = p.shape[0]
    fval = np.empty((npt,))
    for ip, pp in enumerate(p):
        fval[ip] = loss_fn(tch(nn_p(pp, xtrn, nnmodel).reshape(ytrn.shape)), tch(ytrn)).item()

    # print(fval)

    return fval

#####################################################################
#####################################################################
#####################################################################


myrc()

#####################################################################
#####################################################################
#####################################################################

usage_str = 'Script to plot 2d slices of loss function.'
parser = argparse.ArgumentParser(description=usage_str)
# parser.add_argument('ind_show', type=int, nargs='*',
#                    help="indices of requested parameters (count from 0)")
parser.add_argument("-x", "--xdata", dest="xdata", type=str, default='ptrain.txt',
                    help="Xdata file")
parser.add_argument("-y", "--ydata", dest="ydata", type=str, default='ytrain.txt',
                    help="Ydata file")
parser.add_argument("-v", "--valfactor", dest="valfactor", type=float, default=0.1,
                    help="Factor of data used for validation")

args = parser.parse_args()

#####################################################################
#####################################################################
#####################################################################

scale = 10.0


#####################################################################
#####################################################################
#####################################################################

x = np.loadtxt(args.xdata)
domain = get_domain(x)
x = scaleDomTo01(x, domain)

y = np.loadtxt(args.ydata)
y = standardize(y)

valfactor = args.valfactor

if len(x.shape) == 1:
    x = x[:, np.newaxis]
if len(y.shape) == 1:
    y = y[:, np.newaxis]

nsam, ndim = x.shape
nsam_, nout = y.shape

assert(nsam == nsam_)


nval = int(valfactor * nsam)
ntrn = nsam - nval
print(f"Number of training points   : {ntrn}")
print(f"Number of validation points : {nval}")

rperm = np.random.permutation(nsam)
indtrn = rperm[:ntrn]
indval = rperm[ntrn:]

xtrn, ytrn = x[indtrn], y[indtrn]
xval, yval = x[indval], y[indval]

#####################################################################
#####################################################################
#####################################################################


#####################################################################
#####################################################################
#####################################################################

# hdl = (11,11,11)
# nnet = MLP(ndim, nout, hdl)

# nnet = torch.nn.Linear(ndim, nout)

nnet_orig = RNet(13, 3, wp_function=None,
                 indim=ndim, outdim=nout,
                 layer_pre=True, layer_post=True,
                 biasorno=True, nonlin=True,
                 mlp=True, final_layer=None)

# nnet = Polynomial(4)


loss_fcn = torch.nn.MSELoss(reduction='mean')

models = [nnloss, nnloss]
modelpars = [[nnet_orig, loss_fcn, xtrn, ytrn], [nnet_orig, loss_fcn, xval, yval]]

pdim = sum(p.numel() for p in nnet_orig.parameters())

ntry = 3
centers = np.empty((ntry, pdim))
for itry in range(ntry):
    nnet = RNet(13, 3, wp_function=None,
                indim=ndim, outdim=nout,
                layer_pre=True, layer_post=True,
                biasorno=True, nonlin=True,
                mlp=True, final_layer=None)

    snet = SNet(nnet, ndim, nout)
    def loss_xy(x, y): return loss_fcn(snet(x), y)

    snet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01,
             batch_size=None, nepochs=2000, loss_xy=loss_xy)

    nn_best = snet.best_model
    centers[itry] = npy(NNWrap(nn_best).p_flatten()).reshape(-1,)

labels = [f'Training (N={ntrn})', f'Validation (N={nval})']
print(centers.shape)
print(centers)

# Prepare lists of inputs and outputs for plotting
xx_list = [xtrn, xval]
yy_list = [ytrn, yval]
ll_list = ['Training', 'Validation']

if ndim == 1:
    snet.plot_1d_fits(xx_list, yy_list, labels=ll_list)
snet.predict_plot(xx_list, yy_list, labels=ll_list)

plot_1d_anchored(models, modelpars, centers[0], scale=scale,
                 ngr=111, modellabels=labels, ncolrow=(8, 5))
plt.clf()

plt.figure(figsize=(8, 8))
plot_1d_anchored_single(models, modelpars, centers[0],
                        anchor2=centers[1],
                        scale=scale, ngr=111, modellabels=labels,
                        figname='fcn_1d_allcenters_12.png')
plt.clf()

plt.figure(figsize=(8, 8))
plot_1d_anchored_single(models, modelpars, centers[0],
                        anchor2=centers[2],
                        scale=scale, ngr=111, modellabels=labels,
                        figname='fcn_1d_allcenters_13.png')
plt.clf()

plt.figure(figsize=(8, 8))
plot_1d_anchored_single(models, modelpars, centers[1],
                        anchor2=centers[2],
                        scale=scale, ngr=111, modellabels=labels,
                        figname='fcn_1d_allcenters_23.png')
plt.clf()

plt.figure(figsize=(10, 10))
plot_2d_anchored_single(models, modelpars, centers[0],
                        anchor2=centers[1], anchor3=centers[2],
                        scale=scale, ngr=55, squished=False, modellabels=labels,
                        figname='fcn_2d_allcenters.png')
plt.clf()

plot_2d_anchored(models, modelpars, centers[0], anchor2=centers[1],
                 scale=scale, ngr=55, squished=False,
                 modellabels=labels, ncolrow=(4, 3))

