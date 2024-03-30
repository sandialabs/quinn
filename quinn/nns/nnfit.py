#!/usr/bin/env python

import sys
import copy
import torch
import matplotlib.pyplot as plt

from .tchutils import tch
from .losses import NegLogPost, GradLoss


def nnfit(nnmodel, xtrn, ytrn, val=None,
          loss_fn='mse', loss_xy=None, datanoise=None, wd=0.0,
          priorparams=None,
          optimizer='adam', lrate=0.1, lmbd=None,
          nepochs=5000, batch_size=None,
          gradcheck=False,
          scheduler_lr=None,
          cooldown=100,
          factor=0.95,
          freq_out=100, freq_plot=1000, lhist_suffix=''
          ):
    """Generic PyTorch NN fit function that is utilized in appropriate NN classes.

    Args:
        nnmodel (torch.nn.Module): The PyTorch NN module of interest.
        xtrn (np.ndarray): Training input array `x` of size `(N,d)`.
        ytrn (np.ndarray): Training output array `y` of size `(N,o)`.
        val (tuple, optional): `x,y` tuple of validation points. Default uses the training set for validation.
        loss_fn (str, optional): Loss function string identifier. Currently only 'mse' is implemented and it is the default. Only used if the next argument, loss_xy=None.
        loss_xy (None, optional): Optionally, a more flexible loss function (e.g. used in variational inference) with signature :math:`\textrm{loss}(x_{pred}, y_{target})`. The default is None, which triggers the use of previous argument, loss_fn.
        datanoise (None, optional): Datanoise for certain loss types.
        wd (float, optional): Optional weight decay (L2 regularization) parameter.
        optimizer (str, optional): Optimizer string. Currently implemented 'adam' (default) and 'sgd'.
        lrate (float, optional): Learning rate or learning rate schedule factor. Default is 0.1.
        lmbd (callable, optional): Optional learning rate schedule. The actual learning rate is `lrate * lmbd(epoch)`.
        nepochs (int, optional): Number of epochs.
        batch_size (int, optional): Batch size. Default is None, i.e. single batch.
        gradcheck (bool, optional): For code verification, whether we want to check the auto-computed gradients against numerically computed ones. Makes the code slow. Experimental - this is not tested enough.
        scheduler_lr (None, optional): Description
        cooldown (int, optional): cooldown in ReduceLROnPlateau
        factor (float, optional): factor in ReduceLROnPlateau
        freq_out (int, optional): Frequency, in epochs, of screen output. Defaults to 100.
        freq_plot (int, optional): Frequency, in epochs, of plotting loss convergence graph. Defaults to 1000.
        lhist_suffix (str, optional): Optional uffix of loss history figure filename.

    Returns:
        dict: Dictionary of the results. Keys 'best_fepoch', 'best_epoch', 'best_loss', 'best_nnmodel', 'history'.

    Deleted Parameters:
        scheduler_lr(str, optional): Learning rate is adjusted during training according to the ReduceLROnPlateau method from pytTorch.
    """

    ntrn = xtrn.shape[0]

    # Loss function
    if loss_xy is None:
        if loss_fn == 'mse':
            loss = torch.nn.MSELoss(reduction='mean')
            def loss_xy(x, y):
                return loss(nnmodel(x), y)
        elif loss_fn == 'logpost':
            loss_xy = NegLogPost(nnmodel, ntrn, datanoise, priorparams)
        else:
            print(f"Loss function {loss_fn} is unknown. Exiting.")
            sys.exit()


    # Optimizer selection
    if optimizer == 'adam':
        opt = torch.optim.Adam(nnmodel.parameters(), lr=lrate, weight_decay=wd)
    elif optimizer == 'sgd':
        opt = torch.optim.SGD(nnmodel.parameters(), lr=lrate, weight_decay=wd)
    else:
        print(f"Optimizer {optimizer} is unknown. Exiting.")
        sys.exit()

    # Learning rate schedule
    if scheduler_lr == "ReduceLROnPlateau" and not lmbd is None:
            print(f"Trying to use two schedulers. Exiting.")
            sys.exit()    

    if lmbd is None:
        def lmbd(epoch): return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lmbd)

    if scheduler_lr == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', cooldown=cooldown, factor=factor, verbose=False)



    if batch_size is None or batch_size > ntrn:
        batch_size = ntrn

    try:
        device = nnmodel.device
    except AttributeError:
        device = 'cpu'
    
    xtrn_ = tch(xtrn , device=device)
    ytrn_ = tch(ytrn , device=device)

    # Validation data
    if val is None:
        xval, yval = xtrn.copy(), ytrn.copy()
    else:
        xval, yval = val

    xval_ = tch(xval , device=device)
    yval_ = tch(yval , device=device)
    
    # print("device: ", device)
    # print("xval_ is device: ", xval_.get_device(), xval_.device)
    # Training process
    fit_info = {'best_fepoch': 0, 'best_epoch': 0,
                'best_loss': 1.e+100, 'best_nnmodel': nnmodel,
                'history': []}


    fepoch = 0
    for t in range(nepochs):
        permutation = torch.randperm(ntrn)
        # for parameter in model.parameters():
        #     print(parameter)
        nsubepochs = len(range(0, ntrn, batch_size))
        for i in range(0, ntrn, batch_size):
            indices = permutation[i:i + batch_size]

            loss_trn = loss_xy(xtrn_[indices, :], ytrn_[indices, :])
            #loss_val = loss_trn
            with torch.no_grad():
                loss_val = loss_xy(xval_, yval_)
            #loss_trn_full = loss_trn
            if i == 0:  # otherwise too expensive
                with torch.no_grad():
                    loss_trn_full = loss_xy(xtrn_, ytrn_)

            fepoch += 1. / nsubepochs

            curr_state = [fepoch + 0.0, loss_trn.item(), loss_trn_full.item(), loss_val.item()]
            crit = loss_val.item()

            fit_info['history'].append(curr_state)

            if crit < fit_info['best_loss']:
                fit_info['best_loss'] = crit
                fit_info['best_nnmodel'] = copy.copy(nnmodel)

                fit_info['best_fepoch'] = fepoch
                fit_info['best_epoch'] = t


            if gradcheck:
                gc = torch.autograd.gradcheck(nnmodel, (xtrn_,),
                                              eps=1e-2, atol=1e-2)

            opt.zero_grad()
            loss_trn.backward()

            opt.step()
        if scheduler_lr == "ReduceLROnPlateau":
            ## using ValLoss as metric
            scheduler.step(curr_state[3])
        else:
            scheduler.step()

        ## Printout to screen
        if t == 0:
            print('{:>10} {:>10} {:>12} {:>12} {:>12} {:>18} {:>10}'.\
                  format("NEpochs", "NUpdates",
                         "BatchLoss", "TrnLoss", "ValLoss",
                         "BestLoss (Epoch)", "LrnRate"), flush=True)

        if (t + 1) % freq_out == 0 or t == 0 or t == nepochs - 1:
            tlr = opt.param_groups[0]['lr']
            printout = f"{t+1:>10}" \
                  f"{len(fit_info['history']):>10}" \
                  f"{fit_info['history'][-1][1]:>14.6f}" \
                  f"{fit_info['history'][-1][2]:>13.6f}" \
                  f"{fit_info['history'][-1][3]:>13.6f}" \
                  f"{fit_info['best_loss']:>14.6f} ({fit_info['best_epoch']})" \
                  f"{tlr:>10}"
            print(printout, flush=True)

        ## Plotting
        if t % freq_plot == 0 or t == nepochs - 1:
            fepochs = [state[0] for state in fit_info['history']]
            losses_trn = [state[1] for state in fit_info['history']]
            losses_trn_full = [state[2] for state in fit_info['history']]
            losses_val = [state[3] for state in fit_info['history']]

            _ = plt.figure(figsize=(12, 8))

            plt.plot(fepochs, losses_trn, label='Batch loss')
            plt.plot(fepochs, losses_trn_full, label='Training loss')
            plt.plot(fit_info['best_fepoch'], fit_info['best_loss'],
                     'ro', markersize=11)
            plt.vlines(fit_info['best_fepoch'], 0.0, 2.0,
                       colors=None, linestyles='--')
            plt.plot(fepochs, losses_val, label='Validation loss')

            plt.legend()

            plt.savefig(f'loss_history{lhist_suffix}.png')
            plt.yscale('log')
            plt.savefig(f'loss_history{lhist_suffix}_log.png')
            plt.clf()

        # ## Stop if very accurate (did not work for variational inference losses, so comment out)
        # if fit_info['best_loss'] < 1.e-10:
        #     print("AAAA ", fit_info['best_loss'])
        #     break

    return fit_info
