#!/usr/bin/env python
"""Module for the Variational Inference (VI) NN wrapper."""

import copy
import math
import torch

from .rvs import Gaussian, GMM2

from ..quinn import QUiNNBase
from ..nns.tchutils import npy, tch, print_nnparams, nnfit
from ..nns.nnwrap import nn_p, NNWrap

class VI_NN(QUiNNBase):
    """VI wrapper class.

    Attributes:
        bmodel (BNet): The underlying Bayesian model.
        trained (bool): Whether the model is trained or not.
        verbose (bool): Whether to be verbose or not.
        best_epoch (int): The epoch where the best model is reached.
        best_loss (float): The best loss value.
        best_model (torch.nn.Module): The best PyTorch NN model found during training.
    """

    def __init__(self, nnmodel, verbose=False, pi=0.5, sigma1=1.0, sigma2=1.0,
                       mu_init_lower=-0.2, mu_init_upper=0.2,
                       rho_init_lower=-5.0, rho_init_upper=-4.0 ):
        """Instantiate a VI wrapper object.

        Args:
            nnmodel (torch.nn.Module): The underlying PyTorch NN model.
            verbose (bool, optional): Whether to print out model details or not.
            pi (float): Weight of the first gaussian. The second weight is 1-pi.
            sigma1 (float): Standard deviation of the first gaussian. Can also be a scalar torch.Tensor.
            sigma2 (float): Standard deviation of the second gaussian. Can also be a scalar torch.Tensor.
            mu_init_lower (float) : Initialization of mu lower value
            mu_init_upper (float) : Initialization of mu upper value
            rho_init_lower (float) : Initialization of rho lower value
            rho_init_upper (float) : Initialization of rho upper value

        """
        super().__init__(nnmodel)

        self.bmodel = BNet(nnmodel,pi=pi,sigma1=sigma1,sigma2=sigma2,
                                   mu_init_lower=mu_init_lower, mu_init_upper=mu_init_upper,
                                   rho_init_lower=rho_init_lower, rho_init_upper=rho_init_upper )
        device = nnmodel.device
        self.bmodel.to(device)
        self.verbose = verbose

        if self.verbose:
            print("=========== Deterministic model parameters ================")
            self.print_params(names_only=True)
            print("=========== Variational model parameters ==================")
            print_nnparams(self.bmodel, names_only=True)
            print("===========================================================")

    def fit(self, xtrn, ytrn, val=None,
            nepochs=600, lrate=0.01, batch_size=None, freq_out=100,
            freq_plot=1000, wd=0,
            cooldown=100,
            factor=0.95,
            nsam=1,scheduler_lr=None, datanoise=0.05):
        """Fit function to train the network.

        Args:
            xtrn (np.ndarray): Training input array of size `(N,d)`.
            ytrn (np.ndarray): Training output array of size `(N,o)`.
            val (tuple, optional): `x,y` tuple of validation points. Default uses the training set for validation.
            nepochs (int, optional): Number of epochs.
            lrate (float, optional): Learning rate or learning rate schedule factor. Default is 0.01.
            batch_size (int, optional): Batch size. Default is None, i.e. single batch.
            freq_out (int, optional): Frequency, in epochs, of screen output. Defaults to 100.
            nsam (int, optional): Number of samples for ELBO computation. Defaults to 1.
            scheduler_lr(str,optional): Learning rate is adjusted during training according to the ReduceLROnPlateau method from pytTorch. 
            datanoise (float, optional): Datanoise for ELBO computation. Defaults to 0.05.
            freq_out (int, optional): Frequency, in epochs, of screen output. Defaults to 100.
            wd (float, optional): Optional weight decay (L2 regularization) parameter.
            cooldown (int, optional) : cooldown in ReduceLROnPlateau
            factor (float, optional) : factor in ReduceLROnPlateau
        """

        shape_xtrn = xtrn.shape
        ntrn = shape_xtrn[0]
        ntrn_, outdim = ytrn.shape
        assert(ntrn==ntrn_)

        if batch_size is None or batch_size > ntrn:
            batch_size = ntrn

        if batch_size == 1:
            num_batches = ntrn
        else:
            num_batches = (ntrn + 1) // batch_size

        self.bmodel.loss_params = [datanoise, nsam, num_batches]

        fit_info = nnfit(self.bmodel, xtrn, ytrn, val=val,
                         loss_xy=self.bmodel.viloss,
                         lrate=lrate, batch_size=batch_size,
                         nepochs=nepochs,
                         wd=wd,
                         cooldown=cooldown,
                         factor=factor,
                         freq_plot=freq_plot,
                         scheduler_lr=scheduler_lr, freq_out=freq_out)
        self.best_model = fit_info['best_nnmodel']
        self.trained = True


    # def fit_deprecated(self, xtrn, ytrn, datanoise=0.05, nepochs=600, lrate=0.01, batch_size=None, nsam=1, freq_out=100, Xtst=None, ytst=None):
    #     """Deprecated. Should be removed."""
    #     ntrn, ndim = xtrn.shape
    #     print("ytrn.shape = ", ytrn.shape)
    #     ntrn_, outdim = ytrn.shape
    #     if batch_size is None or batch_size > ntrn:
    #         batch_size = ntrn

    #     if batch_size == 1:
    #         num_batches = ntrn
    #     else:
    #         num_batches = (ntrn + 1) // batch_size

    #     optimizer = torch.optim.Adam(self.bmodel.parameters(), lr=lrate)

    #     self.best_loss = 1.e+100
    #     for t in range(nepochs):
    #         permutation = torch.randperm(ntrn)

    #         #net.train()

    #         for i in range(0, ntrn, batch_size):

    #             indices = permutation[i:i + batch_size]
    #             data = torch.tensor(xtrn[indices,:].reshape(-1, ndim), requires_grad=True)
    #             target = torch.tensor(ytrn[indices, :].reshape(-1, outdim), requires_grad=True)

    #             self.bmodel.zero_grad()
    #             #optimizer.zero_grad()

    #             log_prior, log_variational_posterior, negative_log_likelihood = self.bmodel.sample_elbo(data, target, nsam, likparams=[datanoise])

    #             loss = (log_variational_posterior - log_prior)/num_batches + negative_log_likelihood

    #             loss.backward()

    #             optimizer.step()


    #         if loss.item() < self.best_loss:
    #             self.best_loss = loss.item()
    #             self.best_model = copy.copy(self.bmodel)
    #             self.best_epoch = t


    #         if t == 0:
    #             print('{:>10} {:>10} {:>18} {:>10}'.\
    #                   format("NEpochs", "TrnLoss",
    #                          "BestLoss (Epoch)", "LrnRate"))

    #         if (t + 1) % freq_out == 0 or t == 0 or t == nepochs - 1:
    #             tlr = optimizer.param_groups[0]['lr']
    #             printout = f"{t:>10}" \
    #                   f"{loss.item():>12.6f}" \
    #                   f"{self.best_loss:>14.6f} ({self.best_epoch})" \
    #                   f"{tlr:>10}"
    #             print(printout, flush=True)

    def predict_sample(self, x):
        """Predict a single sample.

        Args:
            x (np.ndarray): Input array `x` of size `(N,d)`.

        Returns:
            np.ndarray: Output array `x` of size `(N,o)`.

        Note:
            predict_ens() from the parent class will use this to sample an ensemble.
        """
        assert(self.trained)
        device = self.best_model.device
        y = npy(self.best_model(tch(x, rgrad=False,device=device), sample=True))

        return y

######################################################################
######################################################################
######################################################################

class BNet(torch.nn.Module):
    """Bayesian NN class.

    Attributes:
        nnmodel (torch.nn.Module): The underlying PyTorch NN module.
        nparams (int): Number of deterministic parameters.
        param_names (list[str]): List of parameter names.
        param_priors (list[quinn.vi.rvs.RV]): List of parameter priors.
        rparams (list[quinn.vi.rvs.RV]): List of variational PDFs.
        params (torch.nn.ParameterList): Variational parameters.
        log_prior (float): Value of log-prior.
        log_variational_posterior (float): Value of logarithm of variational posterior.
    """

    def __init__(self, nnmodel, pi=0.5, sigma1=1.0, sigma2=1.0,
                                mu_init_lower=-0.2, mu_init_upper=0.2,
                                rho_init_lower=-5.0, rho_init_upper=-4.0  ):
        """Instantiate a Bayesian NN object given an underlying PyTorch NN module.

        Args:
            nnmodel (torch.nn.Module): The original PyTorch NN module.
            pi (float): Weight of the first gaussian. The second weight is 1-pi.
            sigma1 (float): Standard deviation of the first gaussian. Can also be a scalar torch.Tensor.
            sigma2 (float): Standard deviation of the second gaussian. Can also be a scalar torch.Tensor.
            mu_init_lower (float) : Initialization of mu lower value
            mu_init_upper (float) : Initialization of mu upper value
            rho_init_lower (float) : Initialization of rho lower value
            rho_init_upper (float) : Initialization of rho upper value
        """
        super().__init__()
        assert(isinstance(nnmodel, torch.nn.Module))

        self.nnmodel = copy.deepcopy(nnmodel)
        
        self.device = nnmodel.device

        # for name, param in self.nnmodel.named_parameters():
        #     print(name)
        #     if param.requires_grad:
        #         name_ = (name+'').replace('.', '_')
        #         self.del_attr(self.nnmodel, [name])
        #         self.nnmodel.register_parameter(name, torch.nn.Parameter(param.data))
        # print("######")
        # for name, param in self.nnmodel.named_parameters():
        #     print(name)
        # sys.exit()

        self.param_names = []
        self.rparams = []
        self.param_priors = []
        i=0
        for name, param in self.nnmodel.named_parameters():
            if param.requires_grad:

                #param.requires_grad = False
                mu = torch.nn.Parameter(torch.Tensor(param.shape).uniform_(mu_init_lower, mu_init_upper))
                self.register_parameter(name.replace('.', '_')+"_mu", mu)

                rho = torch.nn.Parameter(torch.Tensor(param.shape).uniform_(rho_init_lower, rho_init_upper))
                self.register_parameter(name.replace('.', '_')+"_rho", rho)

                if i==0:
                    self.params = torch.nn.ParameterList([mu, rho])
                else:
                    self.params.append(mu)
                    self.params.append(rho)
                self.rparams.append(Gaussian(mu, logsigma=rho))
                
                ## PRIOR
                self.param_priors.append(GMM2(pi, sigma1, sigma2))
                self.param_names.append(name)

            #     for i, param_name in enumerate(self.param_names):
            # #print("AAAA ", i, param_name)
            # self.set_attr(self.nnmodel,param_name.split("."), par_samples[i])

                i+=1

        self.log_prior = 0.0
        self.log_variational_posterior = 0.0
        self.nparams = len(self.rparams)

        #print("AAAAA ", self.param_names)
        for param_name in self.param_names:
            self.del_attr(self.nnmodel,param_name.split("."))


    # Inspired by this https://discuss.pytorch.org/t/how-does-one-have-the-parameters-of-a-model-not-be-leafs/70076/10
    # this could be better https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    def del_attr(self, obj, names):
        """Deletes attributes from a given object.

        Args:
            obj (any): The object of interest.
            names (list): List that corresponds to the attribute to be deleted. If list is ['A', 'B', 'C'], the attribute A.B.C is deleted recursively.
        """
        # print("Del ", names)
        if len(names) == 1:
            delattr(obj, names[0])
        else:
            self.del_attr(getattr(obj, names[0]), names[1:])

    def set_attr(self, obj, names, val):
        """Sets attributes of a given object.

        Args:
            obj (any): The object of interest.
            names (list): List that corresponds to the attribute of interest. If list is ['A', 'B', 'C'], the attribute A.B.C is filled with value val.
            val (torch.Tensor): Value to be set.
        """
        # print("Set ", names, val)
        if len(names) == 1:
            setattr(obj, names[0], val)
        else:
            self.set_attr(getattr(obj, names[0]), names[1:], val)


    def forward(self, x, sample=False, par_samples=None):
        """Forward function of Bayesian NN object.

        Args:
            x (torch.Tensor): Input array of size `(N,d)`.
            sample (bool, optional): Whether this is used in a sampling mode or not.
            par_samples (None, optional): Parameter samples. Default is None, in which cases the mean values of variational PDFs are used.

        Returns:
            torch.Tensor: Output array of size `(N,o)`.
        """
        if self.training or sample:
            assert par_samples is None
            par_samples = []
            for rpar in self.rparams:
                par_samples.append(rpar.sample())
        else:
            if par_samples is None:
                par_samples = []
                for rpar in self.rparams:
                    par_samples.append(rpar.mu)

        assert(len(par_samples)==self.nparams)


        if self.training:
            self.log_prior = 0.0
            for par_sample, param_prior in zip(par_samples, self.param_priors):
                self.log_prior += param_prior.log_prob(par_sample)

            self.log_variational_posterior = 0.0
            for par_sample, rpar in zip(par_samples, self.rparams):
                self.log_variational_posterior += rpar.log_prob(par_sample)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0


        for i, param_name in enumerate(self.param_names):
            #print("AAAA ", i, param_name, param_name.split("."))
            self.set_attr(self.nnmodel,param_name.split("."), par_samples[i])
            #print([i for i in self.nnmodel.coefs])
            #self.nnmodel.register_parameter(param_name.replace(".","_"), torch.nn.Parameter(par_samples[i]))
        #print("BBB ", list(self.nnmodel.parameters()))
        #print(dir(self))
        #print(par_samples)


        return self.nnmodel(x)


    def sample_elbo(self, x, target, nsam, likparams=None):
        """Sample from ELBO.

        Args:
            x (torch.Tensor): A 2d input tensor.
            target (torch.Tensor): A 2d output tensor.
            nsam (int): Number of samples
            likparams (tuple, optional): Other parameters of the likelihood, e.g. data noise.

        Returns:
            tuple: (log_prior, log_variational_posterior, negative_log_likelihood)
        """
        shape_x = x.shape
        batch_size = shape_x[0]
        batch_size_, outdim = target.shape
        assert(batch_size == batch_size_)
        # FIXME: 
        device = x.device
        outputs = torch.zeros(nsam, batch_size, outdim, device=device)
        log_priors = torch.zeros(nsam, device=device)
        log_variational_posteriors = torch.zeros(nsam, device=device)
        for i in range(nsam):
            outputs[i] = self(x, sample=True)
            log_priors[i] = self.log_prior
            log_variational_posteriors[i] = self.log_variational_posterior
        #print("AA ", outputs)
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        #print(F.mse_loss(outputs, target, reduction='mean').shape)
        # outputs is MxBxd, target is Bxd, below broadcasting works, and we average over MxBxd (usually d=1)
        #negative_log_likelihood = F.mse_loss(outputs, target, reduction='none').mean()
        #print(outputs.shape, target.shape)
        ## FIXME transfer data to device is expensive.
        datasigma = torch.Tensor([likparams[0]]).to(device)
        negative_log_likelihood = batch_size * torch.log(datasigma) + 0.5*batch_size*1.837+ 0.5 * batch_size * ((outputs - target)**2).mean() / datasigma**2

        return log_prior, log_variational_posterior, negative_log_likelihood

    def viloss(self, data, target):
        """Variational loss function `L(x,y)`.

        Args:
            data (torch.Tensor): A 2d input tensor `x`.
            target (torch.Tensor): A 2d output tensor `y`.

        Returns:
            float: The value of loss function.
        """
        datanoise, nsam, num_batches = self.loss_params
        log_prior, log_variational_posterior, negative_log_likelihood = self.sample_elbo(data, target, nsam, likparams=[datanoise])

        return (log_variational_posterior - log_prior)/num_batches + negative_log_likelihood


