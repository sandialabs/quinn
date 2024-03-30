#!/usr/bin/env python
"""Module for the Bayesian network."""

import copy
import math
import torch

from ..rvar.rvs import Gaussian_1d, GMM2_1d


class BNet(torch.nn.Module):
    """Bayesian NN class.

    Attributes:
        nnmodel (torch.nn.Module): The underlying PyTorch NN module.
        nparams (int): Number of deterministic parameters.
        param_names (list[str]): List of parameter names.
        param_priors (list[quinn.vi.rvs.RV]): List of parameter priors.
        rparams (list[quinn.rvar.rvs.RV]): List of variational PDFs.
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
        
        try:
            self.device = nnmodel.device
        except AttributeError:
            self.device = 'cpu'
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
                self.rparams.append(Gaussian_1d(mu, logsigma=rho))
                
                ## PRIOR
                self.param_priors.append(GMM2_1d(pi, sigma1, sigma2))
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
            #print("AAAA ", i, param_name, param_name.split("."), par_samples[i])
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
        negative_log_likelihood = batch_size * torch.log(datasigma) + 0.5*batch_size*torch.log(2.0*torch.tensor(math.pi))+ 0.5 * batch_size * ((outputs - target)**2).mean() / datasigma**2

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

