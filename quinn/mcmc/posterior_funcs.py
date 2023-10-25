import torch
import numpy as np

from quinn.nns.tchutils import tch, npy


class Gaussian_likelihood_assumed_var(torch.nn.Module):
    """Calculates the likelihood of the parameters given the data assuming
    a gaussian model for the error and pre-specified variance.
    ----------
    Attributes:
        - sigma: Assumed variance of the error model. Type: float.
        - pi: number pi. Type: torch.Tensor.
    """

    def __init__(self, sigma):
        """
        Args:
            - sigma: Assumed variance of the error model.
        """
        super().__init__()

        self.sigma = sigma
        self.pi = tch(np.pi, rgrad=False)

    def forward(self, model, input, target, requires_grad=False):
        """
        Args:
            - model: model from which the parameters are taken. Type: NNWrap
            class.
            - input: model inputs. Type: numpy.ndarray.
            - targets: target predictions. Type: numpy.ndarray.
            - requires_grad: determines whether pytorch should expect to
            compute. Type: bool.
            - gradients in a given evaluation. Type: bool.
        ----------
        Returns:
            - Loss tensor. Type: torch.Tensor.
        """
        if requires_grad:
            predictions = model(input)  # , type_="torch")

            part1 = (
                0.5 * torch.sum(torch.pow(target - predictions, 2)) / self.sigma**2
            )

            part2 = (len(input) / 2) * torch.log(
                2 * torch.Tensor(self.pi) * self.sigma**2
            )

            return -part1 - part2
        else:
            with torch.no_grad():
                predictions = model(input)  # , type_="torch")

                part1 = (
                    0.5
                    * torch.sum(torch.pow(target - predictions, 2))
                    / self.sigma**2
                )

                part2 = (len(input) / 2) * torch.log(
                    2 * torch.Tensor(self.pi) * self.sigma**2
                )

            return -part1 - part2


class Gaussian_prior(torch.nn.Module):
    """Calculates the probability of the parameters given a Gaussian prior.
    ----------
    Attributes:
        - sigma: the standard deviation of the gaussian prior (same for all
        parameters). Type: float.
        - n_params: number of parameters being sampled. Type: int.
        - pi: number pi. Type: torch.Tensor.
    """

    def __init__(self, sigma, n_params):
        """
        Args:
            - sigma: the standard deviation of the gaussian prior (same for
            all parameters). Type: float.
            - n_params: number of parameters being sampled. Type: int.
        """
        super().__init__()

        self.sigma = sigma
        self.n_params = n_params
        self.pi = tch(np.pi, rgrad=False)

    def forward(self, model, requires_grad=False):
        """
        Args:
            - model: model from which the parameters are taken. Type:
            NNWrap class
            - requires_grad: determines whether pytorch should expect to
            compute gradients in a given evaluation. Type: bool.
        ----------
        Returns:
            - Loss tensor. Type: torch.Tensor.
        """
        if requires_grad:
            loss = 0

            for p in model.parameters():
                loss += torch.sum(torch.pow(p, 2))

            loss = loss / 2 / self.sigma**2

            loss += (self.n_params / 2) * torch.log(
                2 * torch.Tensor(self.pi) * self.sigma**2
            )

            return -loss
        else:
            with torch.no_grad():
                loss = 0

                for p in model.parameters():
                    loss += torch.sum(torch.pow(p, 2))

                loss = loss / 2 / self.sigma**2

                loss += (self.n_params / 2) * torch.log(
                    2 * torch.Tensor(self.pi) * self.sigma**2
                )

            return -loss


class Log_Posterior(torch.nn.Module):
    """Used to calculate the log posterior .
    ----------
    Attributes:
        - likelihood_fn: pytorch module that calculates the likelihood of the
        parameters of the model given the data. Type: torch.nn.Module.
        - prior_fn: pytorch module that calculates the probability of the
        parameters. Type: torch.nn.Module.
    """

    def __init__(self, likelihood_fn, prior_fn):
        """
        Args:
            - likelihood_fn: pytorch module that calculates the likelihood of
            the parameters of the model given the data. Type: torch.nn.Module.
            - prior_fn: pytorch module that calculates the probability of the
            parameters. Type: torch.nn.Module
        """
        super().__init__()

        self.likelihood_fn = likelihood_fn
        self.prior_fn = prior_fn

    def forward(self, model, input, target, requires_grad=False):
        """Calculates the log posterior .
        Args:
            - model: model from which the parameters are taken. Type: NNWrap
              class.
            - input: data that is input in the model. Type: numpy.ndarray.
            - target: data corresponding to the output of the model. Type:
            numpy.ndarray.
            - requires_grad: whether we will need to compute the gradients
            with respect to the log posterior. Type: bool.
        Returns:
            torch tensor of Log posterior. Type: torch.Tensor.
        """
        input = tch(input, rgrad=False)
        target = tch(target, rgrad=False)
        likelihood = self.likelihood_fn(model, input, target, requires_grad)
        prior = self.prior_fn(model, requires_grad)

        return likelihood + prior


class U_HMC(torch.nn.Module):
    """Used to calculate the U (potential energy) component of the hamiltonian.
    ----------
    Attributes:
        - likelihood_fn: pytorch module that calculates the likelihood of the
        parameters of the model given the data. Type: torch.nn.Module.
        - prior_fn: pytorch module that calculates the probability of the
        parameters. Type: torch.nn.Module.
    """

    def __init__(self, likelihood_fn, prior_fn):
        """
        Args:
            - likelihood_fn: pytorch module that calculates the likelihood of
            the parameters of the model given the data. Type: torch.nn.Module.
            - prior_fn: pytorch module that calculates the probability of the
            parameters. Type: torch.nn.Module.
        """
        super().__init__()

        self.likelihood_fn = likelihood_fn
        self.prior_fn = prior_fn

    def forward(self, model, input, target, requires_grad=False):
        """Calculates potential energy U of the Hamiltonian.
        Args:
            - model: model from which the parameters are taken.  Type: NNWrap
            class.
            - input: data that is input in the model. Type: numpy.ndarray.
            - target: data corresponding to the output of the model. Type:
            numpy.ndarray.
            - requires_grad: whether we will need to compute the gradients with
            respect to the log posterior. Type: bool.
        ----------
        Returns:
            - torch Tensor potential of hamiltonian, U.  Type: torch.Tensor.
        """
        loss = self.likelihood_fn(model, input, target, requires_grad=requires_grad)
        loss += self.prior_fn(model, requires_grad=requires_grad)

        return -loss
