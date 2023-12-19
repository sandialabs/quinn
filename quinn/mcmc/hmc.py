import torch
import numpy as np
from quinn.nns.nnwrap import NNWrap_Torch


class HMC_NN:
    """
    This class performs HMC for neural networks to estimate
    the posterior over the parameters.
    The likelihood and prior functions are given as pytorch modules.
    This implementation is based on Neal (2011).
    -----------
    Attributes:
        - epsilon (float): step size for the Leap Frog optimization algorithm.
        - L_steps (int): number of steps within the Leap Frog optimization algorithm.
        - u_func (torch.nn.Module): pytorch module calculating the U component of the
        hamiltonian.
        - nmcmc (int): number of samples from sampler.
        - nburning (int): number of initial samples to burn.
        - pos_theta (list[np.ndarray]): list containing samples from the posterior.
    """

    def __init__(self, u_func, sampling_params, nmcmc, nburning) -> None:
        """
        Args:
            - u_func: torch.nn.Module class computing the potential energy, U,
            of the hamiltonian. Type: torch.nn.Module.
            - sampling_params: dictionary including the parameters required
            for the sampling algorithm. Parameters to be included are "epsilon"
            and "L_steps" (see the attributes description for a description of
            these parameters.)
        """

        assert "epsilon" in sampling_params, "Parameter epsilon not provided."
        assert "L_steps" in sampling_params, "Parameter L_steps not provided."

        # HMC parameters
        self.epsilon = sampling_params["epsilon"]
        self.L = sampling_params["L_steps"]
        self.nmcmc = nmcmc
        self.nburning = nburning
        self.U_func = u_func

        # store output
        self.pos_theta = []
        self.pos_U = []

    def calc_K(self, p):
        """
        Calculate the K component of the Hamiltonian ("kinetic energy") given
        the momentum variable, p.
        ---------
        Args:
            - p (np.ndarray): momentum variable. Type: numpy array.
        ---------
        Output:
            - K (float): K component of the Hamiltonian ("kinetic energy"). Type: float.
        """
        return np.sum(np.square(p)) / 2

    def run(self, init_guess, x_data, y_data, model):
        """
        Run the sampler for a defined linear model.
        Sampling is done with the leapfrog method with step size
        epsilon and L steps within loop. The sampled samples are stored in
        the class attribute "pos_theta".
        ---------
        Args:
            - model (NNWrap_Torch class): NNWrap created with the model over whose parameters we are
            obtaining the posterior.
            - init_guess (np.ndarray): initial guess to start the sampling.
            - x_data (np.ndarray): input array with shape (N, M), where M is the number of
            features.
            - y_data (np.ndarray): target array with shape (N, d), where d is the dimension
            is the dimension of the target points.
        ---------
        Returns:
            - dictionary with the following content:
                - "chain" (np.ndarray): list of samples.
                - "mapparameters" (np.ndarray): sample that maximizes the log posterior.
                - "maxpost" (float): maximum value of the log posterior achieved during
                sampling.
                - "accrate" (float): acceptance rate during sampling.
        """
        assert isinstance(
            model, NNWrap_Torch
        ), "The model must be an instance of the NNWrap_Torch class"
        # Initializing theta
        theta = init_guess
        theta_size = len(theta)

        n_accept = 0

        map_theta = 0
        max_post = 0

        self.pos_theta.append(theta)

        for ii in range(self.nmcmc):
            p = np.random.randn(theta_size)

            if ii % (self.nmcmc // 10) == 0 and ii != 0:
                print(f"Iteration {ii}: acceptance rate = ", n_accept / ii)

            current_U = model.calc_loss(theta, self.U_func, x_data, y_data)
            current_K = self.calc_K(p)

            if ii == 0:
                self.pos_U.append(current_U)

            theta_proposal = theta

            # Make a half step for momentum at the beginning (Leapfrog Method step starts here)

            p = (
                p
                - self.epsilon
                * model.calc_grad_wrt_loss(theta_proposal, self.U_func, x_data, y_data)
                / 2
            )

            # print("Mid p", p)

            for jj in range(self.L):
                # Make a full step for the position

                theta_proposal = theta_proposal + self.epsilon * p

                # Make a full step for the momentum, expecpt at the end of the trajectory

                if jj != self.L - 1:
                    p = p - self.epsilon * model.calc_grad_wrt_loss(
                        theta_proposal, self.U_func, x_data, y_data
                    )

            # Make a half step for momentum at the end (Leapfrog Method step ends here)
            p = (
                p
                - self.epsilon
                * model.calc_grad_wrt_loss(theta_proposal, self.U_func, x_data, y_data)
                / 2
            )

            # Negate momentum to make proposal symmetric

            p = -p

            # Evaluate kinetic and potential energies

            proposed_U = model.calc_loss(theta_proposal, self.U_func, x_data, y_data)
            proposed_K = self.calc_K(p)

            current_H = current_U + current_K
            proposed_H = proposed_U + proposed_K

            # MH step (accept/reject)

            mh_prob = min(1, np.exp(current_H - proposed_H))

            u = np.random.uniform(0, 1)

            if u < mh_prob:
                n_accept += 1
                theta = theta_proposal
                self.pos_U.append(proposed_U)
                if max_post < -proposed_U:
                    max_post = -proposed_U
                    map_theta = theta_proposal
            else:
                self.pos_U.append(current_U)

            if ii + 1 > self.nburning:
                self.pos_theta.append(theta)

        # Calculate accept rate
        accept_rate = (n_accept / self.nmcmc) * 100
        print("End of sampling.")
        print("{:.3f}% were accepted".format(accept_rate))

        results = {
            "chain": np.array(self.pos_theta),
            "logpost": np.array(self.pos_U),
            "mapparams": map_theta,
            "maxpost": max_post,
            "accrate": accept_rate,
        }
        return results
