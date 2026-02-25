UQ4NN Solvers
=============

This section provides the mathematical foundations underlying each UQ solver in QUiNN. All solvers share a common goal: given a neural network model :math:`M(x; w)` with weights :math:`w \in \mathbb{R}^K`, and training data :math:`\{(x_i, y_i)\}_{i=1}^{N}`, approximate the posterior distribution

.. math::

    p(w \mid \mathcal{D}) \propto p(\mathcal{D} \mid w) \, p(w),

where :math:`\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}` is the dataset, and use samples from this posterior to propagate uncertainty through the network predictions.


.. _solver_mcmc:

MCMC (``NN_MCMC``)
------------------

Markov chain Monte Carlo directly samples from the posterior :math:`p(w \mid \mathcal{D})` by constructing a Markov chain whose stationary distribution is the target posterior. Given :math:`M_{\text{MCMC}}` chain samples :math:`\{w^{(j)}\}_{j=1}^{M_{\text{MCMC}}}` (after discarding burn-in), predictions are obtained as

.. math::

    M(x; w^{(j)}), \quad j = 1, \ldots, M_{\text{MCMC}}.

The posterior mean and variance of the prediction at a test point :math:`x^*` are estimated as

.. math::

    \bar{y}(x^*) = \frac{1}{M}\sum_{j=1}^{M} M(x^*; w^{(j)}), \qquad \text{Var}[y(x^*)] \approx \frac{1}{M-1}\sum_{j=1}^{M} \left(M(x^*; w^{(j)}) - \bar{y}(x^*)\right)^2.

QUiNN supports three MCMC samplers:


Adaptive Metropolis (AMCMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Adaptive Metropolis algorithm :cite:p:`haario:2001` uses a random-walk Metropolis-Hastings sampler with an adaptively tuned proposal covariance. At step :math:`t`, the proposal is

.. math::

    w' = w^{(t)} + \xi, \qquad \xi \sim \mathcal{N}(0, \Sigma_t),

where the proposal covariance is updated online using the sample covariance of the chain history:

.. math::

    \Sigma_t = \frac{\gamma \cdot 2.4^2}{K} \left( \hat{C}_t + 10^{-8} I_K \right),

with :math:`\hat{C}_t` the running sample covariance of :math:`\{w^{(0)}, \ldots, w^{(t)}\}`, :math:`K` the parameter dimensionality, and :math:`\gamma` a user-tunable scaling factor. The adaptation is triggered after an initial burn-in period :math:`t_0`, and the covariance is refreshed every :math:`t_{\text{adapt}}` steps. The standard Metropolis-Hastings acceptance criterion applies:

.. math::

    \alpha = \min\!\left(1,\; \frac{p(w' \mid \mathcal{D})}{p(w^{(t)} \mid \mathcal{D})}\right).


Hamiltonian Monte Carlo (HMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hamiltonian Monte Carlo :cite:p:`brooks:2011` augments the parameter space with an auxiliary momentum variable :math:`p \in \mathbb{R}^K` and defines a Hamiltonian

.. math::

    H(w, p) = U(w) + \tfrac{1}{2} p^\top p, \qquad U(w) = -\log p(w \mid \mathcal{D}).

The leapfrog integrator evolves the state :math:`(w, p)` for :math:`L` steps with step size :math:`\varepsilon`:

.. math::

    p_{t+\frac{1}{2}} &= p_t + \frac{\varepsilon}{2}\,\nabla_w \log p(w_t \mid \mathcal{D}), \\
    w_{t+1} &= w_t + \varepsilon\, p_{t+\frac{1}{2}}, \\
    p_{t+1} &= p_{t+\frac{1}{2}} + \frac{\varepsilon}{2}\,\nabla_w \log p(w_{t+1} \mid \mathcal{D}).

The proposal :math:`(w', p')` is accepted with probability

.. math::

    \alpha = \min\!\left(1,\; \exp\!\big(H(w, p) - H(w', p')\big)\right).


Metropolis-Adjusted Langevin Algorithm (MALA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MALA :cite:p:`girolami:2011` is a gradient-informed random-walk that uses the Langevin diffusion to construct proposals. The proposal is

.. math::

    w' = w^{(t)} + \frac{\varepsilon^2}{2}\,\nabla_w \log p(w^{(t)} \mid \mathcal{D}) + \varepsilon\, \xi, \qquad \xi \sim \mathcal{N}(0, I_K),

which corresponds to a single Euler-Maruyama discretization step of the Langevin stochastic differential equation. The Metropolis-Hastings correction ensures exact sampling.


.. _solver_ensemble:

Deep Ensemble (``NN_Ens``)
--------------------------

Deep Ensembles train :math:`J` independent networks from random initializations, optionally on random subsets of the data (controlled by the data fraction parameter :math:`\delta \in (0, 1]`). Each ensemble member :math:`j` minimizes the standard MSE loss

.. math::

    \mathcal{L}_j(w_j) = \frac{1}{|\mathcal{D}_j|} \sum_{(x_i, y_i) \in \mathcal{D}_j} \|y_i - M(x_i; w_j)\|^2,

where :math:`\mathcal{D}_j \subseteq \mathcal{D}`, :math:`|\mathcal{D}_j| = \lfloor \delta \cdot N \rfloor`. Predictions from all members are aggregated:

.. math::

    \bar{y}(x^*) = \frac{1}{J}\sum_{j=1}^{J} M(x^*; w_j), \qquad \text{Var}[y(x^*)] \approx \frac{1}{J-1}\sum_{j=1}^{J} \left(M(x^*; w_j) - \bar{y}(x^*)\right)^2.


.. _solver_rms:

Randomized MAP Sampling (``NN_RMS``)
------------------------------------

Randomized MAP Sampling (RMS) :cite:p:`pearce:2018` extends the deep ensemble approach by training each member with a randomized prior anchor. Each ensemble member :math:`j` minimizes the negative log-posterior

.. math::

    \mathcal{L}_j(w_j) = \frac{1}{2\sigma^2}\sum_{i \in \mathcal{D}_j} \|y_i - M(x_i; w_j)\|^2 + \frac{|\mathcal{D}_j|}{N} \cdot \frac{1}{2\sigma_{\text{prior}}^2}\|w_j - w_0^{(j)}\|^2,

where :math:`w_0^{(j)} \sim \mathcal{N}(0, \sigma_{\text{prior}}^2 I_K)` is a random anchor independently drawn for each member. This provides an implicit sampling scheme: the set of MAP solutions :math:`\{w_j^*\}_{j=1}^J` are approximate posterior samples.


.. _solver_vi:

Variational Inference (``NN_VI``)
---------------------------------

Variational inference approximates the posterior :math:`p(w \mid \mathcal{D})` with a tractable distribution :math:`q_\phi(w)` by minimizing the Kullback-Leibler (KL) divergence, which is equivalent to maximizing the Evidence Lower Bound (ELBO). QUiNN implements the *Bayes by Backprop* method :cite:p:`blundell:2015`.


Variational Family
^^^^^^^^^^^^^^^^^^

Each weight :math:`w_k` is parameterized with an independent Gaussian:

.. math::

    q_\phi(w_k) = \mathcal{N}(w_k \mid \mu_k,\; \sigma_k^2), \qquad \sigma_k = \log(1 + e^{\rho_k}),

where :math:`\phi = \{\mu_k, \rho_k\}_{k=1}^K` are the variational parameters. The softplus transformation ensures :math:`\sigma_k > 0`.


Scale Mixture Prior
^^^^^^^^^^^^^^^^^^^

The prior over each weight is a scale mixture of two Gaussians:

.. math::

    p(w_k) = \pi\,\mathcal{N}(w_k \mid 0,\,\sigma_1^2) + (1 - \pi)\,\mathcal{N}(w_k \mid 0,\,\sigma_2^2),

where :math:`\pi \in [0,1]` and :math:`\sigma_1, \sigma_2 > 0` are hyperparameters.


ELBO Loss
^^^^^^^^^

The variational loss (per mini-batch) is

.. math::

    \mathcal{L}(\phi) = \frac{1}{B}\bigl[\log q_\phi(w) - \log p(w)\bigr] + \frac{N}{2}\log(2\pi\sigma^2) + \frac{N}{2\sigma^2}\,\text{MSE}(w),

where :math:`w \sim q_\phi`, :math:`B` is the number of mini-batches, and :math:`\text{MSE}(w) = \frac{1}{|b|}\sum_{i \in b}\|y_i - M(x_i; w)\|^2` over the current mini-batch :math:`b`. At each training step, :math:`S` weight samples are drawn for a Monte Carlo estimate of the ELBO. At prediction time, weight samples from :math:`q_\phi(w)` are drawn to produce an ensemble of outputs.


.. _solver_laplace:

Laplace Approximation (``NN_Laplace``)
--------------------------------------

The Laplace approximation :cite:p:`mackay:1992` constructs a Gaussian approximation to the posterior centered at the MAP estimate :math:`w^*`:

.. math::

    p(w \mid \mathcal{D}) \approx \mathcal{N}\!\left(w \;\Big|\; w^*,\; \bigl[\nabla^2_w \mathcal{L}(w^*)\bigr]^{-1}\right),

where :math:`\mathcal{L}(w) = -\log p(w \mid \mathcal{D})` is the negative log-posterior and :math:`\nabla^2_w \mathcal{L}(w^*)` is its Hessian evaluated at the MAP.

**Step 1: MAP Training.** The network is trained by minimizing the negative log-posterior :math:`\mathcal{L}(w)`, yielding the MAP estimate :math:`w^*`.

**Step 2: Hessian Computation.** QUiNN supports two Hessian approximations:

- **Full Hessian:** The exact :math:`K \times K` Hessian is computed via second-order automatic differentiation:

  .. math::

      H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial w_i \partial w_j}\Bigg|_{w=w^*}.

- **Diagonal (Fisher) approximation:** The diagonal of the empirical Fisher information matrix is used as a Hessian proxy:

  .. math::

      \tilde{H}_{kk} = \frac{1}{N}\sum_{i=1}^{N} \left(\frac{\partial \mathcal{L}_i}{\partial w_k}\Bigg|_{w=w^*}\right)^2,

  where :math:`\mathcal{L}_i` denotes the per-sample loss. The resulting Hessian is diagonal: :math:`\tilde{H} = \text{diag}(\tilde{H}_{11}, \ldots, \tilde{H}_{KK})`.

**Step 3: Posterior Covariance.** The posterior covariance is

.. math::

    \Sigma = \left(s \cdot H\right)^{-1},

where :math:`s` is a user-tunable covariance scaling factor.

**Step 4: Prediction.** A predictive sample is drawn as

.. math::

    w \sim \mathcal{N}(w^*,\, \Sigma), \qquad y(x^*) = M(x^*; w).


.. _solver_swag:

SWAG (``NN_SWAG``)
------------------

Stochastic Weight Averaging-Gaussian (SWAG) :cite:p:`maddox:2019` approximates the posterior by fitting a Gaussian distribution to the SGD trajectory after initial training.

**Step 1: Pre-training.** The network is trained with the negative log-posterior loss to obtain a good initialization.

**Step 2: SGD Trajectory Collection.** Starting from the pre-trained weights, :math:`T` additional SGD steps are performed. At every :math:`c`-th step, the current weight vector :math:`w_t` is recorded and the running moments are updated:

.. math::

    \bar{w}_{n+1} &= \frac{n\,\bar{w}_n + w_t}{n+1}, \\
    \overline{w^2}_{n+1} &= \frac{n\,\overline{w^2}_n + w_t \odot w_t}{n+1},

where :math:`n = \lfloor t/c \rfloor` is the snapshot counter and :math:`\odot` is element-wise product.

**Step 3: Covariance Approximation.** The diagonal variance is

.. math::

    \Sigma_{\text{diag}} = \overline{w^2} - \bar{w} \odot \bar{w}.

For the low-rank variant, the last :math:`k` deviation vectors :math:`d_i = w_{t_i} - \bar{w}` are stored as columns of a matrix :math:`D \in \mathbb{R}^{K \times k}`.

**Step 4: Prediction.** A posterior sample is drawn as

.. math::

    w = \bar{w} + \frac{1}{\sqrt{2}}\,\text{diag}\!\left(\sqrt{\Sigma_{\text{diag}}}\right) z_1 + \frac{1}{\sqrt{2}}\,\frac{D\, z_2}{\sqrt{k-1}},

where :math:`z_1 \sim \mathcal{N}(0, I_K)` and :math:`z_2 \sim \mathcal{N}(0, I_k)`. If the covariance type is not low-rank, the second term is omitted. Predictions are obtained as :math:`y(x^*) = M(x^*; w)`.


Summary of Solvers
------------------

.. list-table::
   :header-rows: 1
   :widths: 18 30 15 15 22

   * - Solver
     - Posterior approximation
     - Training cost
     - Memory cost
     - Key hyperparameters
   * - ``NN_MCMC``
     - Exact (asymptotically)
     - High (:math:`O(M_{\text{MCMC}})` forward/backward passes)
     - :math:`O(M_{\text{MCMC}} \cdot K)`
     - :math:`M_{\text{MCMC}}`, sampler type, :math:`\sigma`, :math:`\varepsilon` (HMC)
   * - ``NN_Ens``
     - Implicit (point estimates)
     - :math:`J \times` single training
     - :math:`O(J \cdot K)`
     - :math:`J`, :math:`\delta`
   * - ``NN_RMS``
     - Implicit (randomized MAP)
     - :math:`J \times` single training
     - :math:`O(J \cdot K)`
     - :math:`J`, :math:`\sigma`, :math:`\sigma_{\text{prior}}`
   * - ``NN_VI``
     - Factored Gaussian :math:`q_\phi(w)`
     - :math:`\sim 2\times` single training
     - :math:`O(2K)` (for :math:`\mu, \rho`)
     - :math:`\pi`, :math:`\sigma_1`, :math:`\sigma_2`, :math:`S`
   * - ``NN_Laplace``
     - Gaussian at MAP
     - Single training + Hessian
     - :math:`O(K^2)` full / :math:`O(K)` diag
     - ``la_type``, ``cov_scale``, :math:`\sigma_{\text{prior}}`
   * - ``NN_SWAG``
     - Low-rank Gaussian
     - Single training + :math:`T` SGD steps
     - :math:`O(K \cdot k)` low-rank
     - :math:`k`, :math:`T`, :math:`c`, ``lr_swag``


References
----------

See the :doc:`refs` page for the full reference list. Key references for the solvers:

- **AMCMC:** :cite:t:`haario:2001`
- **HMC:** :cite:t:`brooks:2011`
- **MALA:** :cite:t:`girolami:2011`
- **RMS:** :cite:t:`pearce:2018`
- **VI (Bayes by Backprop):** :cite:t:`blundell:2015`
- **Laplace:** :cite:t:`mackay:1992`
- **SWAG:** :cite:t:`maddox:2019`
