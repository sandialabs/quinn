Test Suite
==========

QUiNN includes a comprehensive test suite in the ``tests/`` directory with
14 test files covering all library modules (148 tests total).  Run the full
suite with:

.. code-block:: bash

   pytest tests/ -q


Test Modules
------------

.. list-table::
   :header-rows: 1
   :widths: 22 48 10

   * - File
     - What is tested
     - Tests
   * - ``test_funcs.py``
     - Analytical test functions (``blundell``, ``Sine``, ``Sine10``,
       ``Summation``, ``Ackley``, ``x5``): output shapes, formulas, noise.
     - 18
   * - ``test_maps.py``
     - Data mapping utilities (``scale01ToDom``, ``scaleDomTo01``,
       ``Normalizer``, ``Standardizer``, ``Expon``, ``Logar``,
       ``ComposeMap``, ``Domainizer``): roundtrips, boundary values.
     - 14
   * - ``test_stats.py``
     - Statistics helpers (``get_stats``, ``get_domain``,
       ``intersect_domain``, ``diam``): mean/quantile computation, domain
       intersection.
     - 10
   * - ``test_rvar.py``
     - Random variable classes (``Gaussian_1d``, ``GMM2_1d``, ``MVN``):
       sample shapes, log-probability formulas, sample statistics.
     - 12
   * - ``test_mlp.py``
     - ``MLP`` and ``RNet`` construction: forward-pass shapes, parameter
       counts, activations (relu, tanh, sin), dropout, batch-norm, final
       transforms, residual connections.
     - 16
   * - ``test_nnwrap.py``
     - ``NNWrap`` numpy/torch interop: flatten/unflatten roundtrip, predict
       with explicit weights, loss, gradient, and Hessian computation.
     - 9
   * - ``test_nnfit.py``
     - ``nnfit`` training loop: loss decrease, validation split, mini-batch
       training, multi-output, weight decay.
     - 7
   * - ``test_losses.py``
     - Bayesian losses (``NegLogPost``, ``NegLogPrior``): anchor behaviour,
       symmetry, prior weighting, zero-residual baseline.
     - 5
   * - ``test_mcmc.py``
     - MCMC samplers (``AMCMC``, ``HMC``, ``MALA``): chain shape, acceptance
       rate, log-posterior storage, custom proposal covariance.
     - 8
   * - ``test_ensemble.py``
     - ``NN_Ens`` deep ensemble: creation, fit/predict, ensemble predictions,
       moment computation, data fraction, multi-output.
     - 7
   * - ``test_vi.py``
     - ``BNet`` and ``NN_VI``: Bayesian network creation, stochastic vs.
       deterministic forward, ELBO, uncertainty quantification.
     - 8
   * - ``test_solvers.py``
     - ``NN_MCMC``, ``NN_Laplace``, ``NN_SWAG``, ``NN_RMS``: creation, fit,
       predict, ensemble prediction, MAP estimation.
     - 14
   * - ``test_tchutils.py``
     - Torch utilities (``tch``, ``npy``, ``flatten_params``,
       ``recover_flattened``): type conversions, gradient tracking,
       flatten/recover roundtrip.
     - 9
   * - ``test_xutils.py``
     - General utilities (``idt``, ``savepk``/``loadpk``, ``cartes_list``,
       ``sample_sphere``, ``get_opt_bw``, ``get_pdf``, ``safe_cholesky``):
       identity function, serialisation, Cartesian products, KDE, Cholesky.
     - 11


Coverage Summary
----------------

The tests exercise all major subsystems of QUiNN:

- **Neural network architectures**: ``MLP``, ``RNet``, ``BNet``, ``NNWrap``.
- **Training**: ``nnfit`` with various loss functions, optimizers, and schedules.
- **UQ solvers**: ``NN_MCMC`` (AMCMC, HMC, MALA), ``NN_Ens``, ``NN_RMS``,
  ``NN_VI``, ``NN_Laplace``, ``NN_SWAG``.
- **MCMC samplers**: ``AMCMC``, ``HMC``, ``MALA`` independently.
- **Bayesian losses**: ``NegLogPost``, ``NegLogPrior``.
- **Random variables**: ``Gaussian_1d``, ``GMM2_1d``, ``MVN``.
- **Utilities**: data maps, statistics, torch helpers, pickle, KDE, Cholesky.
- **Test functions**: ``blundell``, ``Sine``, ``Sine10``, ``Ackley``, etc.
