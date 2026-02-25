QUiNN ships with several example scripts in the ``examples/`` directory that
demonstrate key capabilities: deterministic fitting, UQ solvers, custom losses,
and MCMC diagnostics.

Fit Examples
============

``ex_fit.py`` — 1-D Function Approximation
-------------------------------------------

Fits a 1-D function (``blundell``) using a three-hidden-layer ``MLP`` with
tanh activations.  Demonstrates:

- Creating an ``MLP(ndim, nout, (11,11,11), activ='tanh')`` architecture.
- Splitting data into training and validation sets.
- Calling ``nnet.fit(xtrn, ytrn, val=[xval, yval], ...)`` for training.
- Diagnostic plots via ``predict_plot`` and ``plot_1d_fits``.

.. code-block:: python

   nnet = MLP(ndim, nout, (11,11,11), activ='tanh', device=device)
   nnet.fit(xtrn, ytrn, val=[xval, yval], lrate=0.01, nepochs=2000)


``ex_fit_2d.py`` — 2-D Fit with Periodic Loss
-----------------------------------------------

Fits the 2-D Ackley function with a periodic boundary regularisation.
Demonstrates:

- Mini-batch training (``batch_size=10``).
- Using a custom loss function (``PeriodicLoss``) to enforce matching values
  on opposite boundaries.
- 2-D contour plotting of the true model vs. the NN approximation.

.. code-block:: python

   loss = PeriodicLoss(nnet.nnmodel, 10.1, [tch(bdry1), tch(bdry2)])
   nnet.fit(xtrn, ytrn, val=[xval, yval], loss_xy=loss,
            batch_size=10, nepochs=1000)


``ex_ufit.py`` — All UQ Solvers
---------------------------------

A single script that runs any of the seven UQ methods, selected via a
command-line argument:

.. code-block:: bash

   python ex_ufit.py amcmc    # Adaptive Metropolis
   python ex_ufit.py hmc      # Hamiltonian Monte Carlo
   python ex_ufit.py vi       # Variational Inference
   python ex_ufit.py ens      # Deep Ensemble
   python ex_ufit.py rms      # Randomized MAP Sampling
   python ex_ufit.py laplace  # Laplace Approximation
   python ex_ufit.py swag     # SWAG

Each branch creates the appropriate solver (``NN_MCMC``, ``NN_VI``, ``NN_Ens``,
``NN_RMS``, ``NN_Laplace``, ``NN_SWAG``), trains it, and plots:

- 1-D fit with uncertainty bands.
- Diagonal comparison plots.
- Predictive mean and variance via ``predict_mom_sample``.

The underlying NN is a ``RNet`` with polynomial weight parametrization.

Linear Regression
=================

``ex_lreg_mcmc.py`` — Linear Regression via MCMC
--------------------------------------------------

Demonstrates Bayesian inference on a simple linear model
(``torch.nn.Linear(1, 1)``) using ``NN_MCMC`` with the Adaptive Metropolis
sampler.  Showcases MCMC-specific diagnostics:

- Chain trace plots (``plot_xrv``).
- Triangle / corner plots of the posterior (``plot_tri``, ``plot_pdfs``).
- Saving the chain to file (``np.savetxt('chain.txt', ...)``).


Loss Visualization
==================

``ex_loss.py`` — Loss Landscape Visualisation
-----------------------------------------------

Visualises the loss landscape of a trained ``RNet`` by:

- Walking along anchor directions in parameter space.
- Plotting 1-D and 2-D slices of the loss surface around the optimum.
- Shows how the loss topology depends on the weight parametrization order.

Requires external data files (``ptrain.txt``, ``ytrain.txt``).

