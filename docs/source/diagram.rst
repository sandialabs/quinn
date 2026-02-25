Class Inheritance Diagrams
==========================

The diagrams below highlight the essential inheritance relationships
in QUiNN — groups where classes form multi-level hierarchies.

UQ Solvers
----------

``QUiNNBase`` is the root.  ``NN_Ens`` adds a second level from which
``NN_SWAG``, ``NN_Laplace``, and ``NN_RMS`` derive.

.. inheritance-diagram:: quinn.solvers.quinn quinn.solvers.nn_ens quinn.solvers.nn_mcmc quinn.solvers.nn_vi quinn.solvers.nn_laplace quinn.solvers.nn_swag quinn.solvers.nn_rms
   :top-classes: quinn.solvers.quinn.QUiNNBase
   :parts: 1

Neural-Network Architectures
-----------------------------

``MLPBase`` (a ``torch.nn.Module``) is extended by ``MLP``, ``RNet``,
and ``SNet``.

.. inheritance-diagram:: quinn.nns.nnbase quinn.nns.mlp quinn.nns.rnet quinn.nns.nnwrap
   :top-classes: quinn.nns.nnbase.MLPBase
   :parts: 1

MCMC Samplers
-------------

``MCMCBase`` is the common ancestor of the three samplers.

.. inheritance-diagram:: quinn.mcmc.mcmc quinn.mcmc.admcmc quinn.mcmc.hmc quinn.mcmc.mala
   :top-classes: quinn.mcmc.mcmc.MCMCBase
   :parts: 1

Random Variables
----------------

``RV`` (a ``torch.nn.Module``) is specialised into ``MVN``,
``Gaussian_1d``, and ``GMM2_1d``.

.. inheritance-diagram:: quinn.rvar.rvs
   :top-classes: quinn.rvar.rvs.RV
   :parts: 1

Input/Output Maps
-----------------

``XMap`` branches into ``LinearScaler`` (with children ``Standardizer``,
``Normalizer``, ``Domainizer``) and several other leaf maps.

.. inheritance-diagram:: quinn.utils.maps
   :top-classes: quinn.utils.maps.XMap
   :parts: 1