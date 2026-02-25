NN Architectures
================

QUiNN provides several neural network architectures that can serve as the
underlying model :math:`M(x;\,w)` for any of the UQ solvers.  All architectures
inherit from :class:`MLPBase`, which extends :class:`torch.nn.Module` with
convenience methods for training, prediction, and parameter management.


Base Class (``MLPBase``)
------------------------

:class:`MLPBase` is the abstract base that every QUiNN architecture must
subclass.  It stores the input/output dimensions, tracks whether the network
has been trained, and exposes the following core interface:

- ``fit(xtrn, ytrn, **kwargs)`` — trains the network via ``nnfit``,
  stores the best model snapshot, and records the loss history.
- ``predict(x)`` — numpy-in / numpy-out convenience: converts the input,
  evaluates the (best) model, and returns a numpy array.
- ``numpar()`` — returns the total number of parameters :math:`K`.
- ``predict_plot`` / ``plot_1d_fits`` — diagnostic plotting helpers.


Multilayer Perceptron (``MLP``)
-------------------------------

The primary architecture in QUiNN is a fully connected feedforward network
(multilayer perceptron).  For an input :math:`x \in \mathbb{R}^d`, hidden layer
widths :math:`(h_1, h_2, \ldots, h_L)`, and output :math:`y \in \mathbb{R}^o`,
the forward pass is

.. math::

    z_0 &= x, \\
    z_{\ell} &= \phi\!\left(W_\ell\, z_{\ell-1} + b_\ell\right),
         \quad \ell = 1, \ldots, L, \\
    y &= W_{L+1}\, z_L + b_{L+1},

where :math:`W_\ell` and :math:`b_\ell` are the weight matrix and bias vector
of layer :math:`\ell`, and :math:`\phi` is the elementwise activation function.


Constructor Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 18 12 50

   * - Parameter
     - Default
     - Description
   * - ``indim``
     -
     - Input dimensionality :math:`d`.
   * - ``outdim``
     -
     - Output dimensionality :math:`o`.
   * - ``hls``
     -
     - Tuple of hidden-layer widths :math:`(h_1, \ldots, h_L)`.
   * - ``activ``
     - ``'relu'``
     - Activation function :math:`\phi`. Options: ``'relu'`` (:math:`\max(0,x)`),
       ``'tanh'`` (:math:`\tanh(x)`), ``'sin'``
       (:math:`A\sin(2\pi x / T)`), or identity.
   * - ``biasorno``
     - ``True``
     - Include bias terms :math:`b_\ell`.
   * - ``bnorm``
     - ``False``
     - Apply Batch Normalization after each layer.
   * - ``bnlearn``
     - ``True``
     - Make batch-norm parameters learnable.
   * - ``dropout``
     - ``0.0``
     - Dropout fraction :math:`p`.  Applied after each linear layer when
       :math:`p > 0`.
   * - ``final_transform``
     - ``None``
     - Optional output transform.  ``'exp'`` applies :math:`e^{(\cdot)}` to enforce
       positivity.
   * - ``device``
     - ``'cpu'``
     - Compute device (``'cpu'`` or ``'cuda'``).


Architecture Diagram
^^^^^^^^^^^^^^^^^^^^

A three-hidden-layer MLP with widths :math:`(h_1, h_2, h_3)`:

.. code-block:: text

    Input (d)
      │
      ├─ Linear(d → h₁) ─ [Dropout] ─ [BatchNorm] ─ Activation
      ├─ Linear(h₁→ h₂) ─ [Dropout] ─ [BatchNorm] ─ Activation
      ├─ Linear(h₂→ h₃) ─ [Dropout] ─ [BatchNorm] ─ Activation
      ├─ Linear(h₃→ o)  ─ [Dropout] ─ [BatchNorm]
      │
      └─ [final_transform]
    Output (o)

The total parameter count is

.. math::

    K = \sum_{\ell=1}^{L+1}\!\bigl(n_{\ell-1}\,n_\ell + n_\ell\bigr),

where :math:`n_0 = d`, :math:`n_{L+1} = o`, and :math:`n_\ell = h_\ell` for
:math:`\ell = 1,\ldots,L` (assuming biases are included).


Residual Network (``RNet``)
----------------------------

The ``RNet`` class implements a Residual Neural Network whose layers can be
viewed as a discretised ODE (the *neural ODE* perspective).  All hidden layers
share the same width :math:`r`, and the forward pass is

.. math::

    z_0 &= x \quad \text{(or } z_0 = \phi(W_{\text{pre}}\,x + b_{\text{pre}})
          \text{ if a pre-layer is used)}, \\
    z_{\ell+1} &= z_\ell + \Delta t\;\phi\!\left(W_\ell\,z_\ell + b_\ell\right),
          \quad \ell = 0,\ldots,L, \\
    y &= z_{L+1} \quad \text{(or } y = W_{\text{post}}\,z_{L+1} + b_{\text{post}}
          \text{ if a post-layer is used)},

where :math:`\Delta t = 1/(L+1)` is the step size.  Setting ``mlp=True``
drops the skip connections, recovering a standard MLP.


Weight Parameterization
^^^^^^^^^^^^^^^^^^^^^^^

A distinguishing feature of ``RNet`` is that the layer weights need not be
independent: they can be parameterised as a function of the *layer index*
(interpreted as a time variable :math:`t \in [0,1]`).  The ``LayerFcn``
hierarchy provides:

.. list-table::
   :header-rows: 1
   :widths: 15 35 12

   * - Class
     - Weight function :math:`W(t)`
     - Parameters
   * - ``NonPar``
     - Independent weight per layer (standard ResNet)
     - :math:`L+1`
   * - ``Const``
     - :math:`W(t) = A`
     - 1
   * - ``Lin``
     - :math:`W(t) = A + Bt`
     - 2
   * - ``Quad``
     - :math:`W(t) = A + Bt + Ct^2`
     - 3
   * - ``Cubic``
     - :math:`W(t) = A + Bt + Ct^2 + Dt^3`
     - 4
   * - ``Poly``
     - :math:`W(t) = \sum_{i=0}^{p} A_i\,t^i`
     - :math:`p+1`

These parameterisations dramatically reduce the number of free parameters
while preserving expressiveness.

Constructor Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 18 12 50

   * - Parameter
     - Default
     - Description
   * - ``rdim``
     -
     - Hidden-layer width :math:`r`.
   * - ``nlayers``
     -
     - Number of residual layers :math:`L`.
   * - ``wp_function``
     - ``None``
     - A ``LayerFcn`` instance for weight parameterization.  ``None`` uses
       ``NonPar`` (standard ResNet).
   * - ``indim``
     - ``rdim``
     - Input dimensionality :math:`d`.  If :math:`d \neq r`, ``layer_pre``
       must be ``True``.
   * - ``outdim``
     - ``rdim``
     - Output dimensionality :math:`o`.  If :math:`o \neq r`, ``layer_post``
       must be ``True``.
   * - ``biasorno``
     - ``True``
     - Include bias terms.
   * - ``mlp``
     - ``False``
     - If ``True``, skip connections are removed (standard MLP behavior).
   * - ``layer_pre``
     - ``False``
     - Prepend a linear layer projecting :math:`d \to r`.
   * - ``layer_post``
     - ``False``
     - Append a linear layer projecting :math:`r \to o`.
   * - ``final_layer``
     - ``None``
     - Output transform: ``'exp'``, ``'logabs'``, or ``'sum'``.
   * - ``init_factor``
     - ``1.0``
     - Multiplicative factor applied to weight initialization.


PyTorch Functions
-----------------

The ``quinn.nns.nns`` module provides small reusable building blocks that can be
used as standalone models or composed within larger architectures:

.. list-table::
   :header-rows: 1
   :widths: 18 40

   * - Module
     - Computation
   * - ``Polynomial(p)``
     - :math:`\sum_{i=0}^{p} c_i\,x^i` with learnable coefficients
   * - ``Polynomial3``
     - :math:`a + bx + cx^2 + dx^3` (four scalar parameters)
   * - ``Constant``
     - :math:`C` (single learnable scalar)
   * - ``Gaussian``
     - :math:`e^{-x^2}`
   * - ``Sine(A, T)``
     - :math:`A\sin(2\pi x / T)`
   * - ``SiLU``
     - :math:`x\,\sigma(x)` (Sigmoid Linear Unit)
   * - ``Expon``
     - :math:`e^{x}`
   * - ``TwoLayerNet``
     - Two linear layers with a cubic polynomial in between
   * - ``MLP_simple``
     - Lightweight MLP with tanh activations, specified as a single layer-width tuple


Numpy Wrapper (``NNWrap``)
--------------------------

:class:`NNWrap` provides a numpy-centric interface around any
:class:`torch.nn.Module`, enabling:

- ``__call__(x)`` — evaluate the wrapped model with numpy arrays.
- ``p_flatten()`` / ``p_unflatten(flat)`` — flatten all parameters into a
  1-D numpy vector and restore them.  This is essential for MCMC and Laplace
  solvers that operate in flat parameter space.
- ``predict(x, weights)`` — evaluate the model after setting parameters
  from a flat weight vector.
- ``calc_loss(weights, loss_fn, x, y)`` — compute a loss after unflattening
  weights.
- ``calc_lossgrad(weights, loss_fn, x, y)`` — compute the gradient of a
  loss w.r.t. the flat parameter vector.
- ``calc_hess_full(weights, loss_fn, x, y)`` — compute the full Hessian of
  the loss (used by ``NN_Laplace``).



Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 15 22 28 20

   * - Architecture
     - Key feature
     - When to use
     - Parameter count
   * - ``MLP``
     - Fully connected
     - General-purpose regression
     - :math:`O\!\left(\sum h_\ell h_{\ell+1}\right)`
   * - ``RNet``
     - Residual / neural ODE
     - Deep networks, parameter-efficient models
     - :math:`O(r^2 \cdot p)` with weight param
   * - ``BNet``
     - Variational weights
     - Used internally by ``NN_VI``
     - :math:`2K`
   * - ``NNWrap``
     - Numpy interface
     - MCMC, Laplace, flat-parameter manipulations
     - Same as wrapped model

