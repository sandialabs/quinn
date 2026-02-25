NN Training
===========

All QUiNN architectures are trained through the generic ``nnfit`` function,
which provides a configurable training loop with support for multiple loss
functions, optimizers, learning-rate schedules, mini-batching, and automatic
early stopping.

The standard entry point is ``MLPBase.fit(xtrn, ytrn, **kwargs)``, which
delegates to ``nnfit`` internally.


Training Objective
------------------

At each gradient step the optimizer minimizes a loss function
:math:`\mathcal{L}(w)` that depends on the current mini-batch
:math:`\mathcal{B} \subseteq \{1,\ldots,N\}`:

.. math::

   w^{(t+1)} = w^{(t)} - \eta_t\,\nabla_w \mathcal{L}_{\mathcal{B}}(w^{(t)}),

where :math:`\eta_t` is the learning rate at step :math:`t`.


Loss Functions
--------------

``nnfit`` selects the loss through either the ``loss_fn`` string or a
user-supplied callable ``loss_xy``.

Mean Squared Error (``loss_fn='mse'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default loss is the mean squared error over the mini-batch:

.. math::

   \mathcal{L}_{\text{MSE}}(w) = \frac{1}{|\mathcal{B}|}
       \sum_{i \in \mathcal{B}} \bigl\|y_i - M(x_i;\,w)\bigr\|^2.

Negative Log-Posterior (``loss_fn='logpost'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a Bayesian prior is used (e.g. for ``NN_MCMC`` or ``NN_RMS``), the
loss is the negative log-posterior combining a Gaussian likelihood and a
Gaussian prior centred at an anchor :math:`w_0`:

.. math::

   \mathcal{L}_{\text{logpost}}(w) =
       \frac{1}{2\sigma^2}\sum_{i \in \mathcal{B}}\bigl\|y_i - M(x_i;\,w)\bigr\|^2
       + \frac{|\mathcal{B}|}{2}\log(2\pi\sigma^2)
       + \frac{|\mathcal{B}|}{N}\!\left(
           \frac{1}{2\sigma_{\text{prior}}^2}\|w - w_0\|^2
           + \frac{K}{2}\log(2\pi\sigma_{\text{prior}}^2)
         \right),

where :math:`\sigma` is the data noise, :math:`\sigma_{\text{prior}}` the
prior standard deviation, and :math:`K` the parameter count.  This loss
requires the ``datanoise`` and ``priorparams`` arguments.

Log-Loss (``loss_fn='logloss'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fits in the log-transformed space with a user-specified shift
:math:`y_{\text{shift}}`:

.. math::

   \mathcal{L}_{\text{log}}(w) = \frac{1}{|\mathcal{B}|}
       \sum_{i \in \mathcal{B}}
       \bigl[\log(M(x_i;\,w) - y_{\text{shift}})
             - \log(y_i - y_{\text{shift}})\bigr]^2.

This is activated by ``loss_fn='logloss'`` and requires the ``lossparams``
argument.

Custom Loss (``loss_xy``)
^^^^^^^^^^^^^^^^^^^^^^^^^

Any callable with signature ``loss_xy(x_batch, y_batch) -> scalar tensor``
can be passed directly.  When ``loss_xy`` is provided, the ``loss_fn``
string is ignored.  This mechanism is used internally by ``NN_VI`` (Bayes
by Backprop) and can be used for any problem-specific objective.


Optimizers
----------

``nnfit`` supports two first-order optimizers:

.. list-table::
   :header-rows: 1
   :widths: 15 55

   * - String
     - Algorithm
   * - ``'adam'``
     - Adam (adaptive moment estimation), the default.  Updates each parameter
       with bias-corrected first and second moment estimates.
   * - ``'sgd'``
     - Stochastic Gradient Descent with optional momentum (via PyTorch
       defaults).

Both accept an optional weight-decay parameter ``wd`` that adds an L2
penalty :math:`\frac{\lambda}{2}\|w\|^2` to the loss:

.. math::

   \mathcal{L}_{\text{reg}}(w) = \mathcal{L}(w) + \frac{\lambda}{2}\|w\|^2.


Learning Rate Schedules
-----------------------

Three scheduling modes are available (mutually exclusive):

1. **Constant** — when neither ``lmbd`` nor ``scheduler_lr`` is set, the
   learning rate stays at ``lrate`` throughout training.

2. **Lambda schedule** — a user-defined function ``lmbd(epoch)`` that returns
   a multiplicative factor.  The effective rate at epoch :math:`t` is

   .. math::

      \eta_t = \texttt{lrate} \times \texttt{lmbd}(t).

3. **ReduceLROnPlateau** — set ``scheduler_lr='ReduceLROnPlateau'``.  The
   scheduler monitors the validation loss and reduces the learning rate by
   ``factor`` whenever the loss plateaus for ``cooldown`` epochs:

   .. math::

      \eta \leftarrow \texttt{factor} \times \eta
      \quad\text{if validation loss stagnates for \texttt{cooldown} epochs}.


Mini-Batch Training
-------------------

When ``batch_size`` is specified and smaller than :math:`N`, each epoch is
split into :math:`\lceil N / B \rceil` sub-epochs, where :math:`B` is the
batch size.  At the start of every epoch the training data is randomly
permuted, and each sub-epoch draws a contiguous slice of size :math:`B`:

.. math::

   \mathcal{B}_k = \bigl\{\pi(kB+1),\;\pi(kB+2),\;\ldots,\;\pi\!\bigl(\min((k+1)B,\,N)\bigr)\bigr\},

where :math:`\pi` is the random permutation.  When ``batch_size`` is
``None`` or exceeds :math:`N`, full-batch training is used.


Early Stopping
--------------

At every gradient step the validation loss is evaluated (without gradients).
If it improves on the current best, a deep copy of the model is
checkpointed:

.. math::

   w^* = \arg\min_{w^{(t)}} \mathcal{L}_{\text{val}}(w^{(t)}).

The returned model is always the best snapshot, not the final-epoch model.
When no separate validation set is provided (``val=None``), the training set
is used for both training and validation.


Arguments
---------

.. list-table::
   :header-rows: 1
   :widths: 18 12 50

   * - Argument
     - Default
     - Description
   * - ``nnmodel``
     -
     - The ``torch.nn.Module`` to train.
   * - ``xtrn``
     -
     - Training inputs, numpy array of shape :math:`(N,\,d)`.
   * - ``ytrn``
     -
     - Training targets, numpy array of shape :math:`(N,\,o)`.
   * - ``val``
     - ``None``
     - Validation data as an ``(x, y)`` tuple.  If ``None``, the training set
       doubles as validation.
   * - ``loss_fn``
     - ``'mse'``
     - Loss identifier: ``'mse'``, ``'logpost'``, or ``'logloss'``.
       Ignored when ``loss_xy`` is provided.
   * - ``loss_xy``
     - ``None``
     - Custom loss callable ``loss_xy(x, y) -> scalar``.  Overrides
       ``loss_fn`` when provided.
   * - ``datanoise``
     - ``None``
     - Data noise :math:`\sigma` for ``'logpost'`` loss.
   * - ``wd``
     - ``0.0``
     - Weight decay (L2 regularisation) coefficient :math:`\lambda`.
   * - ``priorparams``
     - ``None``
     - Dictionary with keys ``'sigma'`` (:math:`\sigma_{\text{prior}}`) and
       ``'anchor'`` (:math:`w_0`) for the Gaussian prior.
   * - ``lossparams``
     - ``None``
     - Parameters for custom losses (e.g. ``[y_shift]`` for ``'logloss'``).
   * - ``optimizer``
     - ``'adam'``
     - Optimizer string: ``'adam'`` or ``'sgd'``.
   * - ``lrate``
     - ``0.1``
     - Base learning rate :math:`\eta`.
   * - ``lmbd``
     - ``None``
     - Lambda schedule ``lmbd(epoch) -> float``.  Effective rate is
       ``lrate * lmbd(epoch)``.
   * - ``scheduler_lr``
     - ``None``
     - Adaptive scheduler.  Currently only ``'ReduceLROnPlateau'`` is
       supported.  Cannot be combined with ``lmbd``.
   * - ``nepochs``
     - ``5000``
     - Total number of training epochs.
   * - ``batch_size``
     - ``None``
     - Mini-batch size :math:`B`.  ``None`` means full-batch.
   * - ``gradcheck``
     - ``False``
     - If ``True``, verify auto-diff gradients numerically (slow,
       experimental).
   * - ``cooldown``
     - ``100``
     - Cooldown epochs for ``ReduceLROnPlateau``.
   * - ``factor``
     - ``0.95``
     - Multiplicative factor for ``ReduceLROnPlateau``.
   * - ``freq_out``
     - ``100``
     - Screen-output frequency (in epochs).
   * - ``freq_plot``
     - ``1000``
     - Loss-history plot frequency (in epochs).
   * - ``lhist_suffix``
     - ``''``
     - Filename suffix for the saved loss-history figures.


Return Value
------------

``nnfit`` returns a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 22 55

   * - Key
     - Content
   * - ``'best_nnmodel'``
     - Deep copy of the model at the best validation loss.
   * - ``'best_loss'``
     - Best validation loss value.
   * - ``'best_epoch'``
     - Epoch index at which the best loss occurred.
   * - ``'best_fepoch'``
     - Fractional epoch (accounts for sub-epochs in mini-batch training).
   * - ``'history'``
     - List of ``[fepoch, batch_loss, train_loss, val_loss]`` recorded at
       every gradient step.

