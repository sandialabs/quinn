Loss Functions
=====================

Negative Log-Likelihood
-----------------------

QUiNN assumes a Gaussian likelihood for regression:

.. math::

    p(\mathcal{D} \mid w) = \prod_{i=1}^{N} \mathcal{N}(y_i \mid M(x_i; w),\, \sigma^2),

where :math:`\sigma` is the data noise standard deviation. The negative log-likelihood is

.. math::

    -\log p(\mathcal{D} \mid w) = \frac{N}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^{N} \|y_i - M(x_i; w)\|^2.


Gaussian Prior
--------------

When a prior is used, QUiNN employs an isotropic Gaussian centered at an anchor :math:`w_0`:

.. math::

    p(w) = \mathcal{N}(w \mid w_0,\, \sigma_{\text{prior}}^2 I_K).

The negative log-prior is

.. math::

    -\log p(w) = \frac{1}{2\sigma_{\text{prior}}^2} \|w - w_0\|^2 + \frac{K}{2}\log(2\pi\sigma_{\text{prior}}^2).


Negative Log-Posterior
---------------------

Combining the likelihood and the prior, the negative log-posterior used for training is

.. math::

    -\log p(w \mid \mathcal{D}) = \frac{1}{2\sigma^2} \sum_{i=1}^{N} \|y_i - M(x_i; w)\|^2 + \frac{N}{2}\log(2\pi\sigma^2) + \frac{N}{N_{\text{full}}} \left( \frac{1}{2\sigma_{\text{prior}}^2}\|w - w_0\|^2 + \frac{K}{2}\log(2\pi\sigma_{\text{prior}}^2) \right),

where :math:`N_{\text{full}}` is the full dataset size (relevant for mini-batch scaling).