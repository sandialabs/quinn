Quantification of Uncertainties in Neural Networks (QUiNN) is a python library centered around various probabilistic wrappers over PyTorch modules in order to provide uncertainty estimation in Neural Network (NN) predictions.

# Build the library
	./build.sh 
	or 
	./setup.py build; setup.py install

# Requirements
	numpy, scipy, matplotlib, pytorch

# Examples
	examples/ex_fit.py
	examples/ex_fit_2d.py
	examples/ex_ufit.py <method> # where method=mcmc, ens or vi.


