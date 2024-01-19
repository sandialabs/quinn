#!/usr/bin/env python

import io
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
README = io.open(os.path.join(here, "README.md"), encoding="latin-1").read()

setup(
    name="QUINN",
    version="1.0",
    url="https://github.com/XYZ",
    description="Library for augmenting NN with UQ",
    long_description=README,
    author="K. Sargsyan and team",
    author_email="ksargsy@sandia.gov",
    license="MIT",
    platforms="BSD 3-clause",
    packages=find_packages(),
    package_dir={"quinn": "quinn"},
    # package_data={"":["*.pdf"]},
    include_package_data=True,
    py_modules=["quinn.__init__"],
    test_suite="tests",
    install_requires=["numpy", "scipy", "matplotlib", "torch", "ucimlrepo"],
    # setup_requires=['setuptools'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Natural Language :: English",
    ],
)
