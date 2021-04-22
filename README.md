

<!--
[![Build Status](https://travis-ci.org/ECSHackWeek/impedance.py.svg?branch=master)](https://travis-ci.org/ECSHackWeek/impedance.py)

[![Coverage Status](https://coveralls.io/repos/github/ECSHackWeek/impedance.py/badge.svg?branch=master)](https://coveralls.io/github/ECSHackWeek/impedance.py?branch=master)

[![Documentation Status](https://readthedocs.org/projects/impedancepy/badge/?version=latest)](https://impedancepy.readthedocs.io/en/latest/?badge=latest)
-->


Ampere - Advanced Model Package for ElectRochemical Experiments
------------

`Ampere` is a Python module for working with battery models.

Using a [scikit-learn-like API](https://arxiv.org/abs/1309.0238), we hope to make visualizing, fitting, and analyzing impedance spectra more intuitive and reproducible.

<i>Ampere is currently in the alpha phase and new features are rapidly being added.</i>
If you have a feature request or find a bug, please feel free to [file an issue](https://github.com/nealde/Ampere/issues) or, better yet, make the code improvements and [submit a pull request](https://help.github.com/articles/creating-a-pull-request-from-a-fork/)! The goal is to build an open-source tool that the entire electrochemical community can use and improve

Ampere currently provides:
- A simple API for fitting, predicting, and plotting discharge curves
- A simple API for generating data, or fitting with arbitrary charge / discharge patterns.


## Installation
### Dependencies

Ampere requires:

- Python (>=3.5)
- SciPy (>=1.0)
- NumPy (>=1.14)
- Matplotlib (>=2.0)
- Cython (>=0.29)


Several example notebooks are provided in the examples/ directory. Opening these will require Jupyter notebook or Jupyter lab.

### User Installation

The easiest way to install Ampere is using pip:

`pip install ampere`


However, it depends on Cython and Microsoft c++ libraries in order to install (on windows). Those should be added as follows:

`pip install --upgrade cython setuptools`

follow [these instructions](https://docs.microsoft.com/en-us/answers/questions/136595/error-microsoft-visual-c-140-or-greater-is-require.html) to install the proper c++ libraries using Microsoft tools.

That may or may not work, depending upon your system. An alternative method of installation that works is:

`git clone https://github.com/nealde/ampere`

I've recently added the Cython-generated c files back to the repo, so it may be as simple as:

`cd ampere`
`python setup.py install`

However, if that doesn't work, the following will rebuild the files:

`cd ampere/models/P2D`

`python setup.py build_ext --inplace`

`cd ../SPM`

`python setup.py build_ext --inplace`

This will build the local C code that is needed by the main compiler.  Then, you can cd back up to the main folder and

`python setup.py install`

That will typically work.  I'm still working on getting pip installation working, and it will likely require some package modifications,
following SKLearn as a guide.

## Examples and Documentation

Examples and documentation will be provided after my Defense, which is set for the end of May.

### On the Horizon

- Currently, all models are solved with Finite Difference discretization.  I would love to use some higher order spatial discretizations.
- Currently, the results have not been verified with external models. That is still on the to-do list, and to incorporate those values into the test suite would be excellent.
- Some of my published work regarding surrogate models for solving and fitting will be implemented once they are fully fleshed out.

- Add ability to serialize / deserialize models from disk, to save the result of an optimization
- add ability to have custom Up / Un functions for different battery chemistries
- add documentation / fix docstrings to be accurate
- add Latex equations and node spacings