========
powerbox
========
.. image:: https://img.shields.io/pypi/v/powerbox.svg
    :target: https://pypi.python.org/pypi/powerbox
.. image:: https://travis-ci.org/steven-murray/powerbox.svg?branch=master
    :target: https://travis-ci.org/steven-murray/powerbox
.. image:: https://coveralls.io/repos/github/steven-murray/powerbox/badge.svg?branch=master
    :target: https://coveralls.io/github/steven-murray/powerbox?branch=master
.. image:: https://api.codacy.com/project/badge/Grade/5853411c78444a5a9c6ec4058c6dbda9
    :target: https://www.codacy.com/app/steven-murray/powerbox?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=steven-murray/powerbox&amp;utm_campaign=Badge_Grade
.. image:: https://zenodo.org/badge/72076717.svg
   :target: https://zenodo.org/badge/latestdoi/72076717
.. image:: http://joss.theoj.org/papers/10.21105/joss.00850/status.svg
   :target: https://doi.org/10.21105/joss.00850

**Make arbitrarily structured, arbitrary-dimension boxes and log-normal mocks.**

``powerbox`` is a pure-python code for creating density grids (or boxes) that have an
arbitrary two-point distribution (i.e. power spectrum). Primary motivations for creating
the code were the simple creation of log-normal mock galaxy distributions, but the
methodology can be used for other applications.

Features
--------
* Works in any number of dimensions.
* Really simple.
* Arbitrary isotropic power-spectra.
* Create Gaussian or Log-Normal fields
* Create discrete samples following the field, assuming it describes an over-density.
* Measure power spectra of output fields to ensure consistency.
* Seamlessly uses pyFFTW if available for ~double the speed.

Installation
------------
Simply ``pip install powerbox``. If you want ~2x speedup for large boxes, you can also
install ``pyfftw`` by doing ``pip install powerbox[all]``. If you are a conda user, you
may want to install ``numpy`` with conda first. If you want to develop ``powerbox``,
clone the repo and install with ``python -m pip install -e ".[dev]"``.

Acknowledgment
--------------
If you find ``powerbox`` useful in your research, please cite the Journal of Open Source Software paper at
https://doi.org/10.21105/joss.00850.

QuickLinks
----------
* Docs: https://powerbox.readthedocs.io
* Quickstart: http://powerbox.readthedocs.io/en/latest/demos/getting_started.html
