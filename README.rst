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

``powerbox`` is a pure-python code for creating density grids (or boxes) that have an arbitrary two-point distribution
(i.e. power spectrum). Primary motivations for creating the code were the simple creation of log-normal mock galaxy
distributions, but the methodology can be used for other applications.

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
``powerbox`` only depends on ``numpy >= 1.6.2``, which will be installed automatically if ``powerbox`` is installed
using ``pip`` (see below). Furthermore, it has the optional dependency of ``pyfftw``, which if installed will offer
~2x performance increase in large fourier transforms. This will be seamlessly used if installed.

To install ``pyfftw``, simply do::

    pip install pyfftw

To install ``powerbox``, do::

    pip install powerbox

Alternatively, the bleeding-edge version from git can be installed with::

    pip install git+git://github.com/steven-murray/powerbox.git

Finally, for a development installation, download the source code and then run (in the top-level directory)::

    pip install -e .

Acknowledgment
--------------
If you find ``powerbox`` useful in your research, please cite the Journal of Open Source Software paper at
https://doi.org/10.21105/joss.00850.

QuickLinks
----------
* Docs: https://powerbox.readthedocs.io
* Quickstart: http://powerbox.readthedocs.io/en/latest/demos/getting_started.html