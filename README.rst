========
powerbox
========
.. image:: https://coveralls.io/repos/github/steven-murray/powerbox/badge.svg?branch=master
    :target: https://coveralls.io/github/steven-murray/powerbox?branch=master
.. image:: https://img.shields.io/pypi/v/powerbox.svg
    :target: https://pypi.python.org/pypi/powerbox
.. image:: https://travis-ci.org/steven-murray/powerbox.svg?branch=master
    :target: https://travis-ci.org/steven-murray/powerbox

**Make arbitrarily structured, arbitrary-dimension boxes.**

`powerbox` is a pure-python code for creating density grids (or boxes) that have an arbitrary two-point distribution
(i.e. power spectrum). Primary motivations for creating the code were the simple creation of lognormal mock galaxy
distributions, but the methodology can be used for other applications.

Features
--------
* Works in any number of dimensions.
* Really simple.
* Arbitrary isotropic power-spectra.
* Create Gaussian or Log-Normal fields
* Create discrete samples following the field, assuming it describes an over-density.

Installation
------------
Clone/Download then ``python setup.py install``. Or just ``pip install powerbox``.

Basic Usage
-----------
There are two useful classes: the basic ``PowerBox`` and one for log-normal fields: ``LogNormalPowerBox``.
You can import them like

.. code:: python

    from powerbox import PowerBox, LogNormalPowerBox

Once imported, to see all the options, just use `help`:

.. code:: python

    help(PowerBox)

For a basic 2D Gaussian field with a power-law power-spectrum, one can use the following:

.. code:: python

    pb = PowerBox(N=512,                     # Number of grid-points in the box
                  dim=2,                     # 2D box
                  pk = lambda k: 0.1*k**-2., # The power-spectrum
                  boxlength = 1.0)           # Size of the box (sets the units of k in pk)
    import matplotlib.pyplot as plt
    plt.imshow(pb.delta_x)

Other attributes of the box can be accessed also -- check them out with tab completion in an interpreter!
The ``LogNormalPowerBox`` class is called in exactly the same way, but the resulting field has a log-normal pdf with the
same power spectrum.


TODO
----
* At this point, log-normal transforms are done by back-and-forward FFTs on the grid, which could be slow for higher
  dimensions. Soon I will implement a more efficient way of doing this using numerical Hankel transforms.
* Some more tests might be nice.