========
powerbox
========

**Make arbitrarily structured, arbitrary-dimension boxes.**

`powerbox` is a pure-python code for creating density grids (or boxes) that have an arbitrary two-point distribution
(i.e. power spectrum). Primary motivations for creating the code were the simple creation of lognormal mock galaxy
distributions, but the methodology can be used for other applications.

Features
--------
* Works in any number of dimensions
* Arbitrary isotropic power-spectra.
* Arbitrary 1-point density distributions.
* Really simple.

Installation
------------
Clone/Download then ``python setup.py install``.

Basic Usage
-----------
At this point, there's just a single class that is useful: `PowerBox`. You can import it like

>>> from powerbox import PowerBox

Once imported, to see all the options, just use `help`:

>>> help(PowerBox)

For a basic 2D Gaussian field with a power-law power-spectrum, one can use the following:

>>> pb = PowerBox(N=512,  ## Number of grid-points in the box
                  dim=2,  ## 2D box
                  pk = lambda k: 0.1*k**-2., ## The power-spectrum
                  boxlength = 1.0 ## Size of the box (sets the units of k in pk)
                  )
>>> import matplotlib.pyplot as plt
>>> plt.imshow(pb.delta_x)

Other attributes of the box can be accessed also -- check them out with tab completion in an interpreter!