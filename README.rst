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

``powerbox`` is a pure-python code for creating density grids (or boxes) that have an arbitrary two-point distribution
(i.e. power spectrum). Primary motivations for creating the code were the simple creation of lognormal mock galaxy
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

Just to be clear, for a more "realistic" example: a log-normal box (let's say with 3 dimensions) with a power-spectrum
given by cosmological density perturbations, can be created like this (this also uses the code
`hmf <https://github.com/steven-murray/hmf>`_):

.. code:: python

    from hmf import MassFunction
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    # Set up a MassFunction instance to access its cosmological power-spectrum
    mf = MassFunction(z=0)

    # Generate a callable function that returns the cosmological power spectrum.
    spl = spline(np.log(mf.k),np.log(mf.power),k=2)
    power = lambda k : np.exp(spl(np.log(k))

    # Create the power-box instance. The boxlength is in inverse units of the k of which pk is a function, i.e.
    # Mpc/h in this case.
    pb = LogNormalPowerBox(N=512, dim=3, pk = power, boxlength= 100.)

Now we can make a plot of a slice of the density field:

.. code:: python

    plt.imshow(pb.delta_x,extent=(0,100,0,100))
    plt.colorbar()
    plt.show()

And we can also compare the power-spectrum of the output field to the input power:

.. code:: python

    from powerbox import get_power

    p_k, kbins = get_power(pb.delta_x,pb.boxlength)
    plt.plot(mf.k,mf.power,label="Input Power")
    plt.plot(kbins,p_k,label="Sampled Power')
    plt.legend()
    plt.show()

Furthermore, we can sample a set of discrete particles on the field, and plot their power spectrum

.. code:: python

    particles = pb.create_discrete_sample(nbar=1.0)
    p_k_sample, kbins_sample = get_power(particles, pb.boxlength,N=pb.N)

    plt.plot(mf.k,mf.power,label="Input Power")
    plt.plot(kbins_sample,p_k_sample,label="Sampled Power Discrete")
    plt.legend()
    plt.show()

TODO
----
* At this point, log-normal transforms are done by back-and-forward FFTs on the grid, which could be slow for higher
  dimensions. Soon I will implement a more efficient way of doing this using numerical Hankel transforms.
* Some more tests might be nice.