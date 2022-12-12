Changelog
=========

0.6.1
-----
**Bugfixes**

- Fix error when doing cross-power [Issue #5].

0.6.0
-----
**Features**

- New ``left_edge`` argument in fft/ifft which gives the ability to specify where
  the x- (or k-) co-ordinates are, in order to set appropriate phase information.
  NOTE: this changes the default behaviour of the function. While the forward and
  inverse transforms remain inverses by default, the phases are interpreted as
  having zero at the centre (for both transforms). See the phasing tutorial for
  more information.
- Fixed transpose issue caused by default behavior of ``numpy.meshgrid``, which
  led to broken correspondence between discrete sample of field and original
  field. See [Issue #15].

**Bugfixes**

- Make warning about pyFFTW slightly less obnoxious.


v0.5.7 [24 Oct 2018]
--------------------
**Enhancements**

- Added ability to use weights on k-modes in ``get_power``.

**Bugfixes**

- Fixed bug on using ``ignore_zero_mode`` introduced in v0.5.6
- Added tests for ``ignore_zero_mode``` and ``k_weights``

v0.5.6 [23 Oct 2018]
--------------------
**Enhancements**

- Added ``ignore_zero_mode`` parameter to ``get_power``.

**Bugfixes**

- Removed redundant ``seed`` parameter from ``create_discrete_sample()``.

v0.5.5 [19 July 2018]
---------------------
**Bugfixes**

- log_bins wasn't being passed through to angular_average correctly.

**Enhancements**

- ``angular_average()`` no longer requires coords to be passed as box of magnitudes.
- improved docs.
- fixed source divide by zero warning in PowerBox()

v0.5.4 [30 May 2018]
--------------------
**Enhancements**

- Added ability to do angular averaging in log-space bins
- When not all radial bins have co-ordinates in them, a more reasonable warning message is emitted.
- Removed redundant bincount call when only summing, not averaging (angularly).

**Bugfixes**

- Now properly deals with co-ordinates outside the bin range in angular_average (will only make a difference when bins
  is passed as a vector). Note that this has meant that by default the highest-valued co-ordinate in the box will *not*
  contribute to any bins any more.
- Fixed a bunch of tests in test_power which were using the wrong power index!

**Internals**

- Re-factored getting radial bins into _getbins() function.

v0.5.3 [22 May 2018]
--------------------
**Bugfixes**

- Fixed a bug introduced in v0.5.1 where using bin_ave=False in angular_average_nd would fail.

v0.5.2 [17 May 2018]
--------------------
**Enhancements**

- Added ability to calculate the variance of an angularly averaged quantity.
- Removed a redundant calculation of the bin weights in angular_average

**Internals**

- Updated version numbers of dev requirements.

v0.5.1 [4 May 2018]
-------------------
**Enhancements**

- Added ability to *not* have dimensionless power spectra from get_power.
- Also return linearly-spaced radial bin edges from angular_average_nd
- Python 3 compatibility

**Bugfixes**

- Fixed bug where field was modified in-place unexpectedly in angular_average
- Now correctly flattens weights before getting the field average in angular_average_nd

v0.5.0 [7 Nov 2017]
------------------~
**Features**

- Input boxes to get_power no longer need to have same length on every dimension.
- New angular_average_nd function to average over first n dimensions of an array.

**Enhancements**

- Huge (5x or so) speed-up for angular_average function (with resulting speedup for get_power).
- Huge memory reduction in fft/ifft routines, with potential loss of some speed (TODO: optimise)
- Better memory consumption in PowerBox classes, at the expense of an API change (cached properties no
  longer cached, or properties).
- Modified fftshift in dft to handle astropy Quantity objects (bit of a hack really)

**Bugfixes**

- Fixed issue where if the boxlength was passed as an integer (to fft/ifft), then incorrect results occurred.
- Fixed issue where incorrect first_edge assignment in get_power resulted in bad power spectrum. No longer require this arg.

v0.4.3 [29 March 2017]
----------------------
**Bugfixes**

- Fixed volume normalisation in get_power.

v0.4.2 [28 March 2017]
----------------------
**Features**

- Added ability to cross-correlate boxes in get_power.

v0.4.1
------
**Bugfixes**

- Fixed cubegrid return value for dft functions when input boxes have different sizes on each dimension.


v0.4.0
------
**Features**

- Added fft/ifft wrappers which consistently return fourier transforms with arbitrary Fourier conventions.
- Boxes now may be composed with arbitrary Fourier conventions.
- Documentation!

**Enhancements**

- New test to compare LogNormalPowerBox with standard PowerBox.
- New project structure to make for easier location of functions.
- Code quality improvements
- New tests, better coverage.

**Bugfixes**

- Fixed incorrect boxsize for an odd number of cells
- Ensure mean density is correct in LogNormalPowerBox

v0.3.2
------
**Bugfixes**

- Fixed bug in pyFFTW cache setting

v0.3.1
------
**Enhancements**

- New interface with pyFFTW to make fourier transforms ~twice as fast. No difference to the API.

v0.3.0
------
**Features**

- New functionality in ``get_power`` function to measure power-spectra of discrete samples.

**Enhancements**

- Added option to not store discrete positions in class (just return them)
- ``get_power`` now more streamlined and intuitive in its API

v0.2.3 [11 Jan 2017]
--------------------
**Enhancements**

- Improved estimation of power (in ``get_power``) for lowest k bin.

v0.2.2 [11 Jan 2017]
--------------------
**Bugfixes**

- Fixed a bug in which the output power spectrum was a factor of sqrt(2) off in normalisation

v0.2.1 [10 Jan 2017]
--------------------
**Bugfixes**

- Fixed output of ``create_discrete_sample`` when not randomising positions.

**Enhancements**

- New option to set bounds of discrete particles to (0, boxlength) rather than centring at 0.

v0.2.0 [10 Jan 2017]
--------------------
**Features**

- New ``LogNormalPowerBox`` class for creating log-normal fields

**Enhancements**

- Restructuring of code for more flexibility after creation. Now requires ``cached_property`` package.

v0.1.0 [27 Oct 2016]
--------------------
First working version. Only Gaussian fields working.
