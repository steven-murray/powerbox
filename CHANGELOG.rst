Changelog
=========

v0.2.3 [11 Jan 2017]
~~~~~~~~~~~~~~~~~~~~
**Enhancements**
- Improved estimation of power (in ``get_power``) for lowest k bin.

v0.2.2 [11 Jan 2017]
~~~~~~~~~~~~~~~~~~~~
**Bugfixes**
- Fixed a bug in which the output power spectrum was a factor of sqrt(2) off in normalisation

v0.2.1 [10 Jan 2017]
~~~~~~~~~~~~~~~~~~~~
**Bugfixes**
- Fixed output of ``create_discrete_sample`` when not randomising positions.

**Enhancements**
- New option to set bounds of discrete particles to (0, boxlength) rather than centring at 0.

v0.2.0 [10 Jan 2017]
~~~~~~~~~~~~~~~~~~~~
**Features**
- New ``LogNormalPowerBox`` class for creating log-normal fields

**Enhancements**
- Restructuring of code for more flexibility after creation. Now requires ``cached_property`` package.

v0.1.0 [27 Oct 2016]
~~~~~~~~~~~~~~~~~~~~
First working version. Only Gaussian fields working.
