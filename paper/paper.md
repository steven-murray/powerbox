---
title: 'powerbox: A Python package for creating structured fields with isotropic power spectra'
tags:
  - Python
  - astronomy
  - power spectrum
  - signal analysis
authors:
  - name: Steven G. Murray
    orcid: 0000-0003-3059-3823
    affiliation: 1, 2
affiliations:
 - name: International Centre for Radio Astronomy Research (ICRAR), Curtin University,  Bentley, WA 6102, Australia
   index: 1
 - name: ARC Centre of Excellence for All-Sky Astrophysics in 3 Dimensions (ASTRO 3D)
   index: 2
date: 19 July 2018
bibliography: paper.bib
---

# Summary

The power spectrum is a cornerstone of both signal analysis and spatial statistics,
encoding the variance of a signal or field on different scales.
Its common usage is in no small part attributable to the fact that it is a *full*
description of a purely Gaussian process -- for such statistical processes, no information
is contained in higher-order statistics. The prevalence of such processes (or
close approximations to them) in physical systems serves to justify the popularity
of the power spectrum as a key descriptive statistic in various physical sciences,
eg. cosmology [@Peacock1999] and fluid mechanics [@Monin2007]. It furthermore
readily avails itself to efficient numerical evaluation, being the  absolute square of
the Fourier Transform.

Another feature of many approximate physical systems, especially those already mentioned,
is that they are both homogeneous and isotropic (at least in some local sample).
In this case, the *n*-dimensional power spectrum may be losslessly compressed into a single
dimension, which is radial in Fourier-space. Such processes approximately describe for example
the over-density field of the early Universe and locally isotropic turbulent flows.
Thus it is of great use to have a numerical code which simplifies the dual operations of; (i)
producing random homogeneous/isotropic fields (of arbitrary dimensionality) consistent with a
given 1D radial power spectrum, and (ii) determination of the 1D radial power spectrum of
random fields (or a sample of tracers of that field). ``powerbox`` exists to perform these
duals tasks with both simplicity and efficiency.

Performing the first of these tasks is especially non-trivial. While the power spectrum
can be evaluated on any field (though it may not fully describe the given field), the
precise machinery for *creating* a field from a given power spectrum depends on the
probability density function (PDF) of the process itself. The machinery for creating a
Gaussian field is well-known. However, other PDF's -- especially those that are positively
bounded -- are extremely useful for describing such physical entities as density fields.
In these cases, the *log-normal* PDF has become a standard approximation [@Coles1991],
and ``powerbox`` makes a point of supporting the machinery for generating log-normal
fields [@Beutler2011] for this purpose. Indeed, ``powerbox`` is *primarily* geared
towards supporting cosmological applications, such as measuring and and producing
samples of galaxy positions in a log-normal density field (while account for standard
effects such as shot-noise and standard normalisation conventions). It is nevertheless
flexible enough to support research in any field (with its own particular conventions)
that is based on the homogeneous and isotropic power spectrum.

 ``Powerbox`` is a pure-Python package devoted to the simple and efficient solution
 of the previous considerations. As the most popular language for astronomy, Python
 is the natural language of choice for ``powerbox``, with its focus on cosmological
 applications, and it also provides for great ease-of-use and extensibility. As an
 example of the former, all functions/classes within ``powerbox`` are able to work
 in arbitrary numbers of dimensions (memory permitting), simply by setting a single
 parameter *n*. As an
 example of the latter, the class-based structure of the field-generator may be used
 to extend the generation to fields with PDF's other than either Gaussian or log-normal
 (indeed, the log-normal class is itself sub-classed from the Gaussian one).
 ``powerbox`` does not sacrifice efficiency for its high-level interface. By default, the
 underlying FFT's are performed by ``numpy``, which uses underlying fast C code. In addition,
 if the ``pyFFTW`` package is installed, ``powerbox`` will seamlessly switch to using its
 optimized C code for up to double the efficiency. It is also written with an eye for
 conserving memory, which is important for the often very large fields that may be required.

``Powerbox`` was written due to research-demand, and as such it is highly likely to be suited
to the requirements of research of a similar nature. Furthermore, as previously stated,
every effort has been made to sufficiently generalize its scope to be of use in related
fields of research. It has already been instrumental in several publications [@Murray2017; @Wolz2018],
and we hope it will be a useful tool for approximate theoretical simulations by many others.

# Acknowledgements

The author acknowledges helpful discussions and contributions from Cathryn Trott, Chris Jordan
and Laura Wolz during the initial development of this project.
Parts of this research were supported by the Australian Research Council Centre of Excellence
for All Sky Astrophysics in 3 Dimensions (ASTRO 3D), through project number CE170100013

# References
