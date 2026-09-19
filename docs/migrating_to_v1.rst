Migrating from pre-v1 to v1
===========================

This page lists, in practical terms, what you need to change in code written for
``powerbox`` 0.x (up to 0.9.0) to run it on v1. Every "before" and "after" snippet below
was run against the corresponding version.

In the snippets, ``pk`` is your own power-spectrum function, ``pb`` is a ``PowerBox`` (or
``LogNormalPowerBox``), and ``np`` is NumPy.

.. contents::
   :local:
   :depth: 1

How to approach the upgrade
---------------------------

1. Fix the **hard breaks** (code that now raises): :ref:`mig-constructor`,
   :ref:`mig-attributes`, :ref:`mig-immutable`, :ref:`mig-discrete`, :ref:`mig-get-power`
   and :ref:`mig-dft`.
2. Read :ref:`mig-silent`. These are changes that raise nothing but change your numbers
   (or the shapes of arrays), so they will not be caught by "does it run?".
3. Run your code with deprecation warnings turned into errors, to find everything that
   still works but is scheduled for removal in v1.2::

      python -W error::DeprecationWarning your_script.py

Requirements
------------

v1 needs Python >= 3.11 (as before) and NumPy >= 2.2. ``attrs`` is a new runtime
dependency, and ``scipy``, which earlier versions imported without declaring, is now declared
too; both are installed automatically. ``pyfftw`` remains optional, and JAX support (see
:doc:`demos/jax`) is an optional extra that you can ignore for the purposes of migrating.

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Pre-v1
     - v1
   * - ``PowerBox(100, pk, ...)`` (positional)
     - ``PowerBox(shape=..., pk=pk, ...)`` (keyword-only)
   * - ``N=128``, ``dim=3``
     - ``shape=(128, 128, 128)``
   * - ``boxlength=100.0``
     - ``size=(100.0, 100.0, 100.0)``
   * - ``pb.boxlength``, ``pb.L``
     - ``pb.size`` (tuple)
   * - ``pb.N``
     - ``pb.shape`` (tuple)
   * - ``pb.V``, ``pb.Ntot``
     - ``pb.volume``, ``pb.total_ncells``
   * - ``pb.x``, ``pb.dx`` (single array / float)
     - ``pb.x``, ``pb.dx`` (tuple, one entry per axis)
   * - ``pb.kvec``, ``pb.k()``, ``pb.delta_k()`` (full spectrum)
     - Half spectrum: last axis has ``shape[-1] // 2 + 1`` entries
   * - ``pb.attr = value``
     - Boxes are frozen: build a new one
   * - ``create_discrete_sample(store_pos=True)``
     - Removed: use the return value
   * - ``p, k, var, nsamples = get_power(...)``
     - ``result = get_power(...)``, a :class:`powerbox.tools.PowerSpectrum`
   * - ``get_power(..., bin_ave=..., return_sumweights=...)``
     - Removed
   * - ``dft.fft(..., left_edge=e)``
     - ``dft.fft(..., x0=e)``, and the *default* changed
   * - ``dft.fft(..., ret_cubegrid=True)``
     - Removed: compute ``|k|`` yourself

.. _mig-constructor:

Constructing a box
------------------

``PowerBox`` and ``LogNormalPowerBox`` now take **keyword arguments only**, and describe
the grid with ``shape`` and ``size`` instead of ``N``, ``dim`` and ``boxlength``.

.. code-block:: python

   # before
   pb = PowerBox(128, pk, dim=3, boxlength=200.0, seed=42)

   # after
   pb = PowerBox(shape=(128, 128, 128), pk=pk, size=(200.0, 200.0, 200.0), seed=42)

* ``shape`` and ``size`` must be **sequences** with one entry per axis. Unlike the old
  ``N`` and ``boxlength``, a bare number is *not* accepted for either
  (``shape=128`` and ``size=200.0`` raise ``TypeError``). Use ``(128,) * 3`` and
  ``(200.0,) * 3`` to build a cubic box concisely.
* The number of dimensions is now the length of ``shape``. You no longer need ``dim``,
  though passing it is still allowed and is checked against ``len(shape)``.
* If you omit ``size`` it defaults to a unit box, as ``boxlength=1.0`` did.
* Non-cubic boxes are new: give each axis its own entry (see :doc:`demos/cuboid_boxes`).

The old spelling still works for now, so you can migrate in two steps.
``N=``, ``dim=`` and ``boxlength=`` are accepted (``N`` and ``boxlength`` emit a
``DeprecationWarning`` and are expanded to every axis) **but only as keywords**, and they
will be removed in v1.2.

.. code-block:: python

   # before
   pb = PowerBox(N=128, pk=pk, dim=3, boxlength=200.0)

   # after (still works in v1, with a DeprecationWarning; removed in v1.2)
   pb = PowerBox(N=128, pk=pk, dim=3, boxlength=200.0)

.. _mig-attributes:

Attributes and array layouts
----------------------------

Renamed
^^^^^^^

* ``pb.boxlength`` and ``pb.L`` still work but warn; use ``pb.size``. Note that they now
  return a tuple, not a single number.
* ``pb.V`` and ``pb.Ntot`` still work but warn; use ``pb.volume`` and ``pb.total_ncells``.
* ``pb.N`` is ``None`` on a box constructed with ``shape=``. Use ``pb.shape``.

Everything is per-axis
^^^^^^^^^^^^^^^^^^^^^^

``pb.size``, ``pb.shape``, ``pb.dx`` and ``pb.x`` are now tuples with one entry per axis.
The one to watch is ``pb.x``: it used to be a single 1-D array shared by all axes, so
indexing it gave a *number*; now indexing it gives a whole *axis*.

.. code-block:: python

   # before
   x = pb.x                 # 1-D array
   left_edge = pb.x[0]      # a number
   dx = pb.dx               # a float

   # after
   x, y = pb.x              # one 1-D array per axis (for a 2-D box)
   left_edge = pb.x[0][0]   # a number
   dx = pb.dx[0]            # a float (one entry per axis)

Half-spectrum outputs
^^^^^^^^^^^^^^^^^^^^^

Fields generated by ``PowerBox`` are real, so v1 only stores the non-redundant half of
their Fourier spectrum: the last axis holds the non-negative frequencies only, with
``shape[-1] // 2 + 1`` entries. Every spectrum-space quantity changed accordingly:

* ``pb.kvec``, a tuple in which the last array has only non-negative frequencies,
* ``pb.k()``, ``pb.power_array()``, ``pb.gauss_hermitian()`` and ``pb.delta_k()``,
* ``LogNormalPowerBox.gaussian_power_array()``.

Real-space arrays (``pb.delta_x()``, ``pb.r``, ``LogNormalPowerBox.correlation_array()``)
keep their full ``shape``.

.. code-block:: python

   pb = PowerBox(shape=(16, 16), pk=pk)
   pb.delta_x().shape    # (16, 16), as before
   pb.k().shape          # (16, 9), was (16, 16)
   pb.delta_k().shape    # (16, 9), was (16, 16)

Code that combines ``pb.k()`` (or ``pb.delta_k()``) elementwise with ``pb.delta_x()`` will
fail on the shape mismatch. If you need a full-grid spectrum, Fourier transform the field
with :func:`powerbox.dft.fft`. You can also now feed a half spectrum back in with
``pb.delta_x(delta_k=...)``.

.. _mig-immutable:

Boxes are immutable
-------------------

``PowerBox`` instances are frozen. Assigning to an attribute raises
``attrs.exceptions.FrozenInstanceError``. (Previously assignment succeeded but silently left
derived values, such as ``V`` and ``dx``, stale.) Make a new box instead:

.. code-block:: python

   # before
   pb.seed = 7
   pb.ensure_physical = True

   # after
   pb = PowerBox(shape=(64, 64), pk=pk, seed=7, ensure_physical=True)

To derive a variant from an existing box, use ``attrs.evolve``:

.. code-block:: python

   import attrs

   pb2 = attrs.evolve(pb, seed=8)

``attrs.evolve`` cannot change ``shape`` or ``size`` on a box that was itself created with
the deprecated ``N`` or ``boxlength`` (it raises ``ValueError``), so build boxes with
``shape`` and ``size`` if you want to do this.

.. _mig-discrete:

Discrete samples
----------------

``create_discrete_sample`` no longer has a ``store_pos`` argument, and the box no longer
records ``tracer_positions`` or ``n_per_cell``. Use the return value.

.. code-block:: python

   # before
   pos = pb.create_discrete_sample(nbar=0.5, store_pos=True)
   pos = pb.tracer_positions

   # after
   pos = pb.create_discrete_sample(nbar=0.5)

The remaining arguments (``nbar``, ``randomise_in_cell``, ``min_at_zero``, ``delta_x``) are
unchanged. When you pass the result to ``get_power``, ``N`` may be the ``pb.shape`` tuple.

.. _mig-get-power:

``get_power`` and the ``PowerSpectrum`` result
----------------------------------------------

``get_power`` now returns a :class:`powerbox.tools.PowerSpectrum` object instead of a tuple.
It cannot be indexed or unpacked; read its attributes.

.. code-block:: python

   # before
   p, k, var, nsamples = get_power(
       pb.delta_x(), pb.boxlength, get_variance=True, bins_upto_boxlen=True
   )

   # after
   result = get_power(pb.delta_x(), pb.size, get_variance=True, bins_upto_boxlen=True)
   p, k, var, nsamples = result.power, result.bin_avg, result.variance, result.nsamples

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pre-v1 (position in tuple)
     - v1 attribute
   * - ``[0]`` power
     - ``result.power``
   * - ``[1]`` ``k`` (with ``bin_ave=True``, the default)
     - ``result.bin_avg`` (weighted mean ``k`` of each bin)
   * - ``[1]`` ``k`` (with ``bin_ave=False``)
     - ``result.bin_edges`` (length ``n_bins + 1``)
   * - ``[2]`` variance (``None`` unless ``get_variance=True``)
     - ``result.variance`` (``None`` unless ``get_variance=True``)
   * - ``[3]`` sum of weights per bin
     - ``result.nsamples``
   * - ``[4]`` unaveraged ``k`` arrays (only when ``res_ndim < dim``)
     - ``result.k_unbinned`` (``None`` otherwise)

Notes:

* The ``bin_ave`` and ``return_sumweights`` arguments are removed (the latter had no effect
  in 0.x). Both the bin edges and the mean-``k`` per bin are always available, and the
  number of samples is always in ``nsamples``.
* ``result.bin_centres`` is new: the midpoint (linear, or geometric when ``log_bins=True``)
  of each pair of edges. It is *not* the same as ``bin_avg``, which is what the old ``k``
  was.
* ``res_ndim=0`` used to return ``[P, None, None, None, freq]``. It now returns a
  ``PowerSpectrum`` whose ``power`` is the n-dimensional power, ``k_unbinned`` holds
  the frequency arrays, and the bin attributes are empty.
* With ``res_ndim`` between 1 and ``dim - 1``, ``result.power`` has shape
  ``(n_bins, *remaining_dims)``, as before.
* ``get_power`` still takes ``boxlength`` positionally, and a per-axis sequence is accepted
  as before. Pass ``pb.size`` (``pb.boxlength`` still works but is deprecated).

.. _mig-dft:

``powerbox.dft``
----------------

``left_edge`` is now ``x0``, and ``ret_cubegrid`` is gone. **The default phase convention
also changed**, which alters the complex values returned even when you pass no new
arguments (the magnitudes are unaffected, so power spectra are not).

* Old default: the input is treated as being centred on the origin, so its first sample sits
  at ``-L/2`` (``left_edge=-L/2``).
* New default: ``x0=0``, i.e. the first sample sits at 0, exactly like ``numpy.fft``.

To reproduce the old default, pass ``x0=-L/2`` (a scalar applies to every axis, or give one
value per axis). An explicit old ``left_edge=e`` for ``fft`` maps directly to ``x0=e``.

.. code-block:: python

   # before
   ft, freq = dft.fft(X, L=L, a=0, b=2 * np.pi)

   # after: same values as before
   ft, freq = dft.fft(X, L=L, a=0, b=2 * np.pi, x0=-L / 2)

For ``ifft``, the old default corresponds to ``x0=-L/2`` where ``L`` is the *real-space*
extent of the output. If you specified ``Lk`` rather than ``L``, that extent is
``N * 2 * pi / (b * Lk)`` per axis. An explicit ``left_edge`` passed to the old ``ifft``
was a k-space quantity; ``x0`` is a real-space quantity, so there is no one-to-one
translation. Work out the phase you want from the real-space coordinate of the first sample.

.. code-block:: python

   # before
   x, xgrid = dft.ifft(Fk, L=L, a=0, b=2 * np.pi)

   # after: same values as before (for even N; see below)
   x, xgrid = dft.ifft(Fk, L=L, a=0, b=2 * np.pi, x0=-L / 2)

``ret_cubegrid`` is removed. Compute the magnitude grid from the returned frequencies:

.. code-block:: python

   # before
   ft, freq, kmag = dft.fft(X, L=L, ret_cubegrid=True)

   # after
   ft, freq = dft.fft(X, L=L)
   kmag = np.sqrt(sum(k**2 for k in np.meshgrid(*freq, indexing="ij")))

Other changes:

* Because ``left_edge`` and ``ret_cubegrid`` were removed, the positional order of the
  later arguments changed. Pass everything after ``b`` by keyword.
* ``dft.ifft`` previously gave wrong results for **odd** grid lengths (a forward/inverse
  round trip did not recover the input). That is fixed, so odd-``N`` inverse transforms
  change. Even-``N`` transforms only differ by the default-phase change above.
* New: :func:`powerbox.dft.irfft` and :func:`powerbox.dft.rfftfreq`, used for the
  half-spectrum layout.

.. _mig-silent:

Changes that don't raise but do change results
----------------------------------------------

Random numbers and ``seed``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Random draws now come from a per-box ``numpy.random.Generator`` (``pb.rng``), created from
``seed``, rather than from NumPy's global ``np.random`` state. Consequences:

* **The same seed gives different fields to pre-v1.** No option restores the old stream.
  Regenerate any stored fixtures or regression targets.
* **Repeated calls on one box now differ.** Pre-v1, a box with a ``seed`` reseeded the global
  generator on every call, so ``pb.delta_x()`` returned the *same* field each time. In v1
  it advances the box's generator, so successive calls give different fields, and a fresh
  box with the same seed reproduces the same *sequence*. If you relied on getting the same
  field twice from one box, generate it once and reuse it, or pass it in
  (``pb.create_discrete_sample(nbar=..., delta_x=field)``).
* ``seed=0`` is now honoured. Before, it was treated as "no seed".
* ``seed`` no longer touches ``np.random``; code that seeded the global state through
  ``PowerBox`` will no longer see that side effect.

.. code-block:: python

   # before: same field from every call
   pb = PowerBox(N=64, pk=pk, seed=1)
   a, b = pb.delta_x(), pb.delta_x()      # identical

   # after: same field only if you keep it
   pb = PowerBox(shape=(64, 64), pk=pk, seed=1)
   a = pb.delta_x()
   b = a                                   # reuse

Power recovered from fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A bug in the NumPy backend halved the variance of the modes lying on the self-conjugate
surfaces of the Hermitian spectrum. It affected all ``PowerBox`` and ``LogNormalPowerBox``
fields, biasing the recovered power low by a few percent overall and by about 10% in the
lowest ``|k|`` modes. That is fixed: generated fields now have the requested power, so
power spectra measured from them will shift upwards relative to pre-v1. If you tuned
anything (amplitudes, tolerances) against the old output, re-check it.

``LogNormalPowerBox``
^^^^^^^^^^^^^^^^^^^^^

* Fields depended on the Fourier convention ``(a, b)`` in a way they should not. Boxes with
  ``(a, b)`` other than the defaults ``(1, 1)`` or the NumPy convention ``(0, 2*pi)``
  now give different (correct) fields.
* Spectra that cannot be realised as a lognormal field on your grid no longer produce a
  silently wrong field. A slight violation emits a ``UserWarning`` and zeroes the offending
  modes; a material one raises ``ValueError``. If you hit the error, reduce the amplitude
  of ``pk``, increase ``size``, coarsen ``shape``, or use a ``pk`` that falls off faster
  at high ``k``.
* ``delta_x()`` now uses the theoretical variance of the underlying Gaussian field in the
  exponent, rather than the sample variance of each realisation, so the values of a given
  realisation differ slightly from pre-v1.

Default ``bins_upto_boxlen``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unchanged from 0.9: ``get_power`` warns (``FutureWarning``) unless you pass
``bins_upto_boxlen`` explicitly. Set it to keep your results stable and silence the warning.

Worked example
--------------

.. code-block:: python

   # before
   import numpy as np
   from powerbox import PowerBox, get_power

   pb = PowerBox(N=128, pk=lambda k: 0.1 * k**-2.0, dim=2, boxlength=100.0, seed=1234)
   field = pb.delta_x()
   p, k, _, _ = get_power(field, pb.boxlength, bins_upto_boxlen=True)

.. code-block:: python

   # after
   import numpy as np
   from powerbox import PowerBox, get_power

   pb = PowerBox(shape=(128, 128), pk=lambda k: 0.1 * k**-2.0, size=(100.0, 100.0), seed=1234)
   field = pb.delta_x()
   result = get_power(field, pb.size, bins_upto_boxlen=True)
   p, k = result.power, result.bin_avg

What's new
----------

None of the following requires action, but you may want to use them:

* Non-cubic boxes with per-axis ``shape`` and ``size`` (:doc:`demos/cuboid_boxes`).
* An optional JAX implementation in ``powerbox.jax`` (:doc:`demos/jax`).
* ``pb.variance``, the field variance implied by the power spectrum, and
  ``pb.synthesis_norm``.
* ``pb.delta_x(delta_k=...)`` to synthesise a field from a spectrum you supply.

Timeline
--------

The deprecated ``N`` and ``boxlength`` constructor arguments, and the ``boxlength``, ``L``,
``V`` and ``Ntot`` attributes, will be removed in v1.2.
