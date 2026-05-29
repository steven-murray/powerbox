Cuboid boxes
============

As of v1.0, ``PowerBox`` and ``LogNormalPowerBox`` can be constructed with per-axis grid
sizes and side lengths. This is useful when one axis of the target volume needs a
different physical extent or resolution from the others.

Constructing a non-cubic Gaussian field
---------------------------------------

Pass tuples for ``N`` and ``boxlength`` with one entry per dimension:

.. code-block:: python

   import matplotlib.pyplot as plt

   from powerbox import PowerBox

   pb = PowerBox(
       N=(128, 192),
       dim=2,
       boxlength=(200.0, 600.0),
       pk=lambda k: (1 + k) ** -2.0,
       seed=1234,
   )

   field = pb.delta_x()
   x, y = pb.x

   plt.imshow(
       field,
       extent=(y[0], y[-1], x[0], x[-1]),
       aspect="auto",
       origin="lower",
   )
   plt.xlabel("y")
   plt.ylabel("x")
   plt.title("Gaussian cuboid field")

The same constructor pattern works for :class:`powerbox.LogNormalPowerBox`.

Checking the recovered power spectrum
-------------------------------------

The resulting field can be passed straight into :func:`powerbox.get_power` with the same
per-axis ``boxlength`` tuple:

.. code-block:: python

   import matplotlib.pyplot as plt

   from powerbox import get_power

   result = get_power(field, pb.boxlength, bins_upto_boxlen=True)

   plt.loglog(result.bin_avg[1:], result.power[1:], label="measured")
   plt.loglog(result.bin_avg[1:], (1 + result.bin_avg[1:]) ** -2.0, label="input")
   plt.xlabel("k")
   plt.ylabel("P(k)")
   plt.legend()

Working with discrete samples
-----------------------------

Discrete sampling uses the same geometry:

.. code-block:: python

   sample = pb.create_discrete_sample(nbar=1e-2, min_at_zero=True)
   discrete_result = get_power(sample, pb.boxlength, N=pb.N, bins_upto_boxlen=True)

When scalar values are passed for ``N`` and ``boxlength``, existing workflows are
unchanged.
