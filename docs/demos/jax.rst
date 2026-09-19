JAX-backed workflows
====================

``powerbox`` provides an experimental JAX-backed namespace at ``powerbox.jax``. It
mirrors the supported public API while keeping the standard NumPy implementation
available at the top level.

The main differences in milestone 1 are:

1. JAX field generators use explicit PRNG keys.
2. ``create_discrete_sample`` is not implemented yet.
3. Interpolation-based power estimation is not implemented yet.
4. ``delta_x()`` uses a heuristic ``usejit`` policy by default.

When to use ``usejit``
---------------------

``usejit=True`` is usually best when you will generate multiple realizations with the
same shape, because JAX can amortize compilation and run faster afterward. Use
``usejit=False`` if you only need a single quick evaluation, or if you are debugging
and want the simplest eager execution path.

If you do not pass ``usejit``, ``powerbox.jax`` chooses a default based on ``Ntot``.
That default is intended to be a sensible starting point, not a guaranteed optimal
choice for every machine or workload.

Gaussian field generation
-------------------------

.. code-block:: python

   import jax
   import powerbox.jax as jpb

   key = jax.random.key(0)
   pb = jpb.PowerBox(
       (128, 192),
       pk=lambda k: (1 + k) ** -2.0,
       dim=2,
       boxlength=(200.0, 600.0),
       key=key,
   )

   field = pb.delta_x()
   result = jpb.get_power(jpb.fftshift(field), pb.boxlength, bins_upto_boxlen=True)

JAX power-spectrum estimation
-----------------------------

.. code-block:: python

   import jax
   import powerbox.jax as jpb

   key = jax.random.key(2)
   pb = jpb.PowerBox(
       (128, 192),
       pk=lambda k: (1 + k) ** -2.0,
       dim=2,
       boxlength=(200.0, 600.0),
       key=key,
   )

   field = pb.delta_x()
   power = jpb.get_power(field, pb.boxlength, bins_upto_boxlen=True)
   print(power.bin_centres)
   print(power.power)

Lognormal field generation
--------------------------

.. code-block:: python

   import jax
   import powerbox.jax as jpb

   pb = jpb.LogNormalPowerBox(
       128,
       pk=lambda k: (1 + k) ** -2.0,
       dim=2,
       boxlength=200.0,
       key=jax.random.key(1),
   )

   lognormal_field = pb.delta_x()
