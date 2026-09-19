"""Test the JAX implementation of powerbox tools."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")
jpb = importlib.import_module("powerbox.jax")
jpb_powerbox = importlib.import_module("powerbox.jax.powerbox")
jdft = importlib.import_module("powerbox.jax.dft")
jtools = importlib.import_module("powerbox.jax.tools")
ndft = importlib.import_module("powerbox.dft")


def test_jax_tooling_validation_branches() -> None:
    with pytest.raises(ValueError, match="n_bins \\+ 1"):
        jtools.PowerSpectrum(
            power=jnp.ones(3),
            bin_edges=jnp.array([0.0, 1.0, 2.0]),
            bin_centres=jnp.array([0.5, 1.5, 2.5]),
        )

    with pytest.raises(ValueError, match="strictly positive"):
        jtools._bin_centres_from_edges(jnp.array([0.0, 1.0, 2.0]), log_bins=True)

    with pytest.raises(ValueError, match="same shape as the averaged"):
        jtools._resolve_radial_weights(
            [jnp.arange(4.0), jnp.arange(5.0)],
            jnp.ones((4, 5)),
            jnp.ones((2, 2)),
            ignore_zero_mode=False,
        )

    with pytest.raises(ValueError, match="coords and weights must have the same shape"):
        jtools._get_binweights(
            jnp.ones((4, 5)),
            jnp.ones((4, 4)),
            bins=4,
            bins_upto_boxlen=True,
        )

    with pytest.raises(NotImplementedError, match="complex field"):
        jtools._field_variance(
            indx=jnp.array([1, 1, 2, 2]),
            field=jnp.array([1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j]),
            average=jnp.array([1.5, 3.5]),
            weights=1.0,
            v1=jnp.array([2.0, 2.0]),
        )

    field = jnp.arange(16.0).reshape(4, 4)
    coords = [jnp.linspace(-1.0, 1.0, 4), jnp.linspace(-2.0, 2.0, 4)]
    with pytest.raises(NotImplementedError, match="interpolation-based averaging"):
        jtools.angular_average(field, coords, bins=4, interpolation_method="linear")
    with pytest.raises(ValueError, match=r"same length as field\.ndim"):
        jtools.angular_average(field, [coords[0]], bins=4)
    with pytest.raises(ValueError, match="same shape as the field"):
        jtools.angular_average(field, jnp.ones((3, 3)), bins=4)

    with pytest.raises(ValueError, match=r"between 1 and field\.ndim"):
        jtools.angular_average_nd(field, coords, bins=4, ndims_to_avg=0)
    with pytest.raises(NotImplementedError, match="interpolation-based averaging"):
        jtools.angular_average_nd(field, coords, bins=4, interpolation_method="linear")

    with pytest.raises(ValueError, match="kmag must be provided"):
        jtools.ignore_zero_absk(coords, None)


def test_jax_tools_additional_branch_coverage() -> None:
    # PowerSpectrum validation for secondary optional arrays.
    with pytest.raises(ValueError, match="bin_avg must have length"):
        jtools.PowerSpectrum(
            power=jnp.ones(2),
            bin_edges=jnp.array([0.0, 1.0, 2.0]),
            bin_centres=jnp.array([0.5, 1.5]),
            bin_avg=jnp.ones(3),
        )

    with pytest.raises(ValueError, match="nsamples must have length"):
        jtools.PowerSpectrum(
            power=jnp.ones(2),
            bin_edges=jnp.array([0.0, 1.0, 2.0]),
            bin_centres=jnp.array([0.5, 1.5]),
            nsamples=jnp.ones(3),
        )

    with pytest.raises(ValueError, match="variance must have first dimension"):
        jtools.PowerSpectrum(
            power=jnp.ones(2),
            bin_edges=jnp.array([0.0, 1.0, 2.0]),
            bin_centres=jnp.array([0.5, 1.5]),
            variance=jnp.ones((3,)),
        )

    # _getbins log branch and _resolve_bins_upto_boxlen warning branch.
    coords = jnp.abs(jnp.arange(1, 10, dtype=float)).reshape(3, 3)
    bins = jtools._getbins(4, coords, log=True, bins_upto_boxlen=True)
    assert bins.shape == (5,)
    with pytest.warns(FutureWarning, match="In the future"):
        assert jtools._resolve_bins_upto_boxlen(4, None) is False

    # Complex averaging path and scalar-weights variance path.
    indx = jnp.array([1, 1, 2, 2])
    complex_avg = jtools._field_average(
        indx, jnp.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j]), 1.0, jnp.array([2.0, 2.0])
    )
    assert complex_avg.dtype.kind == "c"
    var = jtools._field_variance(
        indx,
        jnp.array([1.0, 3.0, 5.0, 7.0]),
        jnp.array([2.0, 6.0]),
        1.0,
        jnp.array([2.0, 2.0]),
    )
    assert np.all(np.isfinite(np.asarray(var)))

    # angular_average_nd branch where full-rank weights are sliced by leading index.
    field = jnp.arange(24.0).reshape(3, 4, 2)
    weights = jnp.ones_like(field)
    coords_nd = [jnp.linspace(-1.0, 1.0, 3), jnp.linspace(-1.5, 1.5, 4)]
    with pytest.warns(UserWarning, match="One or more radial bins had no cells within it."):
        out, outbins, outvar, outwght = jtools.angular_average_nd(
            field, coords_nd, bins=3, ndims_to_avg=2, weights=weights, bins_upto_boxlen=True
        )
    assert out.shape[1:] == (2,)
    assert outbins.shape == out.shape
    assert outvar is None
    assert outwght.shape == out.shape
