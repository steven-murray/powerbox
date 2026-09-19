"""Test of JAX get_power interface."""

from __future__ import annotations

import importlib
from contextlib import nullcontext

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


def test_jax_get_power_partial_average_retains_unbinned_axes() -> None:
    pb = jpb.PowerBox(
        shape=(18, 20, 22),
        pk=lambda k: (1 + k) ** -2.0,
        size=(4.0, 5.0, 6.0),
        key=jax.random.key(2),
        b=1,
    )

    result = jpb.get_power(
        pb.delta_x(),
        pb.size,
        b=1,
        res_ndim=2,
        prefactor_fnc=jpb.power2delta,
        get_variance=True,
        bins_upto_boxlen=True,
    )

    assert result.power.shape[1] == 22
    assert result.variance is not None
    assert result.k_unbinned is not None
    assert len(result.k_unbinned) == 1
    assert result.k_unbinned[0].shape == (22,)


@pytest.mark.parametrize("k_weights", [jpb.ignore_zero_absk, jpb.ignore_zero_ki])
def test_jax_get_power_accepts_boolean_k_weights_helpers(k_weights) -> None:
    pb = jpb.PowerBox(
        shape=(18, 20, 22),
        dim=3,
        pk=lambda k: (1 + k) ** -2.0,
        size=(4.0, 5.0, 6.0),
        key=jax.random.key(4),
        b=1,
    )

    expectation = (
        pytest.warns(UserWarning, match="One or more radial bins had no cells within it.")
        if k_weights is jpb.ignore_zero_ki
        else nullcontext()
    )

    with expectation:
        result = jpb.get_power(
            pb.delta_x(),
            pb.size,
            b=1,
            k_weights=k_weights,
            bins_upto_boxlen=True,
        )

    assert result.power.ndim == 1
    assert result.nsamples is not None
    assert np.all(np.isfinite(np.asarray(result.power[1:])))


def test_jax_get_power_partial_average_accepts_ignore_zero_ki() -> None:
    pb = jpb.PowerBox(
        shape=(18, 20, 22),
        pk=lambda k: (1 + k) ** -2.0,
        size=(4.0, 5.0, 6.0),
        key=jax.random.key(5),
        b=1,
    )

    with pytest.warns(UserWarning, match="One or more radial bins had no cells within it."):
        result = jpb.get_power(
            pb.delta_x(),
            pb.size,
            b=1,
            res_ndim=2,
            k_weights=jpb.ignore_zero_ki,
            bins_upto_boxlen=True,
        )

    assert result.power.shape[1] == 22
    assert result.nsamples is not None
    assert result.nsamples.ndim == 1


def test_jax_get_power_validation_branches() -> None:
    field = jnp.arange(16.0).reshape(4, 4)
    with pytest.raises(ValueError, match="same shape"):
        jpb.get_power(field, 2.0, deltax2=jnp.arange(9.0).reshape(3, 3))
    with pytest.raises(ValueError, match="res_ndim must be between"):
        jpb.get_power(field, 2.0, res_ndim=-1)
    with pytest.raises(ValueError, match="res_ndim must be between"):
        jpb.get_power(field, 2.0, res_ndim=3)

    result = jpb.get_power(field, 2.0, res_ndim=0)
    assert result.bin_edges.size == 0
    assert result.bin_centres.size == 0
