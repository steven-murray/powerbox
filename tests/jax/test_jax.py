"""Tests for the JAX-backed ``powerbox`` namespace."""

from __future__ import annotations

import importlib

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")
jpb = importlib.import_module("powerbox.jax")
jpb_powerbox = importlib.import_module("powerbox.jax.powerbox")
jdft = importlib.import_module("powerbox.jax.dft")
jtools = importlib.import_module("powerbox.jax.tools")
ndft = importlib.import_module("powerbox.dft")


def test_jax_unsupported_paths_raise() -> None:
    pb = jpb.PowerBox(
        shape=(16, 16),
        pk=lambda k: (1 + k) ** -2.0,
        size=(4.0, 4.0),
        key=jax.random.key(3),
    )
    field = pb.delta_x()

    with pytest.raises(NotImplementedError, match="interpolation"):
        jpb.get_power(field, pb.size, interpolation_method="linear")

    with pytest.raises(NotImplementedError, match="discrete samples"):
        jpb.get_power(field, pb.size, N=pb.shape)

    with pytest.raises(NotImplementedError, match="create_discrete_sample"):
        pb.create_discrete_sample(1.0)
