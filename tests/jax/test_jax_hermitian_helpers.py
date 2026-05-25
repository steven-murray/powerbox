"""Test of JAX Hermitian helper functions."""

from __future__ import annotations

import importlib
import itertools

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


def _assert_full_hermitian(arr: np.ndarray) -> None:
    """Assert that a centred Fourier array is Hermitian."""
    unshifted = np.fft.ifftshift(arr)
    shape = unshifted.shape

    for index in itertools.product(*[range(axis_n) for axis_n in shape]):
        partner = tuple((-index[axis]) % shape[axis] for axis in range(len(shape)))
        if partner == index:
            assert np.allclose(unshifted[index].imag, 0, atol=1e-10)
        else:
            assert np.allclose(unshifted[index], np.conj(unshifted[partner]), atol=1e-10)


@pytest.mark.parametrize("shape", [(4,), (5,), (4, 5), (5, 4), (4, 5, 6), (5, 4, 7)])
def test_jax_reduced_gaussian_modes_preserve_real_self_conjugate_modes(
    shape: tuple[int, ...],
) -> None:
    pb = jpb.PowerBox(
        shape=shape,
        dim=len(shape),
        pk=lambda k: (1 + k) ** -2.0,
        size=tuple(float(axis + 2) for axis in range(len(shape))),
        key=jax.random.key(7),
    )
    gh = np.asarray(pb._gaussian_modes_rfft())

    assert gh.shape == (*shape[:-1], shape[-1] // 2 + 1)

    index_sets = []
    for axis, axis_n in enumerate(shape):
        values = [0] if axis == len(shape) - 1 else [axis_n // 2]
        if axis_n % 2 == 0:
            values.append(axis_n // 2 if axis == len(shape) - 1 else 0)
        index_sets.append(tuple(dict.fromkeys(values)))

    for idx in np.ndindex(*(len(values) for values in index_sets)):
        mode_index = tuple(index_sets[axis][i] for axis, i in enumerate(idx))
        assert np.allclose(gh[mode_index].imag, 0, atol=1e-10)


@pytest.mark.parametrize("shape", [(4,), (5,), (4, 5), (5, 4), (4, 5, 6), (5, 4, 7)])
def test_jax_reduced_gaussian_modes_boundary_surfaces_are_hermitian(shape: tuple[int, ...]) -> None:
    pb = jpb.PowerBox(
        shape=shape,
        dim=len(shape),
        pk=lambda k: (1 + k) ** -2.0,
        size=tuple(float(axis + 2) for axis in range(len(shape))),
        key=jax.random.key(8),
    )
    gh = np.asarray(pb._gaussian_modes_rfft())
    surface_indices = [0]
    if shape[-1] % 2 == 0:
        surface_indices.append(shape[-1] // 2)

    for surface_index in surface_indices:
        surface = gh[..., surface_index]
        if np.ndim(surface) == 0:
            assert np.allclose(np.imag(surface), 0, atol=1e-10)
        else:
            _assert_full_hermitian(surface)
