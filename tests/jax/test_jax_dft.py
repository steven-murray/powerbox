"""Test of JAX DFT interface."""

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


def test_jax_fft_roundtrip_matches_input() -> None:
    field = jnp.arange(16.0).reshape(4, 4)
    transformed, _ = jpb.fft(field, L=2.0, a=0, b=2 * np.pi)
    recovered, _ = jpb.ifft(transformed, L=2.0, a=0, b=2 * np.pi)

    np.testing.assert_allclose(np.asarray(recovered), np.asarray(field))


def test_jax_dft_frequency_wrappers_are_callable() -> None:
    field = jnp.arange(8.0).reshape(2, 4)
    shifted = jpb.fftshift(field)
    unshifted = jpb.ifftshift(shifted)
    np.testing.assert_allclose(np.asarray(unshifted), np.asarray(field))

    k = jdft.fftfreq(8, d=0.5, b=1)
    kr = jdft.rfftfreq(8, d=0.5, b=1)
    assert np.asarray(k).shape == (8,)
    assert np.asarray(kr).shape == (5,)


def test_jax_irfft_supports_inferred_and_scalar_n() -> None:
    full = np.random.default_rng(0).normal(size=(6, 8))
    reduced = np.fft.rfftn(full)
    reduced = np.fft.fftshift(reduced, axes=(0,))

    rec_default, _ = jdft.irfft(reduced, axes=(0, 1), a=0, b=2 * np.pi)
    rec_x0, _ = jdft.irfft(reduced, axes=(0, 1), x0=(0.2, -0.1), a=0, b=2 * np.pi)

    assert np.asarray(rec_default).shape == full.shape
    assert np.asarray(rec_x0).shape == full.shape

    with pytest.raises(ValueError, match="inconsistent with the reduced spectrum shape"):
        jdft.irfft(reduced, axes=(0, 1), N=8, a=0, b=2 * np.pi)

    with pytest.raises(ValueError, match="same length"):
        jdft.irfft(reduced, axes=(0, 1), N=(6, 8, 10), a=0, b=2 * np.pi)


def test_dft_default_length_branches_are_exercised() -> None:
    x = np.random.default_rng(1).normal(size=(6, 8))

    # fft default L path (L and Lk both omitted)
    f_default, _ = ndft.fft(x, axes=(0, 1), a=0, b=2 * np.pi, nthreads=1)
    assert f_default.shape == x.shape

    # fft scalar Lk path
    f_lk, _ = ndft.fft(x, Lk=3.0, axes=(0, 1), a=0, b=2 * np.pi, nthreads=1)
    assert f_lk.shape == x.shape

    # ifft default Lk path
    x_ifft, _ = ndft.ifft(f_default, axes=(0, 1), a=0, b=2 * np.pi, nthreads=1)
    assert x_ifft.shape == x.shape

    # irfft scalar Lk path
    reduced = np.fft.rfftn(x)
    reduced = np.fft.fftshift(reduced, axes=(0,))
    x_irfft, _ = ndft.irfft(reduced, Lk=3.0, axes=(0, 1), N=x.shape, a=0, b=2 * np.pi, nthreads=1)
    assert x_irfft.shape == x.shape
