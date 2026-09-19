"""Tests of the hermitian helpers."""

import itertools

import numpy as np
import pytest

from powerbox import PowerBox
from powerbox.dft import irfft


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("ncells", [16, 17])
@pytest.mark.parametrize("ab", [(0, 1), (0, 2 * np.pi)])
def test_deltax_is_real(ndim, ncells, ab):
    pb = PowerBox(
        pk=lambda k: 1,
        size=(1.0,) * ndim,
        shape=(ncells,) * ndim,
        seed=1234,
        a=ab[0],
        b=ab[1],
    )

    dk = pb.delta_k()

    deltax = irfft(
        dk,
        L=1,
        a=ab[0],
        b=ab[1],
        N=pb.shape,
    )[0]

    assert np.isrealobj(deltax)


@pytest.mark.parametrize("shape", [(16, 17), (17, 16), (16, 17, 18), (17, 16, 19)])
@pytest.mark.parametrize("ab", [(0, 1), (0, 2 * np.pi)])
def test_non_cubic_deltax_is_real(shape, ab):
    """Mixed odd/even non-cubic delta_x realizations remain real."""
    boxlength = tuple(float(index + 2) for index in range(len(shape)))
    pb = PowerBox(
        pk=lambda k: 1,
        size=boxlength,
        shape=shape,
        seed=1234,
        a=ab[0],
        b=ab[1],
    )

    dk = pb.delta_k()

    deltax = irfft(
        dk,
        L=boxlength,
        a=ab[0],
        b=ab[1],
        N=pb.shape,
    )[0]

    assert np.isrealobj(deltax)


def _self_conjugate_indices(shape):
    index_sets = []
    for axis, axis_n in enumerate(shape):
        values = [0] if axis == len(shape) - 1 else [axis_n // 2]
        if axis_n % 2 == 0:
            values.append(axis_n // 2 if axis == len(shape) - 1 else 0)
        index_sets.append(tuple(dict.fromkeys(values)))
    return itertools.product(*index_sets)


def _assert_full_hermitian(arr):
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
def test_reduced_gaussian_modes_preserve_real_self_conjugate_modes(shape):
    """The reduced ``irfftn`` spectrum keeps only valid self-conjugate real modes."""
    pb = PowerBox(
        shape=shape,
        pk=lambda k: (1 + k) ** (-2.0),
        size=tuple(float(axis + 2) for axis in range(len(shape))),
        seed=42,
        ensure_physical=False,
    )
    gh = pb.gauss_hermitian()

    assert gh.shape == (*shape[:-1], shape[-1] // 2 + 1)

    for idx in _self_conjugate_indices(shape):
        assert np.allclose(gh[idx].imag, 0, atol=1e-10), (
            f"Reduced self-conjugate mode {idx} is complex"
        )


@pytest.mark.parametrize("shape", [(4,), (5,), (4, 5), (5, 4), (4, 5, 6), (5, 4, 7)])
def test_reduced_gaussian_modes_boundary_surfaces_are_hermitian(shape):
    """Self-conjugate reduced-spectrum surfaces remain Hermitian in lower dimensions."""
    pb = PowerBox(
        shape=shape,
        pk=lambda k: (1 + k) ** (-2.0),
        size=tuple(float(axis + 2) for axis in range(len(shape))),
        seed=52,
        ensure_physical=False,
    )
    gh = pb.gauss_hermitian()
    surface_indices = [0]
    if shape[-1] % 2 == 0:
        surface_indices.append(shape[-1] // 2)

    for surface_index in surface_indices:
        surface = gh[..., surface_index]
        if np.ndim(surface) == 0:
            assert np.allclose(np.imag(surface), 0, atol=1e-10)
        else:
            _assert_full_hermitian(surface)


@pytest.mark.parametrize("shape", [(4,), (5,), (4, 4), (5, 5), (4, 4, 4), (5, 5, 5)])
def test_gauss_hermitian_returns_reduced_hermitian_modes(shape):
    """The public Gaussian mode sampler returns reduced Hermitian rFFT modes."""
    pb = PowerBox(
        shape=shape,
        pk=lambda k: (1 + k) ** (-2.0),
        size=tuple(float(axis + 2) for axis in range(len(shape))),
        seed=42,
        ensure_physical=False,
    )
    gh = pb.gauss_hermitian()

    assert gh.shape == (*shape[:-1], shape[-1] // 2 + 1)

    for idx in _self_conjugate_indices(shape):
        assert np.allclose(gh[idx].imag, 0, atol=1e-10)

    surface_indices = [0]
    if shape[-1] % 2 == 0:
        surface_indices.append(shape[-1] // 2)

    for surface_index in surface_indices:
        surface = gh[..., surface_index]
        if np.ndim(surface) == 0:
            assert np.allclose(np.imag(surface), 0, atol=1e-10)
        else:
            _assert_full_hermitian(surface)


@pytest.mark.parametrize("shape", [(8, 10), (8, 11), (9, 10), (6, 8, 10), (7, 8, 11)])
def test_hermitian_modes_have_unit_variance_everywhere(shape):
    """Enforcing Hermitian symmetry must not change the power of any mode.

    The self-conjugate surfaces of the reduced spectrum (final-axis index 0, and Nyquist
    for an even final axis) are the only modes the Hermitian projection touches. Projecting
    by *averaging* a mode with its conjugate partner halves their variance, which silently
    removes half the power from those whole surfaces -- including many of the lowest-|k|
    modes -- while leaving the structural Hermitian property above perfectly intact. Hence
    this test: the structure alone does not pin down the projection.
    """
    nrealizations = 4000
    accumulated = np.zeros((*shape[:-1], shape[-1] // 2 + 1))
    for seed in range(nrealizations):
        pb = PowerBox(
            shape=shape,
            pk=lambda k: 1.0,
            size=tuple(1.0 for _ in shape),
            seed=seed,
        )
        accumulated += np.abs(pb.gauss_hermitian()) ** 2

    variance = accumulated / nrealizations

    # Standard error on each mean is ~1/sqrt(nrealizations); allow 5 sigma plus slack for
    # the correlations between conjugate partners within a surface.
    tolerance = 8 / np.sqrt(nrealizations)

    surfaces = {"bulk": variance[..., 1:-1], "k_last=0": variance[..., 0]}
    if shape[-1] % 2 == 0:
        surfaces["nyquist"] = variance[..., -1]

    for name, surface in surfaces.items():
        if surface.size:
            assert np.abs(surface.mean() - 1) < tolerance, (
                f"{name} modes have mean |g|^2 = {surface.mean():.4f}, expected 1"
            )
