"""Tests that deltax is real under different assumptions."""

import itertools

import numpy as np
import pytest

from powerbox import PowerBox
from powerbox.dft import irfft


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("ncells", [16, 17])
@pytest.mark.parametrize("ab", [(0, 1), (0, 2 * np.pi)])
def test_deltax_is_real(ndim, ncells, ab):
    pb = PowerBox(pk=lambda k: 1, boxlength=1, N=ncells, dim=ndim, seed=1234, a=ab[0], b=ab[1])

    dk = pb.delta_k()

    deltax = irfft(
        dk,
        L=1,
        a=ab[0],
        b=ab[1],
        N=pb.N,
    )[0]

    assert np.isrealobj(deltax)


@pytest.mark.parametrize("shape", [(16, 17), (17, 16), (16, 17, 18), (17, 16, 19)])
@pytest.mark.parametrize("ab", [(0, 1), (0, 2 * np.pi)])
def test_non_cubic_deltax_is_real(shape, ab):
    """Mixed odd/even non-cubic delta_x realizations remain real."""
    boxlength = tuple(float(index + 2) for index in range(len(shape)))
    pb = PowerBox(
        pk=lambda k: 1,
        boxlength=boxlength,
        N=shape,
        dim=len(shape),
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
        N=pb.N,
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
        N=shape,
        dim=len(shape),
        pk=lambda k: (1 + k) ** (-2.0),
        boxlength=tuple(float(axis + 2) for axis in range(len(shape))),
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
        N=shape,
        dim=len(shape),
        pk=lambda k: (1 + k) ** (-2.0),
        boxlength=tuple(float(axis + 2) for axis in range(len(shape))),
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
        N=shape,
        dim=len(shape),
        pk=lambda k: (1 + k) ** (-2.0),
        boxlength=tuple(float(axis + 2) for axis in range(len(shape))),
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
