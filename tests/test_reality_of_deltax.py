"""Tests that deltax is real under different assumptions."""

import numpy as np
import pytest

from powerbox import PowerBox
from powerbox.dft import ifft


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("ncells", [16, 17])
@pytest.mark.parametrize("ab", [(0, 1), (0, 2 * np.pi)])
def test_deltax_is_real(ndim, ncells, ab):
    pb = PowerBox(pk=lambda k: 1, boxlength=1, N=ncells, dim=ndim, seed=1234)

    dk = pb.delta_k()

    deltax = ifft(
        dk,
        L=1,
        a=ab[0],
        b=ab[1],
    )[0]

    np.testing.assert_allclose(deltax.imag, 0, atol=1e-12)


@pytest.mark.parametrize("ncells", [4, 5])
def test_hermitianity_2d(ncells):
    """Test that the gauss_hermitian method produces a Hermitian array in 2D.

    This test was inspired by https://github.com/steven-murray/powerbox/issues/84
    """
    pb = PowerBox(
        N=ncells,
        dim=2,
        pk=lambda k: (1 + k) ** (-2.0),
        boxlength=32.0,
        seed=42,
        ensure_physical=False,
    )
    gh = pb.gauss_hermitian()

    freq_idx = [(i - ncells // 2, j - ncells // 2) for i in range(ncells) for j in range(ncells)]

    for i, j in freq_idx:
        this = gh.flatten()[freq_idx.index((i, j))]

        if (-i, -j) in freq_idx:
            that = gh.flatten()[freq_idx.index((-i, -j))]
            assert np.allclose(this, np.conj(that), atol=1e-10), (
                f"Hermitian error at ({i},{j}): {this} vs {that}"
            )
        elif (i, -j) in freq_idx:
            that = gh.flatten()[freq_idx.index((i, -j))]
            assert np.allclose(this, np.conj(that), atol=1e-10), (
                f"Hermitian error at ({i},{j}): {this} vs {that}"
            )
        elif (-i, j) in freq_idx:
            that = gh.flatten()[freq_idx.index((-i, j))]
            assert np.allclose(this, np.conj(that), atol=1e-10), (
                f"Hermitian error at ({i},{j}): {this} vs {that}"
            )
        else:
            assert np.allclose(this.imag, 0, atol=1e-10), (
                f"Non-Hermitian error at ({i},{j}): {this}"
            )


@pytest.mark.parametrize("ncells", [4, 5])
def test_hermitianity_1d(ncells):
    """Test that the gauss_hermitian method produces a Hermitian array in 1D.

    This test was inspired by https://github.com/steven-murray/powerbox/issues/84
    """
    pb = PowerBox(
        N=ncells,
        dim=1,
        pk=lambda k: (1 + k) ** (-2.0),
        boxlength=32.0,
        seed=42,
        ensure_physical=False,
    )
    gh = pb.gauss_hermitian()

    freq_idx = [(i - ncells // 2) for i in range(ncells)]

    for i in freq_idx:
        this = gh.flatten()[freq_idx.index(i)]

        if (-i) in freq_idx:
            that = gh.flatten()[freq_idx.index(-i)]
            assert np.allclose(this, np.conj(that), atol=1e-10), (
                f"Hermitian error at ({i}): {this} vs {that}"
            )
        else:
            assert np.allclose(this.imag, 0, atol=1e-10), f"Non-Hermitian error at ({i}): {this}"


@pytest.mark.parametrize("ncells", [4, 5])
def test_hermitianity_3d(ncells):
    """Test that the gauss_hermitian method produces a Hermitian array in 3D.

    https://github.com/steven-murray/powerbox/issues/84
    """
    pb = PowerBox(
        N=ncells,
        dim=3,
        pk=lambda k: (1 + k) ** (-2.0),
        boxlength=32.0,
        seed=42,
        ensure_physical=False,
    )
    gh = pb.gauss_hermitian()

    freq_idx = [
        (i - ncells // 2, j - ncells // 2, k - ncells // 2)
        for i in range(ncells)
        for j in range(ncells)
        for k in range(ncells)
    ]

    for i, j, k in freq_idx:
        this = gh.flatten()[freq_idx.index((i, j, k))]

        if (-i, -j, -k) in freq_idx:
            that = gh.flatten()[freq_idx.index((-i, -j, -k))]
            assert np.allclose(this, np.conj(that), atol=1e-10), (
                f"Hermitian error at ({i},{j},{k}): {this} vs {that}"
            )
        elif (i, -j, -k) in freq_idx:
            that = gh.flatten()[freq_idx.index((i, -j, -k))]
            assert np.allclose(this, np.conj(that), atol=1e-10), (
                f"Hermitian error at ({i},{j},{k}): {this} vs {that}"
            )
        elif (-i, j, -k) in freq_idx:
            that = gh.flatten()[freq_idx.index((-i, j, -k))]
            assert np.allclose(this, np.conj(that), atol=1e-10), (
                f"Hermitian error at ({i},{j},{k}): {this} vs {that}"
            )
        elif (-i, -j, k) in freq_idx:
            that = gh.flatten()[freq_idx.index((-i, -j, k))]
            assert np.allclose(this, np.conj(that), atol=1e-10), (
                f"Hermitian error at ({i},{j},{k}): {this} vs {that}"
            )
        elif (i, j, -k) in freq_idx:
            that = gh.flatten()[freq_idx.index((i, j, -k))]
            assert np.allclose(this, np.conj(that), atol=1e-10), (
                f"Hermitian error at ({i},{j},{k}): {this} vs {that}"
            )
        elif (i, -j, k) in freq_idx:
            that = gh.flatten()[freq_idx.index((i, -j, k))]
            assert np.allclose(this, np.conj(that), atol=1e-10), (
                f"Hermitian error at ({i},{j},{k}): {this} vs {that}"
            )
        elif (-i, j, k) in freq_idx:
            that = gh.flatten()[freq_idx.index((-i, j, k))]
            assert np.allclose(this, np.conj(that), atol=1e-10), (
                f"Hermitian error at ({i},{j},{k}): {this} vs {that}"
            )
        else:
            assert np.allclose(this.imag, 0, atol=1e-10), (
                f"Non-Hermitian error at ({i},{j},{k}): {this}"
            )
