"""Tests for isotropic power-spectrum estimation."""

import warnings
from functools import partial

import numpy as np
import pytest

from powerbox import PowerBox, get_power, ignore_zero_absk, ignore_zero_ki, power2delta

get_power = partial(get_power, bins_upto_boxlen=True)


def test_power1d() -> None:
    p = [0] * 40
    rng = np.random.default_rng(42)
    for i in range(40):
        pb = PowerBox(
            8001,
            dim=1,
            pk=lambda k: 1.0 * k**-2.0,
            boxlength=1.0,
            a=0,
            b=1,
            seed=rng.integers(0, 10000),
        )

        p[i], k, *_ = get_power(pb.delta_x(), pb.boxlength, a=0, b=1)

    np.testing.assert_allclose(np.mean(np.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -2.0, rtol=2)


def test_power1d_n3() -> None:
    p = [0] * 40
    rng = np.random.default_rng(123)
    for i in range(40):
        pb = PowerBox(
            8001,
            dim=1,
            pk=lambda k: 1.0 * k**-3.0,
            boxlength=1.0,
            b=1,
            seed=rng.integers(0, 10000),
        )
        p[i], k, *_ = get_power(pb.delta_x(), pb.boxlength, b=1)

    np.testing.assert_allclose(np.mean(np.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -3.0, rtol=2)


def test_power1d_bigL() -> None:
    p = [0] * 40
    rng = np.random.default_rng(456)
    for i in range(40):
        pb = PowerBox(
            8001,
            dim=1,
            pk=lambda k: 1.0 * k**-3.0,
            boxlength=10.0,
            b=1,
            seed=rng.integers(0, 10000),
        )
        p[i], k, *_ = get_power(pb.delta_x(), pb.boxlength, b=1)

    np.testing.assert_allclose(np.mean(np.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -3.0, rtol=2)


def test_power1d_ordinary_freq() -> None:
    p = [0] * 40
    rng = np.random.default_rng(789)
    for i in range(40):
        pb = PowerBox(
            8001,
            dim=1,
            pk=lambda k: 1.0 * k**-3.0,
            boxlength=1.0,
            b=1,
            seed=rng.integers(0, 10000),
        )
        p[i], k, *_ = get_power(pb.delta_x(), pb.boxlength)

    np.testing.assert_allclose(np.mean(np.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -3.0, rtol=2)


def test_power1d_halfN() -> None:
    p = [0] * 40
    rng = np.random.default_rng(101112)
    for i in range(40):
        pb = PowerBox(
            4001,
            dim=1,
            pk=lambda k: 1.0 * k**-3.0,
            boxlength=1.0,
            b=1,
            seed=rng.integers(0, 100000),
        )
        p[i], k, *_ = get_power(pb.delta_x(), pb.boxlength, b=1)

    np.testing.assert_allclose(np.mean(np.array(p), axis=0)[1000:], 1.0 * k[1000:] ** -3.0, rtol=2)


def test_power2d() -> None:
    p = [0] * 5
    rng = np.random.default_rng(13579)
    for i in range(5):
        pb = PowerBox(
            200,
            dim=2,
            pk=lambda k: 1.0 * k**-2.0,
            boxlength=1.0,
            b=1,
            seed=rng.integers(0, 10000),
        )
        p[i], k, *_ = get_power(pb.delta_x(), pb.boxlength, b=1)

    np.testing.assert_allclose(np.mean(np.array(p), axis=0)[100:], 1.0 * k[100:] ** -2.0, rtol=2)


def test_power3d():
    rng = np.random.default_rng(13579)
    pb = PowerBox(
        50,
        dim=3,
        pk=lambda k: 1.0 * k**-2.0,
        boxlength=1.0,
        b=1,
        seed=rng.integers(0, 10000),
    )
    p, k, *_ = get_power(pb.delta_x(), pb.boxlength, b=1)

    print(p / (1.0 * k**-2.0))
    np.testing.assert_allclose(p, 1.0 * k**-2.0, rtol=2)


def test_k_zero_ignore() -> None:
    pb = PowerBox(50, dim=2, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)

    dx = pb.delta_x()
    p1, k1, *_ = get_power(dx, pb.boxlength, bin_ave=False)
    p0, k0, *_ = get_power(dx, pb.boxlength, ignore_zero_mode=True, bin_ave=False)

    np.testing.assert_allclose(k1, k0)

    np.testing.assert_allclose(p1[1:], p0[1:])

    assert p1[0] != p0[0]


def test_k_weights() -> None:
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)

    dx = pb.delta_x()

    k_weights = np.ones_like(dx)
    k_weights[:, 25] = 0

    p1, k1, *_ = get_power(dx, pb.boxlength, bin_ave=False)
    p0, k0, *_ = get_power(dx, pb.boxlength, bin_ave=False, k_weights=k_weights)

    np.testing.assert_allclose(k1, k0)
    assert not np.allclose(p1, p0)

    k_space_field = dx + dx * 1j
    # large scale modes removed
    k_space_field[15:35, 15:35, 15:35] = 0
    real_space_field = np.fft.ifftn(np.fft.fftshift(k_space_field)).real
    p3, k3, *_ = get_power(real_space_field, pb.boxlength, bin_ave=False)
    k_space_field = dx + dx * 1j
    real_space_field = np.fft.ifftn(np.fft.fftshift(k_space_field)).real
    # mask out the low-k modes
    k_weights = np.ones_like(dx)
    k_weights[15:35, 15:35, 15:35] = 0
    # set the masked region to zero in the full box
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", message="One or more radial bins had no cells within it")
        p2, k2, *_ = get_power(
            real_space_field, pb.boxlength, bin_ave=False, k_weights=k_weights, bins=k3
        )
    # we expect that the PS of the small box is similar to the PS
    # of the big box with the low-k modes removed

    assert np.all(k3 == k2)
    assert np.allclose(p2[~np.isnan(p2)], p3[~np.isnan(p2)])


def test_prefactor_fnc() -> None:
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    pdelta, kdelta, *_ = get_power(pb.delta_x(), pb.boxlength, prefactor_fnc=power2delta)
    p, k, *_ = get_power(pb.delta_x(), pb.boxlength)

    np.testing.assert_allclose(k, kdelta)
    assert np.any(p != pdelta)


def test_k_weights_fnc() -> None:
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", message="One or more radial bins had no cells within it")
        p_ki0, *_ = get_power(pb.delta_x(), pb.boxlength, k_weights=ignore_zero_ki)
    p, *_ = get_power(pb.delta_x(), pb.boxlength, k_weights=ignore_zero_absk)

    assert not np.allclose(p, p_ki0)


def test_res_ndim_zero() -> None:
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    p, k, var, sumweights, remaining_freq = get_power(pb.delta_x(), pb.boxlength, res_ndim=0)

    assert p.ndim == 3
    assert k is None
    assert var is None
    assert sumweights is None
    assert len(remaining_freq) == 3


def test_res_ndim_invalid() -> None:
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    with pytest.raises(ValueError, match="res_ndim must be between"):
        get_power(pb.delta_x(), pb.boxlength, res_ndim=-1)
    with pytest.raises(ValueError, match="res_ndim must be between"):
        get_power(pb.delta_x(), pb.boxlength, res_ndim=4)
