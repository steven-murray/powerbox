"""Tests of the get_power functionality.

Note that these are not intended to be tests of whether the output power is correct.
That is best tested in the roundtrip tests. These tests are more about checking that
the get_power interface behaves as expected, and that the various options are doing what
they are supposed to do.
"""

import warnings
from functools import partial

import numpy as np
import pytest

from powerbox import (
    PowerBox,
    PowerSpectrum,
    get_power,
    ignore_zero_absk,
    ignore_zero_ki,
    power2delta,
)

get_power = partial(get_power, bins_upto_boxlen=True)


@pytest.fixture(scope="module")
def pb_small_2d():
    return PowerBox(shape=(50, 50), pk=lambda k: 1.0 * k**-2.0, b=1, seed=1234)


@pytest.fixture(scope="module")
def pb_small_3d():
    return PowerBox(shape=(50, 50, 50), pk=lambda k: 1.0 * k**-2.0, b=1, seed=1234)


@pytest.fixture(scope="module")
def grf_2d_small(pb_small_2d: PowerBox):
    """Create a small 2D GRF for testing."""
    return pb_small_2d.delta_x()


@pytest.fixture(scope="module")
def grf_3d_small(pb_small_3d: PowerBox):
    """Create a small 3D GRF for testing."""
    return pb_small_3d.delta_x()


def test_k_zero_ignore(grf_2d_small, pb_small_2d) -> None:
    res1 = get_power(grf_2d_small, pb_small_2d.size)
    res0 = get_power(grf_2d_small, pb_small_2d.size, ignore_zero_mode=True)

    # bin_edges and bin_centres are structurally identical; bin_avg differs
    # because the k=0 mode is excluded from the weighted average in bin 0.
    np.testing.assert_allclose(res1.bin_edges, res0.bin_edges)

    np.testing.assert_allclose(res1.power[1:], res0.power[1:])

    assert res1.power[0] != res0.power[0]


def test_k_weights(grf_3d_small, pb_small_3d) -> None:
    pb = pb_small_3d
    dx = grf_3d_small

    k_weights = np.ones_like(dx)
    k_weights[:, 25] = 0

    res1 = get_power(dx, pb.size)
    res0 = get_power(dx, pb.size, k_weights=k_weights)

    np.testing.assert_allclose(res1.bin_edges, res0.bin_edges)
    assert not np.allclose(res1.power, res0.power)

    k_space_field = dx + dx * 1j
    # large scale modes removed
    k_space_field[15:35, 15:35, 15:35] = 0
    real_space_field = np.fft.ifftn(np.fft.fftshift(k_space_field)).real
    res3 = get_power(real_space_field, pb.size)
    k_space_field = dx + dx * 1j
    real_space_field = np.fft.ifftn(np.fft.fftshift(k_space_field)).real
    # mask out the low-k modes
    k_weights = np.ones_like(dx)
    k_weights[15:35, 15:35, 15:35] = 0
    # set the masked region to zero in the full box
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", message="One or more radial bins had no cells within it")
        res2 = get_power(real_space_field, pb.size, k_weights=k_weights, bins=res3.bin_edges)
    # we expect that the PS of the small box is similar to the PS
    # of the big box with the low-k modes removed

    assert np.all(res3.bin_edges == res2.bin_edges)
    assert np.allclose(res2.power[~np.isnan(res2.power)], res3.power[~np.isnan(res2.power)])


def test_prefactor_fnc(grf_3d_small, pb_small_3d) -> None:
    pb = pb_small_3d
    res_delta = get_power(grf_3d_small, pb.size, prefactor_fnc=power2delta)
    res = get_power(grf_3d_small, pb.size)

    np.testing.assert_allclose(res.bin_avg, res_delta.bin_avg)
    assert np.any(res.power != res_delta.power)


def test_k_weights_fnc(grf_3d_small, pb_small_3d) -> None:
    pb = pb_small_3d
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", message="One or more radial bins had no cells within it")
        res_ki0 = get_power(grf_3d_small, pb.size, k_weights=ignore_zero_ki)
    res = get_power(grf_3d_small, pb.size, k_weights=ignore_zero_absk)

    assert not np.allclose(res.power, res_ki0.power)


def test_res_ndim_zero(grf_3d_small, pb_small_3d) -> None:
    pb = pb_small_3d
    result = get_power(grf_3d_small, pb.size, res_ndim=0)

    assert result.power.ndim == 3
    assert len(result.bin_edges) == 0
    assert len(result.bin_centres) == 0
    assert result.bin_avg is None
    assert result.nsamples is None
    assert result.variance is None
    assert len(result.k_unbinned) == 3


def test_res_ndim_invalid(grf_3d_small, pb_small_3d) -> None:
    pb = pb_small_3d
    with pytest.raises(ValueError, match="res_ndim must be between"):
        get_power(grf_3d_small, pb.size, res_ndim=-1)
    with pytest.raises(ValueError, match="res_ndim must be between"):
        get_power(grf_3d_small, pb.size, res_ndim=4)


def test_power_spectrum_is_powerspectrum(grf_2d_small, pb_small_2d) -> None:
    """get_power returns a PowerSpectrum instance."""
    result = get_power(grf_2d_small, pb_small_2d.size, b=1)
    assert isinstance(result, PowerSpectrum)


def test_power_spectrum_attributes(grf_2d_small, pb_small_2d) -> None:
    """All expected attributes exist and have consistent shapes."""
    result = get_power(grf_2d_small, pb_small_2d.size, b=1)

    n = len(result.power)
    assert result.bin_edges.shape == (n + 1,)
    assert result.bin_centres.shape == (n,)
    assert result.bin_avg.shape == (n,)
    assert result.nsamples.shape == (n,)
    assert result.variance is None
    assert result.k_unbinned is None


def test_power_spectrum_variance(grf_2d_small, pb_small_2d) -> None:
    """Variance is populated when get_variance=True."""
    result = get_power(grf_2d_small, pb_small_2d.size, b=1, get_variance=True)

    assert result.variance is not None
    assert result.variance.shape == result.bin_centres.shape


def test_power_spectrum_bin_edges_monotonic(grf_2d_small, pb_small_2d) -> None:
    """bin_edges should be monotonically increasing."""
    result = get_power(grf_2d_small, pb_small_2d.size, b=1)
    assert np.all(np.diff(result.bin_edges) > 0)


def test_power_spectrum_bin_centres_in_edges(grf_2d_small, pb_small_2d) -> None:
    """bin_centres should fall inside corresponding bin_edges."""
    result = get_power(grf_2d_small, pb_small_2d.size, b=1)
    assert np.all(result.bin_centres >= result.bin_edges[:-1])
    assert np.all(result.bin_centres <= result.bin_edges[1:])


def test_power_spectrum_log_bins(grf_2d_small, pb_small_2d) -> None:
    """log_bins=True produces log-spaced bin_edges and geometric bin_centres."""
    result = get_power(grf_2d_small, pb_small_2d.size, b=1, log_bins=True)

    # bin_centres should be geometric means of adjacent edges
    expected_centres = np.exp((np.log(result.bin_edges[1:]) + np.log(result.bin_edges[:-1])) / 2)
    np.testing.assert_allclose(result.bin_centres, expected_centres)


def test_power_spectrum_partial_average(grf_3d_small, pb_small_3d) -> None:
    """res_ndim < ndim fills k_unbinned and keeps power multi-dimensional."""
    result = get_power(grf_3d_small, pb_small_3d.size, b=1, res_ndim=2)

    assert result.power.ndim == 2
    assert result.bin_avg.ndim == 1
    assert result.nsamples.ndim == 1
    assert result.k_unbinned is not None
    assert len(result.k_unbinned) == 1


def test_powerspectrum_validation_mismatch():
    """PowerSpectrum raises ValueError for inconsistent shapes."""
    power = np.ones(10)
    edges = np.linspace(0, 1, 12)  # 11 edges != 10+1
    centres = np.linspace(0, 1, 10)
    avg = np.linspace(0, 1, 10)
    nsamples = np.ones(10)

    with pytest.raises(ValueError, match="bin_edges must have length"):
        PowerSpectrum(
            power=power,
            bin_edges=edges,
            bin_centres=centres,
            bin_avg=avg,
            nsamples=nsamples,
        )


def test_powerspectrum_validation_bin_avg_shape():
    """PowerSpectrum raises ValueError when bin_avg has wrong length."""
    power = np.ones(10)
    edges = np.linspace(0, 1, 11)
    centres = (edges[1:] + edges[:-1]) / 2

    with pytest.raises(ValueError, match="bin_avg must have length"):
        PowerSpectrum(
            power=power,
            bin_edges=edges,
            bin_centres=centres,
            bin_avg=np.ones(5),  # wrong length
        )


def test_powerspectrum_validation_nsamples_shape():
    """PowerSpectrum raises ValueError when nsamples has wrong length."""
    power = np.ones(10)
    edges = np.linspace(0, 1, 11)
    centres = (edges[1:] + edges[:-1]) / 2

    with pytest.raises(ValueError, match="nsamples must have length"):
        PowerSpectrum(
            power=power,
            bin_edges=edges,
            bin_centres=centres,
            nsamples=np.ones(5),  # wrong length
        )


def test_powerspectrum_validation_variance_shape():
    """PowerSpectrum raises ValueError when variance has wrong shape."""
    power = np.ones(10)
    edges = np.linspace(0, 1, 11)
    centres = (edges[1:] + edges[:-1]) / 2
    avg = centres.copy()
    nsamples = np.ones(10)

    with pytest.raises(ValueError, match="variance must have first dimension"):
        PowerSpectrum(
            power=power,
            bin_edges=edges,
            bin_centres=centres,
            bin_avg=avg,
            nsamples=nsamples,
            variance=np.ones(5),
        )
