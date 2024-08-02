import pytest

import numpy as np
import warnings

from powerbox.powerbox import PowerBox
from powerbox.tools import (
    above_mu_min_angular_generator,
    angular_average,
    angular_average_nd,
    get_power,
    regular_angular_generator,
)


def test_warn_interp_weights():
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = r2**-1.0
    P = np.repeat(P, 100).reshape(400, 400, 100)
    freq = [x, x, np.linspace(-2, 2, 100)]
    weights = np.random.rand(np.prod(P.shape)).reshape(P.shape)
    with pytest.warns(RuntimeWarning):
        angular_average(
            P,
            freq,
            bins=10,
            interpolation_method="linear",
            weights=weights,
            interp_points_generator=regular_angular_generator(),
        )


@pytest.mark.parametrize("interpolation_method", [None, "linear"])
def test_angular_avg_nd_3(interpolation_method):
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = r2**-1.0
    P = np.repeat(P, 100).reshape(400, 400, 100)
    freq = [x, x, np.linspace(-2, 2, 100)]
    p_k, k_av_bins, sw = angular_average_nd(
        P,
        freq,
        bins=50,
        n=2,
        interpolation_method=interpolation_method,
        return_sumweights=True,
    )
    if interpolation_method == "linear":
        assert np.max(np.abs((p_k[:, 0] - k_av_bins**-2.0) / k_av_bins**-2.0)) < 0.05
    else:
        # Without interpolation, the radially-averaged power is not very accurate
        # due to the low number of bins at small values of k_av_bins, so we start
        # the comparison at the 6th bin.
        assert (
            np.max(np.abs((p_k[6:, 0] - k_av_bins[6:] ** -2.0) / k_av_bins[6:] ** -2.0))
            < 0.05
        )


def test_weights_shape():
    x = np.linspace(-3, 3, 40)
    P = np.ones(3 * [40])
    weights = np.ones(3 * [20])
    freq = [x for _ in range(3)]

    with pytest.raises(ValueError):
        p_k_lin, k_av_bins_lin = angular_average(
            P,
            freq,
            bins=10,
            weights=weights,
        )


@pytest.mark.parametrize("n", range(1, 5))
def test_interp_w_weights(n):
    x = np.linspace(-3, 3, 40)
    P = np.ones(n * [40])
    weights = np.ones_like(P)
    if n == 1:
        P[2:5] = 0
        weights[2:5] = 0
    elif n == 2:
        P[2:5, 2:5] = 0
        weights[2:5, 2:5] = 0
    elif n == 3:
        P[:4, 3:6, 7:10] = 0
        weights[:4, :, :] = 0
        weights[:, 3:6, :] = 0
        weights[:, :, 7:10] = 0
    else:
        P[:4, 3:6, 7:10, 1:2] = 0
        weights[:4, :, :, :] = 0
        weights[:, 3:6, :, :] = 0
        weights[:, :, 7:10, :] = 0
        weights[:, :, :, 1:2] = 0

    # Test 4D avg works
    freq = [x for _ in range(n)]
    p_k_lin, k_av_bins_lin = angular_average(
        P,
        freq,
        bins=10,
        interpolation_method="linear",
        weights=weights,
        interp_points_generator=regular_angular_generator(),
        log_bins=True,
    )

    assert np.all(p_k_lin == 1.0)


@pytest.mark.parametrize("n", range(1, 3))
def test_zero_ki(n):
    x = np.arange(-100, 100, 1)
    from powerbox.tools import ignore_zero_ki

    # needed only for shape
    freq = n * [x]
    coords = np.array(np.meshgrid(*freq))
    kmag = np.sqrt(np.sum(coords**2, axis=0))
    weights = ignore_zero_ki(freq, kmag)
    L = x[-1] - x[0] + 1
    masked_points = np.sum(weights == 0)
    if n == 1:
        assert masked_points == 1
    elif n == 2:
        assert masked_points == n * L - 1
    elif n == 3:
        assert masked_points == n * L**2 - n * L + 1
    else:
        assert masked_points == n * L**3 - n * L**2 + n * L - 1


@pytest.mark.parametrize("n", range(2, 3))
def test_interp_w_mu(n):
    x = np.linspace(0.0, 3, 40)
    if n == 2:
        kpar_mesh, kperp_mesh = np.meshgrid(x, x)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in divide"
            )
            theta = np.arctan2(kperp_mesh, kpar_mesh)
        mu_mesh = np.cos(theta)
    else:
        kx_mesh, ky_mesh, kz_mesh = np.meshgrid(x, x, x, indexing="ij")
        theta = np.arccos(kz_mesh / np.sqrt(kx_mesh**2 + ky_mesh**2 + kz_mesh**2))
        mu_mesh = np.cos(theta)

    # Need a little cushion so we test against data at mu = 0.95
    # If we test for mu that is higher (default is mu >= 0.97)
    # and put the data also only at mu >= 0.97, then the interped average will
    # not be 1. at low radii so the test fails.
    mask = mu_mesh >= 0.95
    P = np.zeros(mask.shape)
    P[mask] = 1.0

    p_k_lin, k_av_bins_lin = angular_average(
        P,
        n * [x],
        bins=10,
        interpolation_method="linear",
        weights=1.0,
        interp_points_generator=above_mu_min_angular_generator(),
    )
    # Start from the 4th bin due to the average being a bit < 1 at low radii
    assert np.all(p_k_lin[3:] == 1.0)


def test_error_coords_and_mask():
    x = np.linspace(1.0, 3, 40)
    kpar_mesh, kperp_mesh = np.meshgrid(x, x)
    theta = np.arctan2(kperp_mesh, kpar_mesh)
    mu_mesh = np.cos(theta)

    mask = mu_mesh >= 0.97
    P = np.zeros(mask.shape)
    P[mask] = 1.0
    with pytest.raises(ValueError):
        p_k_lin, k_av_bins_lin = angular_average(
            P,
            [x, x],
            bins=10,
            interpolation_method="linear",
            weights=1.0,
            interp_points_generator=above_mu_min_angular_generator(mu=0.97),
        )


def test_interp_method():
    x = np.linspace(-3, 3, 40)
    P = np.ones((40, 40, 40))
    freq = [x, x, x]
    with pytest.raises(ValueError):
        ave, coord, var = angular_average_nd(
            P, freq, bins=20, get_variance=True, interpolation_method="abc"
        )

    with pytest.raises(ValueError):
        ave, coord, var = angular_average(
            P, freq, bins=20, get_variance=True, interpolation_method="abc"
        )


def test_error_w_kmag_coords():
    with pytest.raises(ValueError):
        x = np.linspace(-3, 3, 40)
        P = np.ones((40, 40, 40))
        X, Y = np.meshgrid(x, x)
        (
            ave,
            coord,
        ) = angular_average_nd(
            P, X**2 + Y**2, bins=20, interpolation_method="linear"
        )

    with pytest.raises(ValueError):
        x = np.linspace(-3, 3, 40)
        P = np.ones((40, 40, 40))
        X, Y = np.meshgrid(x, x)
        (
            ave,
            coord,
        ) = angular_average(P, X**2 + Y**2, bins=20, interpolation_method="linear")


def test_kmag_coords_nointerp():
    x = np.linspace(-3, 3, 40)
    P = np.ones((40, 40, 40))
    X, Y = np.meshgrid(x, x)
    with pytest.raises(ValueError):
        (
            ave,
            coord,
        ) = angular_average_nd(
            P, np.sqrt(X**2 + Y**2), bins=20, interpolation_method=None
        )
    with pytest.raises(ValueError):
        (
            ave,
            coord,
        ) = angular_average(
            P, np.sqrt(X**2 + Y**2), bins=20, interpolation_method=None
        )


@pytest.mark.parametrize("n", range(1, 3))
def test_angular_avg_nd(n):
    x = np.linspace(-3, 3, 40)
    X, Y, Z = np.meshgrid(x, x, x)
    r2 = X**2 + Y**2 + Z**2
    P = r2**-1.0

    # Test 4D avg works
    P = np.repeat(P, 10).reshape(40, 40, 40, 10)
    freq = [x, x, x, np.linspace(-2, 2, 10)]

    p_k_lin, k_av_bins_lin = angular_average_nd(
        P, freq, bins=10, n=n, interpolation_method="linear"
    )

    if n == 1:
        # Without interpolation, the radially-averaged power is not very accurate
        # due to the low number of bins at small values of k_av_bins, so we start
        # the comparison at the 6th bin.
        assert (
            np.max(
                np.abs(
                    (
                        p_k_lin[6:, len(x) // 2, len(x) // 2, 0]
                        - k_av_bins_lin[6:] ** -2.0
                    )
                    / k_av_bins_lin[6:] ** -2.0
                )
            )
            < 0.05
        )
    elif n == 2:
        assert (
            np.max(
                np.abs(
                    (p_k_lin[:, len(x) // 2, 0] - k_av_bins_lin**-2.0)
                    / k_av_bins_lin**-2.0
                )
            )
            < 0.05
        )
    else:
        assert (
            np.max(np.abs((p_k_lin[:, 0] - k_av_bins_lin**-2.0) / k_av_bins_lin**-2.0))
            < 0.05
        )


def test_angular_avg_nd_complex_interp():
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = r2**-1.0 + 1j * r2**-1.0
    P = np.repeat(P, 100).reshape(400, 400, 100)
    freq = [x, x, np.linspace(-2, 2, 100)]
    p_k_lin, k_av_bins_lin = angular_average_nd(
        P, freq, bins=50, n=2, interpolation_method="linear"
    )
    real = np.real(p_k_lin)
    imag = np.imag(p_k_lin)
    assert (
        np.max(np.abs((real[:, 0] - k_av_bins_lin**-2.0) / k_av_bins_lin**-2.0)) < 0.05
    )

    assert np.isclose(real, imag).all()


@pytest.mark.parametrize("interpolation_method", [None, "linear"])
def test_angular_avg_nd_4_2(interpolation_method):
    x = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = r2**-1.0
    P = np.repeat(P, 10).reshape(200, 200, 10)
    P = np.repeat(P, 10).reshape(200, 200, 10, 10)

    freq = [x, x, np.linspace(-2, 2, 10), np.linspace(-2, 2, 10)]
    p_k, k_av_bins = angular_average_nd(P, freq, bins=50, n=2)
    p_k_lin, k_av_bins_lin = angular_average_nd(
        P, freq, bins=50, n=2, interpolation_method=interpolation_method
    )
    # The radially-averaged power is not very accurate
    # due to the low number of bins at small values of k_av_bins, so we start
    # the comparison at the 6th bin.
    assert (
        np.max(
            np.abs(
                (p_k_lin[6:, 0, 0] - k_av_bins_lin[6:] ** -2.0)
                / k_av_bins_lin[6:] ** -2.0
            )
        )
        < 0.06
    )


def test_var_not_impl():
    x = np.linspace(-3, 3, 200)
    P = np.ones((200, 10))
    coords = [x, np.linspace(-2, 2, 10)]
    with pytest.raises(NotImplementedError):
        ave, coord, var = angular_average(
            P, coords, bins=20, get_variance=True, interpolation_method="linear"
        )
    with pytest.raises(NotImplementedError):
        ave, coord, var = angular_average_nd(
            P, coords, bins=20, get_variance=True, interpolation_method="linear"
        )


def test_angular_avg_nd_2_1_varnull():
    x = np.linspace(-3, 3, 200)

    P = np.ones((200, 10))

    coords = [x, np.linspace(-2, 2, 10)]
    p_k, k_av_bins, var, sw = angular_average_nd(
        P, coords, bins=20, n=1, get_variance=True, return_sumweights=True
    )

    assert np.all(var == 0)


def test_null_variance_2d():
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = np.ones_like(r2)
    ave, coord, var = angular_average(
        P, np.sqrt(r2), bins=np.linspace(0, x.max(), 20), get_variance=True
    )
    assert np.all(var == 0)


def test_variance_2d():
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = np.ones_like(r2)
    P += np.random.normal(scale=1, size=(len(x), len(x)))
    ave, coord, var = angular_average(
        P, np.sqrt(r2), bins=np.linspace(0, x.max(), 20), get_variance=True
    )
    assert np.all(np.diff(var) <= 0)


def test_complex_variance():
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = np.ones_like(r2) + np.ones_like(r2) * 1j
    with pytest.raises(NotImplementedError):
        ave, coord, var = angular_average(
            P, np.sqrt(r2), bins=np.linspace(0, x.max(), 20), get_variance=True
        )


def test_bin_edges():
    x = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = r2**-1.0
    bins = np.linspace(0, x.max(), 20)
    ave, coord = angular_average(P, np.sqrt(r2), bins=bins, bin_ave=False)
    assert np.all(coord == bins)


def test_sum():
    x = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = r2**-1.0
    ave, coord = angular_average(P, np.sqrt(r2), bins=20, bin_ave=False, average=False)
    assert np.sum(P[r2 < 9.0]) == np.sum(ave)

    ave, coord = angular_average(P, np.sqrt(r2), bins=20, bin_ave=True, average=False)
    assert np.sum(P[r2 < 9.0]) == np.sum(ave)


def test_var_trivial_weights():
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = np.ones_like(r2)
    P += np.random.normal(scale=1, size=(len(x), len(x)))
    ave, coord, var = angular_average(
        P,
        np.sqrt(r2),
        bins=np.linspace(0, x.max(), 20),
        get_variance=True,
        weights=np.ones_like(r2),
    )
    print(np.diff(var))
    assert np.all(np.diff(var) <= 1e-6)


def test_logbins():
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = np.ones_like(r2)
    ave, coord = angular_average(P, np.sqrt(r2), bins=10, bin_ave=False, log_bins=True)

    assert np.all(np.isclose(np.diff(coord[1:] / coord[:-1]), 0))


def test_cross_power_identity():
    pb = PowerBox(200, dim=2, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    dx = pb.delta_x()
    p, k = get_power(dx, pb.boxlength, b=1)
    p_cross, k = get_power(dx, pb.boxlength, b=1, deltax2=dx)
    assert np.all(np.isclose(p, p_cross))
    p, k = get_power(dx, [1, 1], b=1)
    p_cross, k = get_power(dx, [1, 1], b=1, deltax2=dx)
    assert np.all(np.isclose(p, p_cross))


@pytest.mark.skip()
def test_against_multirealisation():
    x = np.linspace(-3, 3, 1000)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    bins = np.linspace(0, x.max(), 20)

    # Get the variance from several realisations
    ave = [0] * 50
    for j in range(50):
        P = np.ones_like(r2) + np.random.normal(scale=1, size=(len(x), len(x)))
        ave[j], coord = angular_average(P, np.sqrt(r2), bins=bins)

    var = np.var(np.array(ave), axis=0)

    # Get the variance from a single realisation
    ave, coord, var2 = angular_average(P, np.sqrt(r2), bins=bins, get_variance=True)

    print(var)
    print(var2)
    assert np.all(np.isclose(var, var2, 1e-2))
