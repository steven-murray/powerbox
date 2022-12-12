import pytest

import numpy as np

from powerbox.powerbox import PowerBox
from powerbox.tools import angular_average, angular_average_nd, get_power


def test_angular_avg_nd_3():
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = r2**-1.0
    P = np.repeat(P, 100).reshape(400, 400, 100)
    freq = [x, x, np.linspace(-2, 2, 100)]
    p_k, k_av_bins = angular_average_nd(P, freq, bins=50, n=2)
    print(p_k[6:, 0], k_av_bins[6:] ** -2.0)
    assert (
        np.max(np.abs((p_k[6:, 0] - k_av_bins[6:] ** -2.0) / k_av_bins[6:] ** -2.0))
        < 0.05
    )


def test_angular_avg_nd_4_2():
    x = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    P = r2**-1.0
    P = np.repeat(P, 10).reshape(200, 200, 10)
    P = np.repeat(P, 10).reshape(200, 200, 10, 10)

    freq = [x, x, np.linspace(-2, 2, 10), np.linspace(-2, 2, 10)]
    p_k, k_av_bins = angular_average_nd(P, freq, bins=50, n=2)

    print(np.abs((p_k[7:, 0, 0] - k_av_bins[7:] ** -2.0) / k_av_bins[7:] ** -2.0))
    assert (
        np.max(np.abs((p_k[6:, 0, 0] - k_av_bins[6:] ** -2.0) / k_av_bins[6:] ** -2.0))
        < 0.06
    )


def test_angular_avg_nd_2_1_varnull():
    x = np.linspace(-3, 3, 200)

    P = np.ones((200, 10))

    coords = [x, np.linspace(-2, 2, 10)]
    p_k, k_av_bins, var = angular_average_nd(P, coords, bins=20, n=1, get_variance=True)

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
    print(np.diff(var))
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
    assert np.sum(P[r2 < 18.0]) == np.sum(ave)


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
