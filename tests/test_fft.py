import numpy as np
import pytest

from powerbox.dft import fft, ifft

ABCOMBOS = [
    (0, 2 * np.pi, 0, 1),
    (0, 2 * np.pi, 1, 1),
    (0, 1, 1, 2 * np.pi),
    (0, 1, 1, 1),
    (1, 1, 0, 2 * np.pi),
    (1, 1, 0, 1),
]


def gauss_ft(k, a, b, n=2):
    return (np.abs(b) / (2 * np.pi) ** (1 - a)) ** (n / 2.) * np.exp(-b ** 2 * k ** 2 / (4 * np.pi))


def gauss(x):
    return np.exp(-np.pi * x ** 2)


@pytest.fixture(scope="module")
def g2d():
    N = 1000
    L = 10.
    dx = L / N
    x = np.arange(-L / 2, L / 2, dx)[:N]
    xgrid = np.sqrt(np.add.outer(x ** 2, x ** 2))
    fx = gauss(xgrid)
    return {"L": L, "fx": fx, "x": x}


@pytest.fixture(scope="module")
def g1d():
    N = 1000
    L = 10.
    dx = L / N
    x = np.arange(-L / 2, L / 2, dx)[:N]
    fx = np.exp(-np.pi * x ** 2)
    return {"L": L, "fx": fx, "x": x}


@pytest.mark.parametrize('a,b', [(0, 2 * np.pi), (0, 1), (1, 1)])
def test_roundtrip_fb(g2d, a, b):
    Fx, freq = fft(g2d['fx'], L=g2d['L'], a=a, b=b, left_edge=-g2d['L']/2)

    Lk = -2 * np.min(freq)
    fx, x = ifft(Fx, Lk=Lk, a=a, b=b)
    assert np.max((fx.real - g2d['fx'])) < 1e-10  # Test FT result
    assert np.max(x[0] - g2d['x']) < 1e-10  # Test x-grid


@pytest.mark.parametrize('a,b', [(0, 2 * np.pi), (0, 1), (1, 1)])
def test_roundtrip_bf(g2d, a, b):
    fx, freq = ifft(g2d['fx'], Lk=g2d['L'], a=a, b=b)

    L = -2 * np.min(freq)
    Fk, k = fft(fx, L=L, a=a, b=b)
    assert np.max((Fk.real - g2d['fx'])) < 1e-10  # Test FT result
    assert np.max(k[0] - g2d['x']) < 1e-10  # Test x-grid


@pytest.mark.parametrize('a,b', [(0, 2 * np.pi), (0, 1), (1, 1)])
def test_forward_only(g1d, a, b):
    Fx, freq = fft(g1d['fx'], L=g1d['L'], a=a, b=b, left_edge=-g1d['L'] / 2)
    assert np.max(np.abs(Fx.real - gauss_ft(freq[0], a, b, n=1))) < 1e-10


def analytic_mix(x, a, b, ainv, binv, n=2):
    return (binv / (b * (2 * np.pi) ** (ainv - a))) ** (n / 2) * gauss(binv * x / b)


@pytest.mark.parametrize("a,b, ainv, binv", ABCOMBOS)
def test_mixed_1d_fb(g1d, a, b, ainv, binv):
    Fk, freq = fft(g1d['fx'], L=g1d['L'], a=a, b=b, left_edge=-g1d['L'] / 2)
    Lk = -2 * np.min(freq)
    fx, x = ifft(Fk, Lk=Lk, a=ainv, b=binv)
    assert np.max(np.abs(fx.real - analytic_mix(x[0], a, b, ainv, binv, n=1))) < 1e-10


@pytest.mark.parametrize("a,b, ainv, binv", ABCOMBOS)
def test_mixed_1d_bf(g1d, a, b, ainv, binv):
    Fk, freq = ifft(g1d['fx'], Lk=g1d['L'], a=ainv, b=binv)
    L = -2 * np.min(freq)
    fx, x = fft(Fk, L=L, a=a, b=b, left_edge=-L / 2)
    assert np.max(np.abs(fx.real - analytic_mix(x[0], a, binv, ainv, b, n=1))) < 1e-10


@pytest.mark.parametrize("a,b, ainv, binv", ABCOMBOS)
def test_mixed_2d_fb(g2d, a, b, ainv, binv):
    Fk, freq = fft(g2d['fx'], L=g2d['L'], a=a, b=b, left_edge=-g2d['L'] / 2)
    Lk = -2 * np.min(freq)
    fx, x, xgrid = ifft(Fk, Lk=Lk, a=ainv, b=binv, ret_cubegrid=True)
    assert np.max(np.abs(fx.real - analytic_mix(xgrid, a, b, ainv, binv))) < 1e-10


@pytest.mark.parametrize("a,b, ainv, binv", ABCOMBOS)
def test_mixed_2d_bf(g2d, a, b, ainv, binv):
    Fk, freq = ifft(g2d['fx'], Lk=g2d['L'], a=ainv, b=binv)
    L = -2 * np.min(freq)
    fx, x, xgrid = fft(Fk, L=L, a=a, b=b, left_edge=-L / 2, ret_cubegrid=True)
    assert np.max(np.abs(fx.real - analytic_mix(xgrid, a, binv, ainv, b))) < 1e-10

#
# def test_mixed_unitary_ordinary_unitary_angular_2d_fb():
#     N = 1000
#     L = 10.
#     dx = L / N
#     x = np.arange(-L / 2, L / 2, dx)[:N]
#     X, Y = np.meshgrid(x, x)
#
#     a_squared = 1
#     fx = np.exp(-np.pi * a_squared * (X ** 2 + Y ** 2))
#
#     Fk, freq = fft(fx, L=L, a=0, b=2 * np.pi)
#     Lk = -2 * np.min(freq)
#     fx_, x_, xgrid = ifft(Fk, Lk=Lk, a=0, b=1, ret_cubegrid=True)
#
#     fx_, bins = angular_average(fx_, xgrid, 200)
#
#     print(np.max(np.abs(fx_ - np.exp(-np.pi * (bins / (2 * np.pi)) ** 2) / (2 * np.pi))))
#     assert np.max(np.abs(fx_ - np.exp(-np.pi * (bins / (2 * np.pi)) ** 2) / (2 * np.pi))) < 1e-4
#
#
# def test_mixed_unitary_ordinary_unitary_angular_2d_bf():
#     N = 1000
#     Lk = 10.
#     dk = Lk / N
#     k = np.arange(-Lk / 2, Lk / 2, dk)[:N]
#     KX, KY = np.meshgrid(k, k)
#
#     alpha = np.pi
#     Fk = np.exp(-alpha * (KX ** 2 + KY ** 2))
#
#     fx, x = ifft(Fk, Lk=Lk, a=0, b=1)
#     Fk_, k_, kgrid = fft(fx, -2 * np.min(x), a=0, b=2 * np.pi, ret_cubegrid=True)
#
#     Fk_, bins = angular_average(Fk_, kgrid, 200)
#     assert np.max(np.abs(Fk_ - 2 * np.pi * np.exp(-alpha * ((2 * np.pi) * bins) ** 2)))
