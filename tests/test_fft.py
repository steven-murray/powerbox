import pytest

import contextlib
import numpy as np

from powerbox.dft import fft, fftfreq, fftshift, ifft, ifftshift
from powerbox.dft_backend import FFTW, JAXFFT, NumpyFFT

ABCOMBOS = [
    (0, 2 * np.pi, 0, 1),
    (0, 2 * np.pi, 1, 1),
    (0, 1, 1, 2 * np.pi),
    (0, 1, 1, 1),
    (1, 1, 0, 2 * np.pi),
    (1, 1, 0, 1),
]

BACKENDS = [
    NumpyFFT(),
]

HAVE_FFTW = False
HAVE_FFTW_MULTITHREAD = False

with contextlib.suppress(ValueError, ImportError):
    import pyfftw

    BACKENDS.append(FFTW(nthreads=1))
    HAVE_FFTW = True

    pyfftw.builders._utils._default_threads(4)

    BACKENDS.append(FFTW(nthreads=2))
    HAVE_FFTW_MULTITHREAD = True


def gauss_ft(k, a, b, n=2):
    return (np.abs(b) / (2 * np.pi) ** (1 - a)) ** (n / 2.0) * np.exp(
        -(b**2) * k**2 / (4 * np.pi)
    )


def gauss(x):
    return np.exp(-np.pi * x**2)


@pytest.fixture(scope="module")
def g2d():
    N = 1000
    L = 10.0
    dx = L / N
    x = np.arange(-L / 2, L / 2, dx)[:N]
    xgrid = np.sqrt(np.add.outer(x**2, x**2))
    fx = gauss(xgrid)
    return {"L": L, "fx": fx, "x": x}


@pytest.fixture(scope="module")
def g1d():
    N = 1000
    L = 10.0
    dx = L / N
    x = np.arange(-L / 2, L / 2, dx)[:N]
    fx = np.exp(-np.pi * x**2)
    return {"L": L, "fx": fx, "x": x}


@pytest.mark.parametrize("a,b", [(0, 2 * np.pi), (0, 1), (1, 1)])
@pytest.mark.parametrize("backend", BACKENDS)
def test_roundtrip_fb(g2d, a, b, backend):
    Fx, freq = fft(
        g2d["fx"], L=g2d["L"], a=a, b=b, left_edge=-g2d["L"] / 2, backend=backend
    )

    Lk = -2 * np.min(freq)
    fx, x = ifft(Fx, Lk=Lk, a=a, b=b, backend=backend)
    assert np.max(fx.real - g2d["fx"]) < 1e-10  # Test FT result
    assert np.max(x[0] - g2d["x"]) < 1e-10  # Test x-grid


@pytest.mark.parametrize("a,b", [(0, 2 * np.pi), (0, 1), (1, 1)])
@pytest.mark.parametrize("backend", BACKENDS)
def test_roundtrip_bf(g2d, a, b, backend):
    fx, freq = ifft(g2d["fx"], Lk=g2d["L"], a=a, b=b, backend=backend)

    L = -2 * np.min(freq)
    Fk, k = fft(fx, L=L, a=a, b=b, backend=backend)
    assert np.max(Fk.real - g2d["fx"]) < 1e-10  # Test FT result
    assert np.max(k[0] - g2d["x"]) < 1e-10  # Test x-grid


@pytest.mark.parametrize("a,b", [(0, 2 * np.pi), (0, 1), (1, 1)])
@pytest.mark.parametrize("backend", BACKENDS)
def test_forward_only(g1d, a, b, backend):
    Fx, freq = fft(
        g1d["fx"], L=g1d["L"], a=a, b=b, left_edge=-g1d["L"] / 2, backend=backend
    )
    assert np.max(np.abs(Fx.real - gauss_ft(freq[0], a, b, n=1))) < 1e-10


def analytic_mix(x, a, b, ainv, binv, n=2):
    return (binv / (b * (2 * np.pi) ** (ainv - a))) ** (n / 2.0) * gauss(binv * x / b)


@pytest.mark.parametrize("a,b, ainv, binv", ABCOMBOS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_1d_fb(g1d, a, b, ainv, binv, backend):
    Fk, freq = fft(
        g1d["fx"], L=g1d["L"], a=a, b=b, left_edge=-g1d["L"] / 2, backend=backend
    )
    Lk = -2 * np.min(freq)
    fx, x = ifft(Fk, Lk=Lk, a=ainv, b=binv, backend=backend)
    assert np.max(np.abs(fx.real - analytic_mix(x[0], a, b, ainv, binv, n=1))) < 1e-10


@pytest.mark.parametrize("a,b, ainv, binv", ABCOMBOS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_1d_bf(g1d, a, b, ainv, binv, backend):
    Fk, freq = ifft(g1d["fx"], Lk=g1d["L"], a=ainv, b=binv, backend=backend)
    L = -2 * np.min(freq)
    fx, x = fft(Fk, L=L, a=a, b=b, left_edge=-L / 2, backend=backend)
    assert np.max(np.abs(fx.real - analytic_mix(x[0], a, binv, ainv, b, n=1))) < 1e-10


@pytest.mark.parametrize("a,b, ainv, binv", ABCOMBOS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_2d_fb(g2d, a, b, ainv, binv, backend):
    Fk, freq = fft(
        g2d["fx"], L=g2d["L"], a=a, b=b, left_edge=-g2d["L"] / 2, backend=backend
    )
    Lk = -2 * np.min(freq)
    fx, x, xgrid = ifft(Fk, Lk=Lk, a=ainv, b=binv, ret_cubegrid=True, backend=backend)
    assert np.max(np.abs(fx.real - analytic_mix(xgrid, a, b, ainv, binv))) < 1e-10


NTHREADS_TO_CHECK = (None, 1, False)

if HAVE_FFTW_MULTITHREAD:
    NTHREADS_TO_CHECK += (2,)


@pytest.mark.parametrize("a,b, ainv, binv", ABCOMBOS)
@pytest.mark.parametrize("nthreads", NTHREADS_TO_CHECK)
def test_mixed_2d_bf(g2d, a, b, ainv, binv, nthreads):
    Fk, freq = ifft(g2d["fx"], Lk=g2d["L"], a=ainv, b=binv, nthreads=nthreads)
    L = -2 * np.min(freq)
    fx, x, xgrid = fft(
        Fk, L=L, a=a, b=b, left_edge=-L / 2, ret_cubegrid=True, nthreads=nthreads
    )
    assert np.max(np.abs(fx.real - analytic_mix(xgrid, a, binv, ainv, b))) < 1e-10


@pytest.mark.parametrize("nthreads", NTHREADS_TO_CHECK)
def test_fftshift(nthreads):
    x = np.linspace(0, 1, 11)

    y = fftshift(ifftshift(x, nthreads=nthreads), nthreads=nthreads)
    assert np.all(x == y)


@pytest.mark.parametrize("nthreads", NTHREADS_TO_CHECK)
@pytest.mark.parametrize("n", (10, 11))
def test_fftfreq(nthreads, n):
    freqs = fftfreq(n, nthreads=nthreads)
    assert np.all(np.diff(freqs)) > 0
