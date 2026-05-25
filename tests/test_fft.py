"""Tests for FFT conventions and round-trip behavior."""

import contextlib

import numpy as np
import pytest

from powerbox.dft import fft, fftfreq, fftshift, ifft, ifftshift, irfft
from powerbox.dft_backend import FFTW, NumpyFFT
from powerbox.tools import _magnitude_grid

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
HAVE_JAX = False

with contextlib.suppress(ValueError, ImportError):
    import pyfftw

    BACKENDS.append(FFTW(nthreads=1))
    HAVE_FFTW = True

    pyfftw.builders._utils._default_threads(4)

    BACKENDS.append(FFTW(nthreads=2))
    HAVE_FFTW_MULTITHREAD = True

with contextlib.suppress(ImportError):
    from powerbox.jax import dft as jax_dft

    HAVE_JAX = True


def gauss_ft(k, a, b, n=2):
    return (np.abs(b) / (2 * np.pi) ** (1 - a)) ** (n / 2.0) * np.exp(-(b**2) * k**2 / (4 * np.pi))


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


@pytest.mark.parametrize(("a", "b"), [(0, 2 * np.pi), (0, 1), (1, 1)])
@pytest.mark.parametrize("backend", BACKENDS)
def test_roundtrip_fb(g2d, a, b, backend) -> None:
    Fx, freq = fft(g2d["fx"], L=g2d["L"], a=a, b=b, x0=g2d["x"][0], backend=backend)

    Lk = -2 * np.min(freq)
    fx, x = ifft(Fx, Lk=Lk, a=a, b=b, x0=g2d["x"][0], backend=backend)
    assert np.max(fx.real - g2d["fx"]) < 1e-10  # Test FT result
    assert np.max(x[0] - g2d["x"]) < 1e-10  # Test x-grid


@pytest.mark.parametrize(("a", "b"), [(0, 2 * np.pi), (0, 1), (1, 1)])
@pytest.mark.parametrize("backend", BACKENDS)
def test_roundtrip_bf(g2d, a, b, backend) -> None:
    fx, freq = ifft(g2d["fx"], Lk=g2d["L"], a=a, b=b, x0=g2d["x"][0], backend=backend)

    L = -2 * np.min(freq)
    Fk, k = fft(fx, L=L, a=a, b=b, x0=g2d["x"][0], backend=backend)
    assert np.max(Fk.real - g2d["fx"]) < 1e-10  # Test FT result
    assert np.max(k[0] - g2d["x"]) < 1e-10  # Test x-grid


@pytest.mark.parametrize(("a", "b"), [(0, 2 * np.pi), (0, 1), (1, 1)])
@pytest.mark.parametrize("backend", BACKENDS)
def test_forward_only(g1d, a, b, backend) -> None:
    Fx, freq = fft(g1d["fx"], L=g1d["L"], a=a, b=b, x0=g1d["x"][0], backend=backend)
    assert np.max(np.abs(Fx.real - gauss_ft(freq[0], a, b, n=1))) < 1e-10


def analytic_mix(x, a, b, ainv, binv, n=2):
    return (binv / (b * (2 * np.pi) ** (ainv - a))) ** (n / 2.0) * gauss(binv * x / b)


@pytest.mark.parametrize(("a", "b", "ainv", "binv"), ABCOMBOS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_1d_fb(g1d, a, b, ainv, binv, backend) -> None:
    Fk, freq = fft(g1d["fx"], L=g1d["L"], a=a, b=b, backend=backend)
    Lk = -2 * np.min(freq)
    fx, x = ifft(Fk, Lk=Lk, a=ainv, b=binv, backend=backend)
    assert np.max(np.abs(fx.real - analytic_mix(x[0], a, b, ainv, binv, n=1))) < 1e-10


@pytest.mark.parametrize(("a", "b", "ainv", "binv"), ABCOMBOS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_1d_bf(g1d, a, b, ainv, binv, backend) -> None:
    Fk, freq = ifft(g1d["fx"], Lk=g1d["L"], a=ainv, b=binv, backend=backend)
    L = -2 * np.min(freq)
    fx, x = fft(Fk, L=L, a=a, b=b, backend=backend)
    assert np.max(np.abs(fx.real - analytic_mix(x[0], a, binv, ainv, b, n=1))) < 1e-10


@pytest.mark.parametrize(("a", "b", "ainv", "binv"), ABCOMBOS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_2d_fb(g2d, a, b, ainv, binv, backend) -> None:
    Fk, freq = fft(g2d["fx"], L=g2d["L"], a=a, b=b, backend=backend)
    Lk = -2 * np.min(freq)
    fx, _x = ifft(Fk, Lk=Lk, a=ainv, b=binv, backend=backend)
    xgrid = _magnitude_grid(_x)
    assert np.max(np.abs(fx.real - analytic_mix(xgrid, a, b, ainv, binv))) < 1e-10


NTHREADS_TO_CHECK = (None, 1, False)

if HAVE_FFTW_MULTITHREAD:
    NTHREADS_TO_CHECK += (2,)


@pytest.mark.skipif(not HAVE_FFTW, reason="pyFFTW not installed")
def test_fftw_backend_enables_interface_cache() -> None:
    """The FFTW backend should enable the pyFFTW interface cache for plan reuse."""
    import pyfftw

    pyfftw.interfaces.cache.disable()
    assert not pyfftw.interfaces.cache.is_enabled()

    FFTW(nthreads=1)

    assert pyfftw.interfaces.cache.is_enabled()


@pytest.mark.parametrize("shape", [(8,), (9,), (4, 5), (5, 4)])
@pytest.mark.parametrize("backend", BACKENDS)
def test_irfft_roundtrip_even_odd_shapes(shape, backend) -> None:
    """Round-trip real fields via full spectrum -> reduced spectrum -> irfft."""
    rng = np.random.default_rng(4)
    field = rng.normal(size=shape)
    boxlength = tuple(float(i + 2) for i in range(len(shape)))

    full_spectrum, _ = fft(field, L=boxlength, a=0, b=1, backend=backend)
    reduced = backend.ifftshift(full_spectrum, axes=(-1,))
    reduced = reduced[(slice(None),) * (field.ndim - 1) + (slice(None, shape[-1] // 2 + 1),)]

    reconstructed, _ = irfft(reduced, L=boxlength, a=0, b=1, N=shape, backend=backend)
    reference, _ = ifft(full_spectrum, L=boxlength, a=0, b=1, backend=backend)

    assert reconstructed.shape == shape
    np.testing.assert_allclose(reconstructed, reference.real, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("shape", [(8,), (9,), (6, 5)])
def test_irfft_rejects_n_inconsistent_with_reduced_shape(shape) -> None:
    """Explicit N must match the reduced-spectrum dimensions."""
    rng = np.random.default_rng(5)
    field = rng.normal(size=shape)
    full_spectrum, _ = fft(field, L=1.0, a=0, b=1, backend=NumpyFFT())
    reduced = np.fft.ifftshift(full_spectrum, axes=(-1,))
    reduced = reduced[(slice(None),) * (field.ndim - 1) + (slice(None, shape[-1] // 2 + 1),)]

    wrong_last = (*shape[:-1], shape[-1] + 2)
    with pytest.raises(ValueError, match="inconsistent with the reduced spectrum shape"):
        irfft(reduced, N=wrong_last, a=0, b=1, backend=NumpyFFT())

    if len(shape) > 1:
        wrong_nonlast = (shape[0] + 1, shape[1])
        with pytest.raises(ValueError, match="inconsistent with the reduced spectrum shape"):
            irfft(reduced, N=wrong_nonlast, a=0, b=1, backend=NumpyFFT())


@pytest.mark.parametrize("shape", [(6, 5), (5, 7)])
@pytest.mark.parametrize("backend", BACKENDS)
def test_irfft_phase_convention_matches_ifft_with_x0_and_bb(shape, backend) -> None:
    """Irfft and ifft should agree under non-zero x0 with explicit bb."""
    rng = np.random.default_rng(6)
    field = rng.normal(size=shape)
    boxlength = tuple(float(i + 2) for i in range(len(shape)))
    x0 = tuple(0.2 * (i + 1) for i in range(len(shape)))

    full_spectrum, _ = fft(field, L=boxlength, a=0, b=1, x0=x0, backend=backend)
    reduced = backend.ifftshift(full_spectrum, axes=(-1,))
    reduced = reduced[(slice(None),) * (field.ndim - 1) + (slice(None, shape[-1] // 2 + 1),)]

    via_irfft, _ = irfft(reduced, L=boxlength, a=0, b=1, x0=x0, bb=2.5, N=shape, backend=backend)
    via_ifft, _ = ifft(full_spectrum, L=boxlength, a=0, b=1, x0=x0, bb=2.5, backend=backend)

    np.testing.assert_allclose(via_irfft, via_ifft.real, rtol=1e-7, atol=1e-7)


@pytest.mark.skipif(not HAVE_JAX, reason="JAX backend not installed")
def test_irfft_jax_backend_roundtrip() -> None:
    """JAX irfft backend should match the NumPy reference for a fixed spectrum."""
    rng = np.random.default_rng(7)
    shape = (8, 7)
    field = rng.normal(size=shape)
    full_spectrum, _ = fft(field, L=(3.0, 4.0), a=0, b=1, backend=NumpyFFT())
    reduced = np.fft.ifftshift(full_spectrum, axes=(-1,))
    reduced = reduced[..., : shape[-1] // 2 + 1]

    reference, _ = irfft(reduced, L=(3.0, 4.0), a=0, b=1, N=shape, backend=NumpyFFT())
    jax_out, _ = jax_dft.irfft(reduced, L=(3.0, 4.0), a=0, b=1, N=shape)

    np.testing.assert_allclose(np.asarray(jax_out), reference, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(("a", "b", "ainv", "binv"), ABCOMBOS)
@pytest.mark.parametrize("nthreads", NTHREADS_TO_CHECK)
def test_mixed_2d_bf(g2d, a, b, ainv, binv, nthreads) -> None:
    Fk, freq = ifft(g2d["fx"], Lk=g2d["L"], a=ainv, b=binv, x0=g2d["x"][0], nthreads=nthreads)
    L = -2 * np.min(freq)
    fx, _x = fft(Fk, L=L, a=a, b=b, x0=g2d["x"][0], nthreads=nthreads)
    xgrid = _magnitude_grid(_x)
    assert np.max(np.abs(fx.real - analytic_mix(xgrid, a, binv, ainv, b))) < 1e-10


@pytest.mark.parametrize("nthreads", NTHREADS_TO_CHECK)
def test_fftshift(nthreads) -> None:
    x = np.linspace(0, 1, 11)

    y = fftshift(ifftshift(x, nthreads=nthreads), nthreads=nthreads)
    assert np.all(x == y)


@pytest.mark.parametrize("nthreads", NTHREADS_TO_CHECK)
@pytest.mark.parametrize("n", [10, 11])
def test_fftfreq(nthreads, n) -> None:
    freqs = fftfreq(n, nthreads=nthreads)
    assert np.all(np.diff(freqs)) > 0


@pytest.mark.parametrize("nthreads", [False, True])
def test_get_fft_backend_bool_returns_numpy(nthreads) -> None:
    """Passing a bool to get_fft_backend should always return the NumpyFFT backend."""
    from powerbox.dft_backend import NumpyFFT, get_fft_backend

    backend = get_fft_backend(nthreads)
    assert isinstance(backend, NumpyFFT)


def test_bad_x0_raises():
    """Test that passing an x0 of the wrong shape raises an error."""
    with pytest.raises(ValueError, match="x0 must be a scalar or have the same length"):
        fft(np.array([1, 2, 3]), L=10, a=0, b=1, x0=(0, 1))


def test_multidim_x0():
    res = fft(np.array([[1, 2], [3, 4]]), L=10, a=0, b=1, x0=(0, 0))[0]
    res2 = fft(np.array([[1, 2], [3, 4]]), L=10, a=0, b=1)[0]

    assert np.allclose(res, res2)
