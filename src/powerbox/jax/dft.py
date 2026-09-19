"""JAX-backed Fourier transform wrappers."""

from __future__ import annotations

from .. import dft as _base_dft
from ..dft_backend import JaxFFT

__all__ = [
    "fft",
    "fftfreq",
    "fftshift",
    "get_fft_backend",
    "ifft",
    "ifftshift",
    "irfft",
    "rfftfreq",
]


def get_fft_backend() -> JaxFFT:
    """Return the JAX FFT backend used by :mod:`powerbox.jax`."""
    return JaxFFT()


def fftshift(x, *args, **kwargs):  # noqa: D103
    kwargs.setdefault("backend", get_fft_backend())
    return _base_dft.fftshift(x, *args, **kwargs)


def ifftshift(x, *args, **kwargs):  # noqa: D103
    kwargs.setdefault("backend", get_fft_backend())
    return _base_dft.ifftshift(x, *args, **kwargs)


def fftfreq(x, *args, **kwargs):  # noqa: D103
    kwargs.setdefault("backend", get_fft_backend())
    return _base_dft.fftfreq(x, *args, **kwargs)


def rfftfreq(x, *args, **kwargs):  # noqa: D103
    kwargs.setdefault("backend", get_fft_backend())
    return _base_dft.rfftfreq(x, *args, **kwargs)


def fft(*args, **kwargs):  # noqa: D103
    kwargs.setdefault("backend", get_fft_backend())
    return _base_dft.fft(*args, **kwargs)


def ifft(*args, **kwargs):  # noqa: D103
    kwargs.setdefault("backend", get_fft_backend())
    return _base_dft.ifft(*args, **kwargs)


def irfft(*args, **kwargs):  # noqa: D103
    kwargs.setdefault("backend", get_fft_backend())
    return _base_dft.irfft(*args, **kwargs)


fftshift.__doc__ = _base_dft.fftshift.__doc__
ifftshift.__doc__ = _base_dft.ifftshift.__doc__
fftfreq.__doc__ = _base_dft.fftfreq.__doc__
rfftfreq.__doc__ = _base_dft.rfftfreq.__doc__
fft.__doc__ = _base_dft.fft.__doc__
ifft.__doc__ = _base_dft.ifft.__doc__
irfft.__doc__ = _base_dft.irfft.__doc__
