"""FFT backends."""

from __future__ import annotations

import contextlib
import warnings
from abc import ABC
from functools import cache
from multiprocessing import cpu_count

import numpy as np

with contextlib.suppress(ImportError):
    import pyfftw


class FFTBackend(ABC):  # noqa: B024
    """Abstract base class for FFT backends."""

    def fftshift(self, x, *args, **kwargs):
        """
        Apply ``numpy.fftshift`` while preserving units when present.

        All extra arguments are passed directly to numpy's ``fftshift``.
        """
        out = self._fftshift(x, *args, **kwargs)

        return out * x.unit if hasattr(x, "unit") else out

    def ifftshift(self, x, *args, **kwargs):
        """
        Apply ``numpy.ifftshift`` while preserving units when present.

        All extra arguments are passed directly to numpy's ``ifftshift``.
        """
        out = self._ifftshift(x, *args, **kwargs)

        return out * x.unit if hasattr(x, "unit") else out

    def fftfreq(self, N: int, d: float = 1.0, b: float = 2 * np.pi):
        """
        Return fourier frequencies for a box with N cells, using general Fourier convention.

        Parameters
        ----------
        N : int
            The number of grid cells
        d : float, optional
            The interval between cells
        b : float, optional
            The fourier-convention of the frequency component (see :mod:`powerbox.dft` for
            details).

        Returns
        -------
        freq : array
            The N symmetric frequency components of the Fourier transform. Always centred at 0.
        """
        return self.fftshift(self._fftfreq(N, d=d)) * (2 * np.pi / b)

    def rfftfreq(self, N: int, d: float = 1.0, b: float = 2 * np.pi):
        """
        Return the non-negative Fourier frequencies for a real FFT axis.

        Parameters
        ----------
        N : int
            The number of grid cells along the real-space axis.
        d : float, optional
            The interval between cells.
        b : float, optional
            The Fourier-convention coefficient (see :mod:`powerbox.dft` for details).

        Returns
        -------
        freq : array
            The non-negative Fourier frequencies corresponding to ``numpy.fft.rfftfreq``.
        """
        return self._rfftfreq(N, d=d) * (2 * np.pi / b)


class NumpyFFT(FFTBackend):
    """FFT backend using numpy.fft."""

    def __init__(self) -> None:
        self.xp = np
        self.fftn = np.fft.fftn
        self.ifftn = np.fft.ifftn
        self.rfftn = np.fft.rfftn
        self.irfftn = np.fft.irfftn
        self._fftshift = np.fft.fftshift
        self._ifftshift = np.fft.ifftshift
        self._fftfreq = np.fft.fftfreq
        self._rfftfreq = np.fft.rfftfreq
        self.empty = np.empty
        self.have_fftw = False
        self.nthreads = 1


class FFTW(FFTBackend):
    """FFT backend using pyfftw."""

    _CACHE_KEEPALIVE_SECONDS = 60.0

    def __init__(self, nthreads: int | None = None) -> None:
        try:
            import pyfftw
        except ImportError as err:
            raise ImportError("pyFFTW could not be imported...") from err

        try:
            pyfftw.builders._utils._default_threads(4)
        except ValueError:
            if nthreads and nthreads > 1:
                warnings.warn(
                    "pyFFTW was not installed with multithreading. Using 1 thread.",
                    stacklevel=2,
                )
            nthreads = 1

        if nthreads is None:
            nthreads = cpu_count()

        if not pyfftw.interfaces.cache.is_enabled():
            pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(self._CACHE_KEEPALIVE_SECONDS)

        self.xp = np
        self.nthreads = nthreads

        self._fftshift = pyfftw.interfaces.numpy_fft.fftshift
        self._ifftshift = pyfftw.interfaces.numpy_fft.ifftshift
        self._fftfreq = pyfftw.interfaces.numpy_fft.fftfreq
        self.empty = pyfftw.empty_aligned

    def ifftn(self, *args, **kwargs):
        """Inverse fast fourier transform."""
        return pyfftw.interfaces.numpy_fft.ifftn(*args, threads=self.nthreads, **kwargs)

    def irfftn(self, *args, **kwargs):
        """Inverse real fast fourier transform."""
        return pyfftw.interfaces.numpy_fft.irfftn(*args, threads=self.nthreads, **kwargs)

    def fftn(self, *args, **kwargs):
        """Fast fourier transform."""
        return pyfftw.interfaces.numpy_fft.fftn(*args, threads=self.nthreads, **kwargs)

    def rfftn(self, *args, **kwargs):
        """Real fast fourier transform."""
        return pyfftw.interfaces.numpy_fft.rfftn(*args, threads=self.nthreads, **kwargs)

    def _rfftfreq(self, *args, **kwargs):
        """Real FFT frequencies."""
        return pyfftw.interfaces.numpy_fft.rfftfreq(*args, **kwargs)


class JaxFFT(FFTBackend):
    """FFT backend using ``jax.numpy.fft``."""

    def __init__(self) -> None:
        try:
            import jax.numpy as jnp
        except ImportError as err:  # pragma: no cover - exercised when jax is unavailable
            raise ImportError("JAX could not be imported.") from err

        self.xp = jnp
        self.fftn = jnp.fft.fftn
        self.ifftn = jnp.fft.ifftn
        self.rfftn = jnp.fft.rfftn
        self.irfftn = jnp.fft.irfftn
        self._fftshift = jnp.fft.fftshift
        self._ifftshift = jnp.fft.ifftshift
        self._fftfreq = jnp.fft.fftfreq
        self._rfftfreq = jnp.fft.rfftfreq
        self.empty = jnp.empty
        self.have_fftw = False
        self.nthreads = 1


@cache
def get_fft_backend(nthreads: int | None = None) -> FFTW | NumpyFFT:
    """Choose a backend based on nthreads.

    Will return the FFTW backend when ``nthreads`` is ``None`` or greater than one, and
    otherwise the NumPy backend.
    """
    # Handle bool explicitly: False → 0 (numpy backend), True → 1 (numpy backend).
    # Explicit conversion avoids implicit boolean-to-integer coercion in the
    # comparison below, making the intent clear rather than relying on True == 1.
    if isinstance(nthreads, bool):
        nthreads = int(nthreads)

    if nthreads is None or nthreads > 1:
        try:
            fftbackend = FFTW(nthreads=nthreads)
        except ImportError:
            if nthreads is not None:
                warnings.warn("Could not import pyfftw... Proceeding with numpy.", stacklevel=2)
            fftbackend = NumpyFFT()
    else:
        fftbackend = NumpyFFT()
    return fftbackend
