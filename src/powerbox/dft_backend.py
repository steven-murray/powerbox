"""FFT backends."""

import numpy as np
import warnings
from functools import cache

try:
    import pyfftw
except ImportError:
    pass


class FFTBackend:
    """Base class for FFT backends."""

    def ifftn(self):
        """Abstract method for the ifftn."""
        pass

    def fftn(self):
        """Abstract method for the fftn."""
        pass

    def fftshift(self, x, *args, **kwargs):
        """
        The same as numpy, except that it preserves units (if Astropy quantities are used).

        All extra arguments are passed directly to numpy's ``fftshift``.
        """
        out = self._fftshift(x, *args, **kwargs)

        return out * x.unit if hasattr(x, "unit") else out

    def ifftshift(self, x, *args, **kwargs):
        """
        The same as numpy except it preserves units (if Astropy quantities are used).

        All extra arguments are passed directly to numpy's ``ifftshift``.
        """
        out = self._ifftshift(x, *args, **kwargs)

        return out * x.unit if hasattr(x, "unit") else out

    def fftfreq(self, N, d=1.0, b=2 * np.pi):
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


class NumpyFFT(FFTBackend):
    """FFT backend using numpy.fft."""

    def __init__(self):
        self.fftn = np.fft.fftn

        self.ifftn = np.fft.ifftn

        self._fftshift = np.fft.fftshift
        self._ifftshift = np.fft.ifftshift
        self._fftfreq = np.fft.fftfreq

        self.empty = np.empty
        self.have_fftw = False


class FFTW(FFTBackend):
    """FFT backend using pyfftw."""

    def __init__(self, nthreads=None):
        if nthreads is None:
            from multiprocessing import cpu_count

            nthreads = cpu_count()

        self.nthreads = nthreads
        try:
            import pyfftw
        except ImportError:
            raise ImportError("pyFFTW could not be imported...")

        self._fftshift = pyfftw.interfaces.numpy_fft.fftshift
        self._ifftshift = pyfftw.interfaces.numpy_fft.ifftshift
        self._fftfreq = pyfftw.interfaces.numpy_fft.fftfreq
        self.empty = pyfftw.empty_aligned

    def ifftn(self, *args, **kwargs):
        """Inverse fast fourier transform."""
        return pyfftw.interfaces.numpy_fft.ifftn(*args, threads=self.nthreads, **kwargs)

    def fftn(self, *args, **kwargs):
        """Fast fourier transform."""
        return pyfftw.interfaces.numpy_fft.fftn(*args, threads=self.nthreads, **kwargs)


@cache
def get_fft_backend(nthreads=None):
    """Choose a backend based on nthreads.

    Will return the Numpy backend if nthreads is None, otherwise the FFTW backend with
    the given number of threads.
    """
    if nthreads is None or nthreads > 0:
        try:
            fftbackend = FFTW(nthreads=nthreads)
        except ImportError:
            if nthreads is not None:
                warnings.warn(
                    "Could not import pyfftw... Proceeding with numpy.", stacklevel=2
                )
            fftbackend = NumpyFFT()
    else:
        fftbackend = NumpyFFT()
    return fftbackend
