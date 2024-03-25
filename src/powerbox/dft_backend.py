import numpy as np
import warnings
from functools import cache

try:
    import pyfftw
except ImportError:
    pass


class FFTBackend:
    def ifftn(self):
        pass

    def fftn(self):
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
    def __init__(self):
        self.fftn = np.fft.fftn

        self.ifftn = np.fft.ifftn

        self._fftshift = np.fft.fftshift
        self._ifftshift = np.fft.ifftshift
        self._fftfreq = np.fft.fftfreq

        self.empty = np.empty
        self.have_fftw = False

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
        return np.fft.fftshift(np.fft.fftfreq(N, d=d)) * (2 * np.pi / b)


class FFTW(FFTBackend):
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
        self.have_fftw = True

    def ifftn(self, *args, **kwargs):
        return pyfftw.interfaces.numpy_fft.ifftn(*args, threads=self.nthreads, **kwargs)

    def fftn(self, *args, **kwargs):
        return pyfftw.interfaces.numpy_fft.fftn(*args, threads=self.nthreads, **kwargs)


@cache
def get_fft_backend(nthreads):
    if nthreads is None or nthreads > 0:
        try:
            fftbackend = FFTW(nthreads=nthreads)
        except ImportError:
            warnings.warn("Could not import pyfftw... Proceeding with numpy.")
            fftbackend = NumpyFFT()
    else:
        fftbackend = NumpyFFT()
    return fftbackend
