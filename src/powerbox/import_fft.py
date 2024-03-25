import warnings
import numpy as np


def config(THREADS=None):

    # Try importing the pyFFTW interface
    if THREADS is None:
        if THREADS is None:
            from multiprocessing import cpu_count
            THREADS = cpu_count()
    if THREADS > 0:
        try:
            #warnings.warn("Using pyFFTW with " + str(THREADS) + " threads...")
            from pyfftw.interfaces.cache import enable, set_keepalive_time
            from pyfftw.interfaces.numpy_fft import fftfreq as _fftfreq
            from pyfftw.interfaces.numpy_fft import fftn as _fftn
            from pyfftw.interfaces.numpy_fft import fftshift as _fftshift
            from pyfftw.interfaces.numpy_fft import ifftn as _ifftn
            from pyfftw.interfaces.numpy_fft import ifftshift as _ifftshift
            from pyfftw import empty_aligned as empty

            def fftn(*args, **kwargs):
                return _fftn(*args, threads=THREADS, **kwargs)

            def ifftn(*args, **kwargs):
                return _ifftn(*args, threads=THREADS, **kwargs)

            HAVE_FFTW = True

        except ImportError:
            HAVE_FFTW = False
            #warnings.warn("USE_FFTW set to True but pyFFTW could not be loaded. Make sure pyFFTW is installed properly. Proceeding with numpy...", UserWarning)
            from numpy.fft import fftfreq as _fftfreq
            from numpy.fft import fftn
            from numpy.fft import fftshift as _fftshift
            from numpy.fft import ifftn
            from numpy.fft import ifftshift as _ifftshift
            empty = np.empty
    else:
        HAVE_FFTW = False
        #warnings.warn("Using numpy FFT...")
        from numpy.fft import fftfreq as _fftfreq
        from numpy.fft import fftn
        from numpy.fft import fftshift as _fftshift
        from numpy.fft import ifftn
        from numpy.fft import ifftshift as _ifftshift
        empty = np.empty

    def fftshift(x, *args, **kwargs):
        """
        The same as numpy, except that it preserves units (if Astropy quantities are used).

        All extra arguments are passed directly to numpy's ``fftshift``.
        """
        out = _fftshift(x, *args, **kwargs)

        return out * x.unit if hasattr(x, "unit") else out


    def ifftshift(x, *args, **kwargs):
        """
        The same as numpy except it preserves units (if Astropy quantities are used).

        All extra arguments are passed directly to numpy's ``ifftshift``.
        """
        out = _ifftshift(x, *args, **kwargs)

        return out * x.unit if hasattr(x, "unit") else out


    def fftfreq(N, d=1.0, b=2 * np.pi):
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
        return fftshift(_fftfreq(N, d=d)) * (2 * np.pi / b)
    return fftn, ifftn, fftfreq, fftshift, ifftshift, empty, HAVE_FFTW