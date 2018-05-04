r"""
A module defining some "nicer" fourier transform functions.

We define only two functions -- an arbitrary-dimension forward transform, and its inverse. In each case, the transform
is designed to replicate the continuous transform. That is, the transform is volume-normalised and obeys correct
Fourier conventions.

The actual FFT backend is provided by ``pyFFTW`` if it is installed, which provides a significant speedup, and
multi-threading.

Notes
-----

Conveniently, we allow for arbitrary Fourier convention, according to the scheme in
http://mathworld.wolfram.com/FourierTransform.html. That is, we define the forward and inverse *n*-dimensional
transforms respectively as

.. math:: F(k) = \sqrt{\frac{|b|}{(2\pi)^{1-a}}}^n \int f(r) e^{-i b\mathbf{k}\cdot\mathbf{r}} d^n\mathbf{r}

and

.. math:: f(r) = \sqrt{\frac{|b|}{(2\pi)^{1+a}}}^n \int F(k) e^{+i b\mathbf{k}\cdot\mathbf{r}} d^n \mathbf{k}.

In both transforms, the corresponding co-ordinates are returned so a completely consistent transform is simple to get.
This makes switching from standard frequency to angular frequency very simple.

We note that currently, only positive values for b are implemented (in fact, using negative b is consistent, but
one must be careful that the frequencies returned are descending, rather than ascending).
"""
import warnings


# Try importing the pyFFTW interface
try:

    from multiprocessing import cpu_count
    THREADS = cpu_count()

    from pyfftw.interfaces.numpy_fft import fftn as _fftn, ifftn as _ifftn, ifftshift as _ifftshift, fftshift as _fftshift, fftfreq as _fftfreq
    from pyfftw.interfaces.cache import enable, set_keepalive_time
#    enable()
#    set_keepalive_time(100.)

    def fftn(*args,**kwargs):
        return _fftn(threads=THREADS,*args,**kwargs)

    def ifftn(*args, **kwargs):
        return _ifftn(threads=THREADS,*args, **kwargs)

    HAVE_FFTW = True

except ImportError:
    warnings.warn("You do not have pyFFTW installed. Installing it should give some speed increase.")
    HAVE_FFTW = False
    from numpy.fft import fftn, ifftn, ifftshift as _ifftshift, fftshift as _fftshift, fftfreq as _fftfreq

# NOTE: to avoid MKL-related bugs, numpy needs to be imported after pyfftw: see https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

def fft(X, L=None, Lk=None, a=0, b=2*np.pi, axes=None, ret_cubegrid=False):
    r"""
    Arbitrary-dimension nice Fourier Transform.

    This function wraps numpy's ``fftn`` and applies some nice properties. Notably, the returned fourier transform
    is equivalent to what would be expected from a continuous Fourier Transform (including normalisations etc.). In
    addition, arbitrary conventions are supported (see :mod:`powerbox.dft` for details).

    Default parameters return exactly what ``numpy.fft.fftn`` would return.

    The output object always has the zero in the centre, with monotonically increasing spectral arguments.

    Parameters
    ----------
    X : array
        An array with arbitrary dimensions defining the field to be transformed. Should correspond exactly
        to the continuous function for which it is an analogue. A lower-dimensional transform can be specified by using
        the ``axes`` argument.

    L : float or array-like, optional
        The length of the box which defines ``X``. If a scalar, each transformed dimension in ``X`` is assumed to have
        the same length. If array-like, must be of the same length as the number of transformed dimensions. The default
        returns the un-normalised DFT (same as numpy).

    Lk : float or array-like, optional
        The length of the fourier-space box which defines the dual of ``X``. Only one of L/Lk needs to be provided. If
        provided, L takes precedence. If a scalar, each transformed dimension in ``X`` is assumed to have
        the same length. If array-like, must be of the same length as the number of transformed dimensions.

    a,b : float, optional
        These define the Fourier convention used. See :mod:`powerbox.dft` for details. The defaults return the standard DFT
        as defined in :mod:`numpy.fft`.

    axes : sequence of ints, optional
        The axes to take the transform over. The default is to use all axes for the transform.

    ret_cubegrid : bool, optional
        Whether to return the entire grid of frequency magnitudes.

    Returns
    -------
    ft : array
        The DFT of X, normalised to be consistent with the continuous transform.

    freq : list of arrays
        The frequencies in each dimension, consistent with the Fourier conventions specified.

    grid : array
        Only returned if ``ret_cubegrid`` is ``True``. An array with shape given by ``axes`` specifying the magnitude
        of the frequencies at each point of the fourier transform.
    """


    if axes is None:
        axes = list(range(len(X.shape)))

    N = np.array([X.shape[axis] for axis in axes])

    # Get the box volume if given the fourier-space box volume
    if L is None and Lk is None:
        L = N
    elif L is not None:    # give precedence to L
        if np.isscalar(L):
            L = L*np.ones(len(axes))
    elif Lk is not None:
        if np.isscalar(Lk):
            Lk = Lk * np.ones(len(axes))
        L = N*2*np.pi/(Lk*b) # Take account of the fourier convention.

    V = float(np.product(L))    # Volume of box
    Vx = V/np.product(N) # Volume of cell

    ft = Vx*fftshift(fftn(X, axes=axes),axes=axes)*np.sqrt(np.abs(b)/(2*np.pi) ** (1 - a)) ** len(axes)

    dx = np.array([float(l)/float(n) for l, n in zip(L, N)])

    freq = np.array([fftfreq(n, d=d,b=b) for n, d in zip(N, dx)])
    if not ret_cubegrid:
        return ft, freq
    else:
        grid = freq[0] ** 2
        for i in range(len(axes) - 1):
            grid = np.add.outer(grid, freq[i+1] ** 2)

        return ft, freq, np.sqrt(grid)


def ifft(X, Lk=None,L=None, a=0, b=2*np.pi, axes=None,ret_cubegrid=False):
    r"""
    Arbitrary-dimension nice inverse Fourier Transform.

    This function wraps numpy's ``ifftn`` and applies some nice properties. Notably, the returned fourier transform
    is equivalent to what would be expected from a continuous inverse Fourier Transform (including normalisations etc.).
    In addition, arbitrary conventions are supported (see :mod:`powerbox.dft` for details).

    Default parameters return exactly what ``numpy.fft.ifftn`` would return.

    Parameters
    ----------
    X : array
        An array with arbitrary dimensions defining the field to be transformed. Should correspond exactly
        to the continuous function for which it is an analogue. A lower-dimensional transform can be specified by using
        the ``axes`` argument. Note that this should have its zero in the center.

    Lk : float or array-like, optional
        The length of the box which defines ``X``. If a scalar, each transformed dimension in ``X`` is assumed to have
        the same length. If array-like, must be of the same length as the number of transformed dimensions. The default
        returns the un-normalised DFT (the same as numpy).

    L : float or array-like, optional
        The length of the real-space box, defining the dual of ``X``. Only one of Lk/L needs to be passed. If L is
        passed, it is used. If a scalar, each transformed dimension in ``X`` is assumed to have
        the same length. If array-like, must be of the same length as the number of transformed dimensions. The default
        of ``Lk=1`` returns the un-normalised DFT.

    a,b : float, optional
        These define the Fourier convention used. See :mod:`powerbox.dft` for details. The defaults return the standard DFT
        as defined in :mod:`numpy.fft`.

    axes : sequence of ints, optional
        The axes to take the transform over. The default is to use all axes for the transform.

    ret_cubegrid : bool, optional
        Whether to return the entire grid of real-space co-ordinate magnitudes.

    Returns
    -------
    ft : array
        The IDFT of X, normalised to be consistent with the continuous transform.

    freq : list of arrays
        The real-space co-ordinate grid in each dimension, consistent with the Fourier conventions specified.

    grid : array
        Only returned if ``ret_cubegrid`` is ``True``. An array with shape given by ``axes`` specifying the magnitude
        of the real-space co-ordinates at each point of the inverse fourier transform.
    """


    if axes is None:
        axes = list(range(len(X.shape)))

    N = np.array([X.shape[axis] for axis in axes])

    # Get the box volume if given the real-space box volume
    if Lk is None and L is None:
        Lk = 1
    elif L is not None:
        if np.isscalar(L):
            L = np.array([L]*len(axes))

        dx = L/N
        Lk = 2*np.pi/(dx*b)

    elif np.isscalar(Lk):
        Lk = [Lk]*len(axes)

    Lk = np.array(Lk)

    V = np.product(Lk)
    dk = np.array([float(lk)/float(n) for lk, n in zip(Lk, N)])

    ft = V*ifftn(ifftshift(X,axes=axes), axes=axes)*np.sqrt(np.abs(b)/(2*np.pi) ** (1 + a)) ** len(axes)

    freq = np.array([fftfreq(n, d=d,b=b) for n, d in zip(N, dk)])

    if not ret_cubegrid:
        return ft, freq
    else:
        grid = freq[0] ** 2
        for i in range(len(axes) - 1):
            grid = np.add.outer(grid, freq[i] ** 2)

        return ft, freq, np.sqrt(grid)


def fftshift(x,*args,**kwargs):
    out = _fftshift(x,*args,**kwargs)

    if hasattr(x,"unit"):
        return out*x.unit
    else:
        return out


def ifftshift(x, *args, **kwargs):
    out = _ifftshift(x, *args, **kwargs)

    if hasattr(x, "unit"):
        return out * x.unit
    else:
        return out


def fftfreq(N,d=1.0,b=2*np.pi):
    """
    Return the fourier frequencies for a box with N cells, using general Fourier convention.

    Parameters
    ----------
    N : int
        The number of grid cells

    d : float, optional
        The interval between cells

    b : float, optional
        The fourier-convention of the frequency component (see :mod:`powerbox.dft` for details).

    Returns
    -------
    freq : array
        The N symmetric frequency components of the Fourier transform. Always centred at 0.

    """
    return fftshift(_fftfreq(N, d=d))*(2*np.pi/b)
