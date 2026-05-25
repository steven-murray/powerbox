r"""
A module defining some "nicer" fourier transform functions.

We define only two functions: an arbitrary-dimension forward transform, and its
inverse. In each case, the transform is designed to replicate the continuous
transform. That is, the transform is volume-normalised and obeys correct Fourier
conventions.

The actual FFT backend is provided by ``pyFFTW`` if it is installed, which
provides a significant speedup and multi-threading.

Conveniently, we allow for arbitrary Fourier convention, according to the scheme
in http://mathworld.wolfram.com/FourierTransform.html. That is, we define the
forward and inverse *n*-dimensional transforms respectively as

.. math::
    F(k) = \sqrt{\frac{|b|}{(2\pi)^{1-a}}}^n
    \int f(r) e^{-i b\mathbf{k}\cdot\mathbf{r}} d^n\mathbf{r}

and

.. math::
    f(r) = \sqrt{\frac{|b|}{(2\pi)^{1+a}}}^n
    \int F(k) e^{+i b\mathbf{k}\cdot\mathbf{r}} d^n \mathbf{k}.

In both transforms, the corresponding co-ordinates are returned so a completely
consistent transform is simple to get. This makes switching from standard
frequency to angular frequency very simple.

We note that currently, only positive values for ``b`` are implemented. Using
negative ``b`` is consistent, but one must be careful that the frequencies
returned are descending, rather than ascending.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["fft", "fftfreq", "fftshift", "ifft", "ifftshift", "irfft", "rfftfreq"]

# To avoid MKL-related bugs, numpy needs to be imported after pyfftw: see https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
import numpy.typing as npt

from .dft_backend import get_fft_backend


def fftshift(x, *args, **kwargs):  # noqa: D103
    backend = kwargs.pop("backend", get_fft_backend(kwargs.pop("nthreads", None)))
    return backend.fftshift(x, *args, **kwargs)


fftshift.__doc__ = get_fft_backend().fftshift.__doc__


def ifftshift(x, *args, **kwargs):  # noqa: D103
    backend = kwargs.pop("backend", get_fft_backend(kwargs.pop("nthreads", None)))
    return backend.ifftshift(x, *args, **kwargs)


ifftshift.__doc__ = get_fft_backend().ifftshift.__doc__


def fftfreq(x, *args, **kwargs):  # noqa: D103
    backend = kwargs.pop("backend", get_fft_backend(kwargs.pop("nthreads", None)))
    return backend.fftfreq(x, *args, **kwargs)


fftfreq.__doc__ = get_fft_backend().fftfreq.__doc__


def rfftfreq(x, *args, **kwargs):  # noqa: D103
    backend = kwargs.pop("backend", get_fft_backend(kwargs.pop("nthreads", None)))
    return backend.rfftfreq(x, *args, **kwargs)


rfftfreq.__doc__ = get_fft_backend().rfftfreq.__doc__


def fft(
    X,
    L: float | np.typing.ArrayLike | None = None,
    Lk: float | np.typing.ArrayLike | None = None,
    a: float = 0,
    b: float = 2 * np.pi,
    x0: float | tuple[float, ...] = 0,
    axes: Sequence[int] | None = None,
    nthreads=None,
    backend=None,
):
    r"""
    Arbitrary-dimension nice Fourier Transform.

    This function wraps numpy's ``fftn`` and applies some nice properties. Notably, the
    returned fourier transform is equivalent to what would be expected from a continuous
    Fourier Transform (including normalisations etc.). In addition, arbitrary
    conventions are supported (see :mod:`powerbox.dft` for details).

    Default parameters have the same normalising conventions as ``numpy.fft.fftn``.

    The output object always has the zero frequency in the centre, with monotonically
    increasingspectral arguments.

    Parameters
    ----------
    X : array
        An array with arbitrary dimensions defining the field to be transformed. Should
        correspond exactly to the continuous function for which it is an analogue. A
        lower-dimensional transform can be specified by using the ``axes`` argument.
    L : float or array-like, optional
        The length of the box which defines ``X``. If a scalar, each transformed
        dimension in ``X`` is assumed to have the same length. If array-like, must be of
        the same length as the number of transformed dimensions. The default returns the
        un-normalised DFT (same as numpy).
    Lk : float or array-like, optional
        The length of the fourier-space box which defines the dual of ``X``. Only one of
        L/Lk needs to be provided. If provided, L takes precedence. If a scalar, each
        transformed dimension in ``X`` is assumed to have the same length. If
        array-like, must be of the same length as the number of transformed dimensions.
    a,b : float, optional
        These define the Fourier convention used. See :mod:`powerbox.dft` for details.
        The defaults return the standard DFT as defined in :mod:`numpy.fft`.
    x0 : float or tuple[float, ...], optional
        The co-ordinate of the first sample for each dimension that is being transformed.
        This is useful when using the fft to approximate a continuous fourier transform
        of a function defined on a finite support, and the precise phases of the result
        are important. By the Fourier shift theorem, changing this argument simply
        changes the phases of the result, but does not affect the magnitudes. For a
        standard DFT to approximate a continuous fourier transform, the first sample is
        assumed to be at 0.
    axes : sequence of ints, optional
        The axes to take the transform over. The default is to use all axes for the
        transform.
    nthreads : int, optional
        Number of threads for pyFFTW. If set to None, uses pyFFTW with the number of
        threads equal to the number of available CPUs. If set to 0 or 1, uses numpy's
        FFT routine instead. If set to an integer greater than 1, uses pyFFTW with that
        many threads.
    backend : FFTBackend, optional
        The backend to use for the FFT. If not provided, the backend is chosen based on
        the value of nthreads.

    Returns
    -------
    ft : array
        The DFT of X, normalised to be consistent with the continuous transform.
    freq : list of arrays
        The frequencies in each dimension, consistent with the Fourier conventions
        specified.
    """
    if backend is None:
        backend = get_fft_backend(nthreads)
    xp = getattr(backend, "xp", np)

    if axes is None:
        axes = tuple(range(X.ndim))

    N = np.array([X.shape[axis] for axis in axes])

    # Get the box volume if given the fourier-space box volume
    if L is None and Lk is None:
        L = N
    elif L is not None:  # give precedence to L
        if np.isscalar(L):
            L = L * np.ones(len(axes))
    else:
        if np.isscalar(Lk):
            Lk = Lk * np.ones(len(axes))
        L = N * 2 * np.pi / (Lk * b)  # Take account of the fourier convention.

    V = float(np.prod(L))  # Volume of box
    Vx = V / np.prod(N)  # Volume of cell

    ft = (
        Vx
        * backend.fftshift(backend.fftn(X, axes=axes), axes=axes)
        * xp.sqrt(xp.abs(b) / (2 * xp.pi) ** (1 - a)) ** len(axes)
    )

    dx = np.array([float(length) / float(n) for length, n in zip(L, N, strict=True)])

    freq = [backend.fftfreq(n, d=d, b=b) for n, d in zip(N, dx, strict=True)]

    # Adjust phases of the result to align with the left edge properly.
    # In the default case, don't adjust phases at all -- i.e. by default let this
    # function do the same thing as np.fft. When the fft is interpreted as an
    # approximation to a continuous Fourier transform, the first bin *centre* is at zero,
    # (which makes the left edge at -dx/2). This is a bit weird, but the standard
    # convention really doesn't care about the continuous FT, so it doesn't matter.
    # We allow the user to set left_edge in case their objective really is to take a FT
    # of some discretized function between some finiite boundaries.
    if x0 != 0:
        bin_centre = _set_x0(x0, axes)
        ft = _adjust_phase(ft, bin_centre, freq, axes, b, xp)

    return ft, freq


def ifft(
    X,
    Lk=None,
    L=None,
    a: float = 0,
    b: float = 2 * np.pi,
    axes: Sequence[int] | None = None,
    x0: float | tuple[float, ...] = 0,
    nthreads=None,
    backend=None,
    bb: float | None = None,
) -> tuple[npt.NDArray[complex], list[np.ndarray]]:
    r"""
    Arbitrary-dimension nice inverse Fourier Transform.

    This function wraps numpy's ``ifftn`` and applies some nice properties. Notably,
    the returned fourier transform is equivalent to what would be expected from a
    continuous inverse Fourier Transform (including normalisations etc.). In addition,
    arbitrary conventions are supported (see :mod:`powerbox.dft` for details).

    Default parameters have the same normalising conventions as ``numpy.fft.ifftn``.

    Parameters
    ----------
    X : array
        An array with arbitrary dimensions defining the field to be transformed. Should
        correspond exactly to the continuous function for which it is an analogue. A
        lower-dimensional transform can be specified by using the ``axes`` argument.
        Note that if using a non-periodic function, the co-ordinates should be
        monotonically increasing.
    Lk : float or array-like, optional
        The length of the box which defines ``X``. If a scalar, each transformed
        dimension in ``X`` is assumed to have the same length. If array-like, must be of
        the same length as the number of transformed dimensions. The default returns the
        un-normalised DFT (the same as numpy).
    L : float or array-like, optional
        The length of the real-space box, defining the dual of ``X``. Only one of Lk/L
        needs to be passed. If L is passed, it is used. If a scalar, each transformed
        dimension in ``X`` is assumed to have the same length. If array-like, must be of
        the same length as the number of transformed dimensions. The default of ``Lk=1``
        returns the un-normalised DFT.
    a,b : float, optional
        These define the Fourier convention used. See :mod:`powerbox.dft` for details.
        The defaults return the standard DFT as defined in :mod:`numpy.fft`.
    axes : sequence of ints, optional
        The axes to take the transform over. The default is to use all axes for the
        transform.
    x0 : float or array-like, optional
        The co-ordinate of the first sample (in real-space) for each dimension that is being
        transformed. This affects only the phases of the result. By default, don't apply
        any additional phase shift to the result of the standard numpy ifft. This is
        equivalent to the first sample being at x=0.
    nthreads : int, optional
        Number of threads for pyFFTW. If set to None, uses pyFFTW with the number of
        threads equal to the number of available CPUs. If set to 0 or 1, uses numpy's
        FFT routine instead. If set to an integer greater than 1, uses pyFFTW with that
        many threads.
    backend : FFTBackend, optional
        The backend to use for the FFT. If not provided, the backend is chosen based on
        the value of nthreads.

    Returns
    -------
    ft : array
        The IDFT of X, normalised to be consistent with the continuous transform.
    freq : list of arrays
        The real-space co-ordinate grid in each dimension, consistent with the Fourier
        conventions specified.
    """
    if backend is None:
        backend = get_fft_backend(nthreads)
    xp = getattr(backend, "xp", np)

    if axes is None:
        axes = tuple(range(len(X.shape)))

    N = np.array([X.shape[axis] for axis in axes])

    # Get the box volume if given the real-space box volume
    if Lk is None and L is None:
        Lk = [1] * len(axes)
    elif L is not None:
        if np.isscalar(L):
            L = np.array([L] * len(axes))

        dx = L / N
        Lk = 2 * np.pi / (dx * b)

    elif np.isscalar(Lk):
        Lk = [Lk] * len(axes)

    dx = np.array([2 * np.pi / (lk * b) for lk in Lk])
    Lk = np.array(Lk)

    V = np.prod(Lk)
    dk = np.array([float(lk) / float(n) for lk, n in zip(Lk, N, strict=True)])

    if x0 != 0:
        X = X.astype(complex)  # Ensure we can apply the phase shift without erroring if X is real.
        x0 = _set_x0(x0, axes)
        if bb is None:
            bb = b
        freq = [backend.fftfreq(n, d=d, b=bb) for n, d in zip(N, dx, strict=True)]
        X = _adjust_phase(X, x0, freq, axes, -bb, xp)

    X = backend.ifftshift(X, axes=axes)
    ft = V * backend.ifftn(X, axes=axes) * xp.sqrt(xp.abs(b) / (2 * xp.pi) ** (1 + a)) ** len(axes)

    freq = [backend.fftfreq(n, d=d, b=b) for n, d in zip(N, dk, strict=True)]

    return ft, freq


def irfft(
    X,
    Lk=None,
    L=None,
    a: float = 0,
    b: float = 2 * np.pi,
    axes: Sequence[int] | None = None,
    x0: float | tuple[float, ...] = 0,
    nthreads=None,
    backend=None,
    bb: float | None = None,
    N: int | Sequence[int] | None = None,
):
    r"""
    Arbitrary-dimension nice inverse Fourier Transform for half-Hermitian spectra.

    This function is the real-FFT analogue of :func:`ifft`. The input spectrum is
    assumed to be stored in ``rfftn`` layout: all transformed axes except the final
    one are centred, and the final transformed axis contains only the non-negative
    frequencies in the standard ``rfftn`` order.

    Parameters
    ----------
    X, Lk, L, a, b, axes, x0, nthreads, backend, bb
        As for :func:`ifft`.
    N : int or sequence of int, optional
        Real-space output shape along the transformed axes. This should be provided
        whenever the final transformed axis has odd length, because the half-spectrum
        shape alone cannot distinguish odd from even real-space sizes.

    Returns
    -------
    ft : array
        The inverse transform of ``X``, normalised to be consistent with the continuous
        transform.
    freq : list of arrays
        The real-space co-ordinate grid in each transformed dimension.
    """
    if backend is None:
        backend = get_fft_backend(nthreads)
    xp = getattr(backend, "xp", np)

    if axes is None:
        axes = tuple(range(len(X.shape)))

    if N is None:
        real_shape = [X.shape[axis] for axis in axes[:-1]]
        real_shape.append(2 * (X.shape[axes[-1]] - 1))
    elif np.isscalar(N):
        real_shape = [int(N)] * len(axes)
    else:
        real_shape = [int(n) for n in N]
        if len(real_shape) != len(axes):
            raise ValueError("N must be a scalar or have the same length as the number of axes.")

    N = np.array(real_shape)

    if Lk is None and L is None:
        Lk = [1] * len(axes)
    elif L is not None:
        if np.isscalar(L):
            L = np.array([L] * len(axes))

        dx = L / N
        Lk = 2 * np.pi / (dx * b)
    elif np.isscalar(Lk):
        Lk = [Lk] * len(axes)

    dx = np.array([2 * np.pi / (lk * b) for lk in Lk])
    Lk = np.array(Lk)

    V = np.prod(Lk)
    dk = np.array([float(lk) / float(n) for lk, n in zip(Lk, N, strict=True)])

    fft_axes = axes[:-1]
    freq = [
        backend.fftfreq(n, d=d, b=bb if bb is not None else b)
        for n, d in zip(N[:-1], dx[:-1], strict=True)
    ]
    freq.append(backend.rfftfreq(N[-1], d=dx[-1], b=bb if bb is not None else b))

    if x0 != 0:
        X = X.astype(complex)
        x0 = _set_x0(x0, axes)
        X = _adjust_phase(X, x0, freq, axes, -(bb if bb is not None else b), xp)

    if fft_axes:
        X = backend.ifftshift(X, axes=fft_axes)

    ft = (
        V
        * backend.irfftn(X, s=tuple(int(n) for n in N), axes=axes)
        * xp.sqrt(xp.abs(b) / (2 * xp.pi) ** (1 + a)) ** len(axes)
    )

    freq = [backend.fftfreq(n, d=d, b=b) for n, d in zip(N, dk, strict=True)]

    return ft, freq


def _adjust_phase(
    ft: np.ndarray,
    x0: tuple[float, ...],
    freq: list[np.ndarray],
    axes: tuple[int, ...],
    b: float,
    xp=np,
):
    """Apply a phase shift to the Fourier transform to adjust for the left edge.

    The default DFT assumes that the first sample is at x=0. If the first sample is at
    some other co-ordinate, then the Fourier shift theorem tells us that the Fourier
    transform is multiplied by a phase factor of exp(-i b k x0), where k is the
    frequency and x0 is the co-ordinate of the first sample. This function applies this
    phase shift to the Fourier transform.
    """
    for i, (ledge, fq) in enumerate(zip(x0, freq, strict=True)):
        phase = xp.exp(-b * 1j * fq * ledge)
        obj = (
            *([None] * axes[i]),
            slice(None, None, None),
            *([None] * (ft.ndim - axes[i] - 1)),
        )
        ft = ft * phase[obj]

    return ft


def _set_x0(x0: float | tuple[float, ...], axes: tuple[int, ...]) -> tuple[float, ...]:
    if np.isscalar(x0):
        return (x0,) * len(axes)
    else:
        if len(x0) != len(axes):
            raise ValueError(
                "x0 must be a scalar or have the same length as the number of dimensions."
            )
        return tuple(xx for xx in x0)
