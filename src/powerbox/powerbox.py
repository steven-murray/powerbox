"""Classes that can create arbitrary-dimensional fields with given power spectra.

One such function produces *Gaussian* fields, and the other *LogNormal* fields.

In principle, these may be extended to other 1-point density distributions by
subclassing :class:`PowerBox` and over-writing the same methods as are over-written in
:class:`LogNormalPowerBox`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import cached_property

import attrs
import numpy as np
from attrs import validators as vld

from . import dft
from ._fft_layout import full_spectrum_to_rfft, irfft_to_field
from ._hermitianity import hermitianize_rfft_array
from .dft_backend import FFTBackend
from .tools import _magnitude_grid


@attrs.define(kw_only=True, slots=False, frozen=True)
class PowerBox:
    r"""
    Generate real- and fourier-space Gaussian fields with a given power spectrum.

    Parameters
    ----------
    N : int or sequence of int
        Number of grid-points on each side of the resulting box (equivalently, number of
        wavenumbers to use). If a scalar, the same number is used along every axis. If a
        sequence, it must have length ``dim``.
    pk : callable
        A callable of a single (vector) variable `k`, which is the isotropic power
        spectrum. The relationship of the `k` of which this is a function to the
        real-space co-ordinates, `x`, is determined by the parameters ``a,b``.
    dim : int, default 2
        Number of dimensions of resulting box.
    boxlength : float or sequence of float, default 1.0
        Length of the final signal along each axis. This may have arbitrary units, so
        long as `pk` is a function of a variable which has the inverse units. If a
        scalar, the same length is used along every axis. If a sequence, it must have
        length ``dim``.
    ensure_physical : bool, optional
        Interpreting the power spectrum as a spectrum of density fluctuations, the
        minimum physical value of the real-space field, :meth:`delta_x`, is -1. With
        ``ensure_physical`` set to ``True``, :meth:`delta_x` is clipped to return values
        >-1. If this is happening a lot, consider using :class:`LogNormalPowerBox`.
    a,b : float, optional
        These define the Fourier convention used. See :mod:`powerbox.dft` for details.
        The defaults define the standard usage in *cosmology* (for example, as defined
        in Cosmological Physics, Peacock, 1999, pg. 496.). Standard numerical usage
        (eg. numpy) is (a,b) = (0,2pi).
    vol_normalised_power : bool, optional
        Whether the input power spectrum, ``pk``, is volume-weighted. Default True
        because of standard cosmological usage.
    seed: int, optional
        A random seed to define the initial conditions. If not set, it will remain
        random, and each call to eg. :meth:`delta_x()` will produce a *different*
        realisation.
    nthreads : int, optional
        Number of threads for pyFFTW. If set to None, uses pyFFTW with the number of
        threads equal to the number of available CPUs. If set to 0 or 1, uses numpy's
        FFT routine instead. If set to an integer greater than 1, uses pyFFTW with that
        many threads.

    Notes
    -----
    A number of conventions need to be listed.

    The conventions of using `x` for "real-space" and `k` for "fourier space" arise from
    cosmology, but this does not affect anything -- `x` could just as well stand for
    "time domain" and `k` for "frequency domain".

    The important convention is the relationship between `x` and `k`, or in other words,
    whether `k` is interpreted as an angular frequency or ordinary frequency. By
    default, because of cosmological conventions, `k` is an angular frequency, so that
    the fourier transform integrand is delta_k*exp(-ikx). The conventions can be changed
    arbitrarily by setting the ``a,b`` parameters (see :mod:`powerbox.dft` for details).

    The primary quantity of interest is :meth:`delta_x`, which is a zero-mean Gaussian
    field with a power spectrum equivalent to that which was input. Being zero-mean
    enables its direct interpretation as an overdensity field, and this interpretation
    is enforced in the :meth:`make_discrete_sample` method.

    When scalar ``N`` and scalar ``boxlength`` are provided, the public attributes
    ``N``, ``boxlength``, ``x``, and ``kvec`` retain their historical scalar/1-D forms.
    When either quantity is specified per-axis, ``x`` and ``kvec`` return tuples of
    1-D arrays, one for each axis.

    .. note:: None of the n-dimensional arrays that are created within the class are
              stored, due to the inefficiency in memory consumption that this would
              imply. Thus, each large array is created and *returned* by their
              respective method, to be stored/discarded by the user.

    .. warning:: Due to the above note, repeated calls to eg. :meth:`delta_x()` will
                 produce *different* realisations of the real-space field, unless the
                 `seed` parameter is set in the constructor.

    Examples
    --------
    To create a 3-dimensional box of gaussian over-densities, gridded into 100 bins,
    with cosmological conventions, and a power-law power spectrum, simply use

    >>> pb = PowerBox(100,lambda k : 0.1*k**-3., dim=3, boxlength=100.0)
    >>> overdensities = pb.delta_x()
    >>> grid = pb.x
    >>> radii = pb.r

    To create a 2D turbulence structure, with arbitrary units, once can use

    >>> import matplotlib.pyplot as plt
    >>> pb = PowerBox(1000, lambda k : k**-7./5.)
    >>> plt.imshow(pb.delta_x())

    To create a 2D non-cubic box with different resolutions and side lengths:

    >>> pb = PowerBox((128, 192), lambda k: (1 + k) ** -2.0, dim=2, boxlength=(200.0, 600.0))
    >>> field = pb.delta_x()
    >>> x, y = pb.x
    """

    N: int | None = attrs.field(
        default=None, converter=attrs.converters.optional(int), validator=vld.optional(vld.gt(0))
    )
    _dim: int | None = attrs.field(
        converter=attrs.converters.optional(int), default=None, validator=vld.optional(vld.gt(0))
    )
    shape: tuple[int, ...] = attrs.field(converter=lambda x: tuple(int(i) for i in x))

    _pk: Callable[[np.ndarray], np.ndarray] = attrs.field(repr=False)

    _boxlength: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float), validator=vld.optional(vld.gt(0))
    )
    size: tuple[float, ...] = attrs.field(converter=lambda x: tuple(float(i) for i in x))

    ensure_physical: bool = attrs.field(default=False, converter=bool)
    fourier_a: float = attrs.field(default=1.0, converter=float, alias="a")
    fourier_b: float = attrs.field(default=1.0, converter=float, alias="b")
    vol_normalised_power: bool = attrs.field(default=True, converter=bool)
    nthreads: int | None = attrs.field(
        default=None, converter=attrs.converters.optional(int), validator=vld.optional(vld.gt(0))
    )
    seed: int | None = attrs.field(default=None, converter=attrs.converters.optional(int))

    fftbackend: FFTBackend = attrs.field()

    @N.validator
    def _validate_N(self, attribute: attrs.Attribute, value: int | None) -> None:
        """Validate the N parameter."""
        if value is not None:
            warnings.warn(
                "The `N` parameter is deprecated in favor of `shape`. Support for `N` "
                "will be removed in v1.2.",
                DeprecationWarning,
                stacklevel=2,
            )

    @shape.default
    def _shape_default(self) -> tuple[int, ...]:
        """Default the shape parameter to the value of N."""
        if self.N is None:
            raise ValueError("You must provide 'shape'")
        elif self._dim is None:
            # If using N, have two dimensions by default
            return (self.N, self.N)
        else:
            return (self.N,) * self._dim

    @shape.validator
    def _validate_shape(self, attribute: attrs.Attribute, value: tuple[int, ...]) -> None:
        """Validate the shape parameter."""
        if self._dim is not None and len(value) != self._dim:
            raise ValueError(f"shape must have same length as dim ({self._dim}), but got {value}.")
        if self.N is not None and any(axis_n != self.N for axis_n in value):
            raise ValueError(f"Don't provide both N and shape. Got N={self.N} and shape={value}.")
        for axis_n in value:
            if axis_n <= 0:
                raise ValueError("All elements of shape must be positive integers.")

    @property
    def dim(self) -> int:
        """The number of spatial dimensions."""
        return len(self.shape)

    @_boxlength.validator
    def _validate_boxlength(self, attribute: attrs.Attribute, value: float | None) -> None:
        """Validate the boxlength parameter."""
        if value is not None:
            warnings.warn(
                "The `boxlength` parameter is deprecated in favor of `size`. Support "
                "for `boxlength` will be removed in v1.2.",
                DeprecationWarning,
                stacklevel=2,
            )

    @size.default
    def _size_default(self) -> tuple[float, ...]:
        """Default the size parameter to the value of boxlength."""
        if self._boxlength is None:
            return (1.0,) * self.dim
        return (float(self._boxlength),) * self.dim

    @size.validator
    def _validate_size(self, attribute: attrs.Attribute, value: tuple[float, ...]) -> None:
        """Validate the size parameter."""
        if self._boxlength is not None and any(length != self._boxlength for length in value):
            raise ValueError(
                f"Don't provide both boxlength and size. Got boxlength={self._boxlength} "
                f"and size={value}."
            )
        for length in value:
            if length <= 0:
                raise ValueError("All elements of size must be positive real numbers.")
        if len(value) != self.dim:
            raise ValueError(f"size must have same length as dim ({self.dim}), but got {value}.")

    @property
    def L(self) -> tuple[float, ...]:
        """Alias for ``boxlength``."""
        warnings.warn(
            "The `L` attribute is deprecated and will be removed in v1.2. Use `boxlength` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.size

    def pk(self, k: np.ndarray) -> np.ndarray:
        """Return the input power spectrum evaluated at k."""
        if self.vol_normalised_power:
            return self._pk(k) / self.volume
        return self._pk(k)

    @property
    def volume(self) -> float:
        """The physical volume of the box."""
        return float(np.prod(self.size))

    @property
    def V(self) -> float:
        """The physical volume of the box.

        Deprecated alias for `volume`. Will be removed in v1.2. Use `volume` instead.
        """
        warnings.warn(
            "The `V` attribute is deprecated and will be removed in v1.2. Use `volume` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.volume

    @property
    def total_ncells(self) -> int:
        """The total number of grid cells in the box."""
        return int(np.prod(self.shape))

    @property
    def Ntot(self) -> int:
        """The total number of grid cells in the box.

        Deprecated alias for `total_ncells`. Will be removed in v1.2. Use `total_ncells`
        instead.
        """
        warnings.warn(
            "The `Ntot` attribute is deprecated and will be removed in v1.2. Use "
            "`total_ncells` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.total_ncells

    @property
    def dx(self) -> tuple[float, ...]:
        """The physical grid spacing along each axis."""
        return tuple(length / axis_n for length, axis_n in zip(self.size, self.shape, strict=True))

    @fftbackend.default
    def _default_fftbackend(self) -> FFTBackend:
        return dft.get_fft_backend(self.nthreads)

    @cached_property
    def rng(self) -> np.random.Generator:
        """The random number generator used for creating the fields."""
        return np.random.default_rng(self.seed)

    @property
    def x(self) -> tuple[np.ndarray, ...]:
        """The co-ordinates of the grid along each axis."""
        return tuple(
            np.arange(-length / 2, length / 2, axis_dx)[:axis_n]
            for length, axis_dx, axis_n in zip(self.size, self.dx, self.shape, strict=True)
        )

    @property
    def kvec(self) -> tuple[np.ndarray, ...]:
        """The reduced wavenumber vectors for the half-Hermitian spectrum."""
        axes = [
            self.fftbackend.fftfreq(axis_n, d=axis_dx, b=self.fourier_b)
            for axis_n, axis_dx in zip(self.shape[:-1], self.dx[:-1], strict=True)
        ]
        axes.append(self.fftbackend.rfftfreq(self.shape[-1], d=self.dx[-1], b=self.fourier_b))
        return tuple(axes)

    @property
    def _rfft_shape(self) -> tuple[int, ...]:
        """Shape of the half-Hermitian spectrum compatible with ``irfftn``."""
        return (*self.shape[:-1], self.shape[-1] // 2 + 1)

    def _irfft_to_field(self, spectrum, scale: float):
        """Transform a reduced half-spectrum into a real-space field."""
        return irfft_to_field(
            spectrum,
            scale=scale,
            irfft_function=dft.irfft,
            L=self.size,
            a=self.fourier_a,
            b=self.fourier_b,
            N=self.shape,
            backend=self.fftbackend,
        )

    def _full_spectrum_to_rfft(self, spectrum: np.ndarray) -> np.ndarray:
        """Convert a centred full spectrum to reduced rFFT layout."""
        return full_spectrum_to_rfft(
            spectrum,
            dim=self.dim,
            rfft_last_axis_size=self._rfft_shape[-1],
            backend=self.fftbackend,
        )

    def k(self):
        """Return the full grid of wavenumber magnitudes."""
        return _magnitude_grid(list(self.kvec))

    @property
    def r(self):
        """The radial position of every point in the grid."""
        return _magnitude_grid(list(self.x))

    def gauss_hermitian(self):
        """Return reduced Hermitian Gaussian modes sampled directly in rFFT layout."""
        modes = (
            self.rng.normal(0, 1, size=self._rfft_shape)
            + 1j * self.rng.normal(0, 1, size=self._rfft_shape)
        ) / np.sqrt(2)

        hermitianize_rfft_array(modes, has_nyquist=self.shape[-1] % 2 == 0)

        return modes

    def power_array(self):
        """Return the volume-normalized power spectrum evaluated on ``self.k``."""
        k = self.k()
        mask = k != 0
        # Re-use the k array to conserve memory
        k[mask] = self.pk(k[mask])
        return k

    def delta_k(self):
        """Return a realization of ``delta_k``.

        The gaussianised square root of the power spectrum (i.e. the Fourier
        co-efficients).
        """
        p = self.power_array()

        if np.any(p < 0):
            raise ValueError("The power spectrum function has returned negative values.")

        gh = self.gauss_hermitian()
        gh[...] = np.sqrt(p) * gh
        return gh

    def delta_x(self, delta_k: np.ndarray | None = None):
        """Return the realized real-space field from the input power spectrum."""
        # Here we multiply by V because the inverse Fourier transform of the
        # dimensionless power has units of 1/V, and we require a unitless
        # quantity for delta_x.
        dk = self.delta_k() if delta_k is None else delta_k
        dk = self._irfft_to_field(dk, scale=self.volume)

        if self.ensure_physical:
            np.clip(dk, -1, np.inf, dk)

        return dk

    def create_discrete_sample(
        self,
        nbar: float,
        randomise_in_cell: bool = True,
        min_at_zero: bool = False,
        delta_x=None,
    ):
        r"""Create a sample of tracers of the underlying density distribution.

        This function assumes that the real-space signal represents an over-density
        with respect to some mean,.

        Parameters
        ----------
        nbar : float
            Mean tracer density within the box.
        randomise_in_cell : bool, optional
            Whether to randomise the positions of the tracers within the cells, or put
            them at the grid-points (more efficient).
        min_at_zero : bool, optional
            Whether to make the lower corner of the box at the origin, otherwise the
            centre of the box is at the origin.
        delta_x : numpy.ndarray
            Field from which to draw discrete samples. This is likely the
            output of a previous call to `delta_x()`, but could in principle be
            any field. Note that if not supplied, the field will be generated
            from scratch. As a result, unless the user has supplied a random seed
            at initialization, the discrete samples will be a new realization of
            a field with the specified power spectrum.

        Returns
        -------
        tracer_positions : float, array_like
            ``(n, d)``-array, with ``n`` the number of tracers and ``d`` the number of
            dimensions. Each row represents a single tracer's co-ordinates.
        """
        if delta_x is None:
            if self.seed is None:
                warnings.warn(
                    "You Should provide `seed` at initialization if one"
                    " wants a correspondence between parent field and"
                    " discrete samples.",
                    stacklevel=2,
                )
            dx = self.delta_x()
        else:
            dx = delta_x

        dx = (dx + 1) * np.prod(self.dx) * nbar
        n = dx

        n_per_cell = self.rng.poisson(n)

        # Get all source positions
        args = self.x
        X = np.meshgrid(*args, indexing="ij")

        tracer_positions = np.array([x.flatten() for x in X]).T
        tracer_positions = tracer_positions.repeat(n_per_cell.flatten(), axis=0)

        if randomise_in_cell:
            tracer_positions += self.rng.uniform(size=(np.sum(n_per_cell), self.dim)) * np.asarray(
                self.dx
            )

        if min_at_zero:
            tracer_positions += np.asarray(self.size) / 2.0

        return tracer_positions


class LogNormalPowerBox(PowerBox):
    r"""Calculate Log-Normal density fields with given power spectra.

    See the documentation of :class:`PowerBox` for a detailed explanation of the
    arguments, as this class has exactly the same arguments, including scalar or
    per-axis ``N`` and ``boxlength`` inputs.

    This class calculates an (over-)density field of arbitrary dimension given an input
    isotropic power spectrum. In this case, the field has a log-normal distribution of
    over-densities, always yielding a physically valid field.

    Examples
    --------
    To create a log-normal over-density field:

    >>> from powerbox import LogNormalPowerBox
    >>> lnpb = LogNormalPowerBox(100,lambda k : k**-7./5.,dim=2, boxlength=1.0)
    >>> overdensities = lnpb.delta_x
    >>> grid = lnpb.x
    >>> radii = lnpb.r

    To plot the overdensities:

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(pb.delta_x)

    Compare the fields from a Gaussian and Lognormal realisation with the same power:

    >>> lnpb = LogNormalPowerBox(300,lambda k : k**-7./5.,dim=2, boxlength=1.0)
    >>> pb = PowerBox(300,lambda k : k**-7./5.,dim=2, boxlength=1.0)
    >>> fig,ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(12,5))
    >>> ax[0].imshow(lnpb.delta_x,aspect="equal",vmin=-1,vmax=lnpb.delta_x.max())
    >>> ax[1].imshow(pb.delta_x,aspect="equal",vmin=-1,vmax = lnpb.delta_x.max())

    To create and plot a discrete version of the field:

    >>> positions = lnpb.create_discrete_sample(
    >>>     nbar=1000.0, # Number density in terms of boxlength units
    >>>     randomise_in_cell=True
    >>> )
    >>> plt.scatter(positions[:,0],positions[:,1],s=2,alpha=0.5,lw=0)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def correlation_array(self) -> np.ndarray:
        """Return the correlation function from the input power on the grid."""
        pa = self.power_array()
        return self._irfft_to_field(pa, scale=self.volume)

    def gaussian_correlation_array(self) -> np.ndarray:
        """Correlation required for a Gaussian field to produce the input power."""
        return np.log(1 + self.correlation_array())

    def gaussian_power_array(self) -> np.ndarray:
        """Power spectrum required for a Gaussian field to produce the input power."""
        gca = self.fftbackend.empty(self.shape)
        gca[...] = self.gaussian_correlation_array()
        gpa = np.abs(
            self._full_spectrum_to_rfft(
                dft.fft(
                    gca,
                    L=self.size,
                    a=self.fourier_a,
                    b=self.fourier_b,
                    backend=self.fftbackend,
                )[0]
            )
        )
        gpa[self.k() == 0] = 0
        return gpa

    def delta_k(self) -> np.ndarray:
        """
        Return a realization of ``delta_k``.

        i.e. the gaussianised square root of the unitless power spectrum
        (i.e. the Fourier co-efficients)
        """
        p = self.gaussian_power_array()
        gh = self.gauss_hermitian()
        gh[...] = np.sqrt(p) * gh
        return gh

    def delta_x(self) -> np.ndarray:
        """Return the real-space over-density field from the input power spectrum."""
        dk = self.delta_k()
        dk = self._irfft_to_field(dk, scale=np.sqrt(self.volume))

        sg = np.var(dk)
        return np.exp(dk - sg / 2) - 1
