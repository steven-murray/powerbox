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

# How negative the most negative mode of a lognormal box's required Gaussian power
# spectrum may be, as a fraction of its largest mode, before the construction is rejected
# outright rather than merely reported.
#
# Some violation is normal and harmless: a realistic cosmological power spectrum on a
# 256^3 grid reaches about -4e-5 here, and zeroing those modes changes the realized field
# negligibly. A field whose variance genuinely exceeds what the log-transform supports
# looks quite different: about -0.02 for a marginal case, -0.7 for a hopeless one. This
# threshold sits in the gap, with roughly an order of magnitude of margin on each side.
#
# The measure is deliberately the depth of the worst mode relative to the spectrum's own
# scale, rather than the share of the total power held by negative modes: the latter is
# diluted by the number of grid cells, so it would call the same physical configuration
# acceptable on a fine grid and unacceptable on a coarse one.
_GAUSSIAN_POWER_MAX_DEPTH = 1e-3


_UNREALIZABLE_ADVICE = (
    "Consider reducing the amplitude of `pk`, increasing `size`, coarsening `shape`, or "
    "using a `pk` that falls off more steeply at high k."
)


def _require_sequence(value, name: str, example: str) -> None:
    """Reject a bare number where one entry per axis is required."""
    if np.ndim(value) == 0:
        raise TypeError(
            f"`{name}` must be a sequence with one entry per axis, e.g. `{name}={example}`, "
            f"but got the single number {value!r}."
        )


def _as_int_tuple(value) -> tuple[int, ...]:
    """Convert the ``shape`` argument to a tuple of ints."""
    _require_sequence(value, "shape", "(128, 128)")
    return tuple(int(i) for i in value)


def _as_float_tuple(value) -> tuple[float, ...]:
    """Convert the ``size`` argument to a tuple of floats."""
    _require_sequence(value, "size", "(100.0, 100.0)")
    return tuple(float(i) for i in value)


@attrs.define(kw_only=True, slots=False, frozen=True)
class PowerBox:
    r"""
    Generate real- and fourier-space Gaussian fields with a given power spectrum.

    Parameters
    ----------
    shape : sequence of int
        Number of grid-points along each axis of the resulting box (equivalently, number
        of wavenumbers to use), one entry per axis. The number of dimensions of the box
        is ``len(shape)``. This is required unless the deprecated ``N`` is given.
    pk : callable
        A callable of a single (vector) variable `k`, which is the isotropic power
        spectrum. The relationship of the `k` of which this is a function to the
        real-space co-ordinates, `x`, is determined by the parameters ``a,b``.
    size : sequence of float, optional
        Length of the box along each axis, one entry per axis. This may have arbitrary
        units, so long as `pk` is a function of a variable which has the inverse units.
        Defaults to a unit box.
    dim : int, optional
        Number of dimensions of the box. This is inferred from ``shape``, so it is only
        useful as a consistency check, or (with the deprecated scalar ``N``) to say how
        many axes ``N`` applies to.
    N : int, optional
        Deprecated in favour of ``shape``; will be removed in v1.2. Number of grid-points
        along every axis. Uses ``dim`` axes, or two if ``dim`` is not given.
    boxlength : float, optional
        Deprecated in favour of ``size``; will be removed in v1.2. Length of the box
        along every axis.
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
        A random seed to define the initial conditions. If not set, the box is seeded
        from system entropy and is not reproducible. If set, a new box with the same seed
        reproduces the same *sequence* of realisations, but successive calls to eg.
        :meth:`delta_x()` on one box still produce *different* realisations.
    nthreads : int, optional
        Number of threads for pyFFTW. If set to None, uses pyFFTW with the number of
        threads equal to the number of available CPUs. If set to 0 or 1 (or ``False`` or
        ``True``), uses numpy's FFT routine instead. If set to an integer greater than 1,
        uses pyFFTW with that many threads.

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
    is enforced in the :meth:`create_discrete_sample` method.

    The per-axis quantities ``shape``, ``size``, ``dx``, ``x`` and ``kvec`` are all
    tuples with one entry per axis. The spectrum-space quantities (:attr:`kvec`,
    :meth:`k`, :meth:`power_array`, :meth:`gauss_hermitian` and :meth:`delta_k`) use the
    half-spectrum layout of a real FFT: the last axis holds only the non-negative
    frequencies, ``shape[-1] // 2 + 1`` of them.

    .. note:: None of the n-dimensional arrays that are created within the class are
              stored, due to the inefficiency in memory consumption that this would
              imply. Thus, each large array is created and *returned* by their
              respective method, to be stored/discarded by the user.

    .. warning:: Due to the above note, repeated calls to eg. :meth:`delta_x()` will
                 produce *different* realisations of the real-space field, even if the
                 `seed` parameter is set in the constructor (which makes a *new* box
                 reproducible, not repeated calls on one box). Keep the returned array if
                 you need the same field twice.

    Examples
    --------
    To create a 3-dimensional box of gaussian over-densities, gridded into 100 bins,
    with cosmological conventions, and a power-law power spectrum, simply use

    >>> pb = PowerBox(shape=(100,) * 3, pk=lambda k: 0.1 * k**-3.0, size=(100.0,) * 3)
    >>> overdensities = pb.delta_x()
    >>> grid = pb.x
    >>> radii = pb.r

    To create a 2D turbulence structure, with arbitrary units, once can use

    >>> import matplotlib.pyplot as plt
    >>> pb = PowerBox(shape=(1000, 1000), pk=lambda k: k ** (-7.0 / 5.0))
    >>> plt.imshow(pb.delta_x())

    To create a 2D non-cubic box with different resolutions and side lengths:

    >>> pb = PowerBox(shape=(128, 192), pk=lambda k: (1 + k) ** -2.0, size=(200.0, 600.0))
    >>> field = pb.delta_x()
    >>> x, y = pb.x
    """

    N: int | None = attrs.field(
        default=None, converter=attrs.converters.optional(int), validator=vld.optional(vld.gt(0))
    )
    _dim: int | None = attrs.field(
        converter=attrs.converters.optional(int), default=None, validator=vld.optional(vld.gt(0))
    )
    shape: tuple[int, ...] = attrs.field(converter=_as_int_tuple)

    _pk: Callable[[np.ndarray], np.ndarray] = attrs.field(repr=False)

    _boxlength: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float), validator=vld.optional(vld.gt(0))
    )
    size: tuple[float, ...] = attrs.field(converter=_as_float_tuple)

    ensure_physical: bool = attrs.field(default=False, converter=bool)
    fourier_a: float = attrs.field(default=1.0, converter=float, alias="a")
    fourier_b: float = attrs.field(default=1.0, converter=float, alias="b")
    vol_normalised_power: bool = attrs.field(default=True, converter=bool)
    nthreads: int | None = attrs.field(
        default=None, converter=attrs.converters.optional(int), validator=vld.optional(vld.ge(0))
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
    def boxlength(self) -> tuple[float, ...]:
        """The box length along each axis.

        Deprecated alias for `size`. Will be removed in v1.2. Use `size` instead. Unlike the
        pre-v1 attribute, this is a tuple with one entry per axis.
        """
        warnings.warn(
            "The `boxlength` attribute is deprecated and will be removed in v1.2. Use `size` "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.size

    @property
    def L(self) -> tuple[float, ...]:
        """The box length along each axis.

        Deprecated alias for `size`. Will be removed in v1.2. Use `size` instead.
        """
        warnings.warn(
            "The `L` attribute is deprecated and will be removed in v1.2. Use `size` instead.",
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

    @property
    def _xp(self):
        """The array namespace (``numpy`` or ``jax.numpy``) of the active FFT backend."""
        return getattr(self.fftbackend, "xp", np)

    @property
    def synthesis_norm(self) -> float:
        r"""The amplitude with which a single Fourier mode enters the real-space field.

        The field synthesis performed by :meth:`delta_x` is equivalent to

        .. math:: \delta_j = A \sum_m \sqrt{P_m}\, g_m e^{i\mathbf{k}_m\cdot\mathbf{x}_j},

        where :math:`P_m` is :meth:`power_array`, :math:`g_m` are unit-variance Hermitian
        Gaussian modes, and :math:`A` is this quantity. It arises from the normalisation
        applied by ``_irfft_to_field(..., scale=volume)``, which contributes
        :math:`V_{\rm box} V_k / N_{\rm tot}` together with the inverse-transform
        convention pre-factor :math:`(b/(2\pi)^{1+a})^{n/2}`, and evaluates to

        .. math:: A = \left(\frac{(2\pi)^{1-a}}{b}\right)^{n/2}.

        It is exactly one for both the cosmological convention ``(a, b) = (1, 1)`` and
        the numpy convention ``(a, b) = (0, 2\pi)``, but not in general.

        Any quantity that is *quadratic* in the field -- most importantly the correlation
        function, :math:`\xi(r) = A^2 \sum_m P_m e^{i\mathbf{k}\cdot\mathbf{r}}` -- picks
        up a second power of this factor. Linear pipelines never see it, because it
        cancels between generation and measurement; :class:`LogNormalPowerBox` does see
        it, because :math:`\log(1 + \xi)` depends on the absolute scale of :math:`\xi`.
        """
        return float(((2 * np.pi) ** (1 - self.fourier_a) / self.fourier_b) ** (self.dim / 2))

    @property
    def _hermitian_multiplicity(self):
        r"""Multiplicity of each mode of the reduced spectrum in the full spectrum.

        Every mode of the reduced ``rfftn`` spectrum stands for two modes of the full
        spectrum -- itself and its conjugate partner -- except on the self-conjugate
        surfaces of the final axis, which are counted once.
        """
        index = self._xp.arange(self._rfft_shape[-1])
        self_conjugate = index == 0
        if self.shape[-1] % 2 == 0:
            self_conjugate = self_conjugate | (index == self._rfft_shape[-1] - 1)
        return self._xp.where(self_conjugate, 1.0, 2.0)

    @cached_property
    def variance(self) -> float:
        r"""The variance of the field, i.e. the zero-lag correlation :math:`\xi(0)`.

        This is the value implied by the input power spectrum on this grid, not the sample
        variance of any realization. Because :math:`\xi(0) = A^2 \sum_m P_m` over the
        *full* spectrum (with :math:`A` being :attr:`synthesis_norm`), it is obtained by
        summing the reduced spectrum with Hermitian multiplicities, and so costs no
        Fourier transform.
        """
        summed = self._xp.sum(self.power_array() * self._hermitian_multiplicity)
        return float(self.synthesis_norm**2 * summed)

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

    def _mode_amplitudes(self):
        """Return the per-mode amplitude that multiplies the Hermitian Gaussian modes.

        Subclasses that realize a non-Gaussian field override this to substitute the
        power spectrum of the underlying *latent* Gaussian field.

        Validation happens here, on the array that has just been computed, so that no
        quantity is computed twice. It is skipped when the values are abstract (inside a
        traced, compiled function), where a data-dependent branch is impossible; a backend
        that traces is responsible for touching :attr:`_validated_spectrum` first.
        """
        power = self.power_array()
        if not self.fftbackend.is_traced(power) and bool(self._xp.any(power < 0)):
            raise ValueError("The power spectrum function has returned negative values.")
        return self._xp.sqrt(power)

    @cached_property
    def _validated_spectrum(self) -> None:
        """Run the spectrum checks on concrete values, at most once per instance.

        The checks live inside :meth:`_mode_amplitudes` so that they see the array that has
        just been computed. That works directly for an eagerly-evaluated backend, but not
        inside a compiled kernel, where the values are abstract. Touching this beforehand
        performs them once, outside any trace, and caches the result.

        It depends only on the constructor arguments, never on a realization, so it could
        in principle be an ``attrs`` validator -- but that would force a lognormal box to
        perform two Fourier transforms at construction whether or not a field is ever
        generated, so it stays lazy.
        """
        self._mode_amplitudes()

    def _transform_field(self, field):
        """Map the realized latent Gaussian field to the output field.

        The identity (up to ``ensure_physical`` clipping) for a Gaussian box. Subclasses
        override this to apply their one-point transformation.
        """
        if self.ensure_physical:
            return self._xp.clip(field, -1, None)
        return field

    def delta_k(self):
        """Return a realization of ``delta_k``.

        The gaussianised square root of the power spectrum (i.e. the Fourier
        co-efficients).
        """
        return self._mode_amplitudes() * self.gauss_hermitian()

    def delta_x(self, delta_k: np.ndarray | None = None):
        """Return the realized real-space field from the input power spectrum."""
        # Here we multiply by V because the inverse Fourier transform of the
        # dimensionless power has units of 1/V, and we require a unitless
        # quantity for delta_x.
        dk = self.delta_k() if delta_k is None else delta_k
        return self._transform_field(self._irfft_to_field(dk, scale=self.volume))

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
    arguments, as this class has exactly the same arguments, including per-axis
    ``shape`` and ``size`` inputs.

    This class calculates an (over-)density field of arbitrary dimension given an input
    isotropic power spectrum. In this case, the field has a log-normal distribution of
    over-densities, always yielding a physically valid field.

    Examples
    --------
    To create a log-normal over-density field:

    >>> from powerbox import LogNormalPowerBox
    >>> lnpb = LogNormalPowerBox(shape=(100, 100), pk=lambda k: 0.01 * k**-2.0, size=(1.0, 1.0))
    >>> overdensities = lnpb.delta_x()
    >>> grid = lnpb.x
    >>> radii = lnpb.r

    To plot the overdensities:

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(overdensities)

    Compare the fields from a Gaussian and Lognormal realisation with the same power:

    >>> kwargs = dict(shape=(300, 300), pk=lambda k: 0.01 * k**-2.0, size=(1.0, 1.0), seed=1)
    >>> lnpb = LogNormalPowerBox(**kwargs)
    >>> pb = PowerBox(**kwargs)
    >>> ln_field, gauss_field = lnpb.delta_x(), pb.delta_x()
    >>> fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12, 5))
    >>> ax[0].imshow(ln_field, aspect="equal", vmin=-1, vmax=ln_field.max())
    >>> ax[1].imshow(gauss_field, aspect="equal", vmin=-1, vmax=ln_field.max())

    To create and plot a discrete version of the field:

    >>> positions = lnpb.create_discrete_sample(
    >>>     nbar=1000.0, # Number density in terms of size units
    >>>     randomise_in_cell=True
    >>> )
    >>> plt.scatter(positions[:,0],positions[:,1],s=2,alpha=0.5,lw=0)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def correlation_array(self):
        r"""Return the correlation function of the input power on the grid.

        This is the true dimensionless correlation function of the field that
        :class:`PowerBox` would generate from the same power spectrum, so that its
        zero-lag value, ``correlation_array()[(0,) * dim]``, is the variance of that
        field. Obtaining that requires two powers of :attr:`synthesis_norm` (one from the
        inverse transform itself, and one because :math:`\xi` is quadratic in the field),
        which is why the scale here is ``volume * synthesis_norm`` rather than ``volume``.
        """
        return self._irfft_to_field(self.power_array(), scale=self.volume * self.synthesis_norm)

    def gaussian_correlation_array(self):
        """Correlation required for a Gaussian field to produce the input power."""
        return self._xp.log1p(self.correlation_array())

    @cached_property
    def gaussian_variance(self) -> float:
        r"""The variance of the latent Gaussian field, :math:`\xi_G(0) = \ln(1 + \xi(0))`.

        This is the theoretical value implied by the input power spectrum, rather than the
        sample variance of any particular realization, so that :meth:`delta_x` realizes
        the lognormal model exactly as specified. It reuses :attr:`PowerBox.variance`, and
        so needs no Fourier transform of its own.
        """
        return float(np.log1p(self.variance))

    def gaussian_power_array(self):
        r"""Power spectrum required for a Gaussian field to produce the input power.

        Returned in the same normalisation as :meth:`PowerBox.power_array`, so that the
        inherited Gaussian synthesis machinery consumes it unchanged.

        Negative modes are clipped to zero so that the square root taken by :meth:`delta_k`
        is well-defined. They are not clipped *silently*: :meth:`_validate_gaussian_power`
        warns about them, and raises if the violation is large enough to matter.
        """
        gpa = self._unclipped_gaussian_power_array()
        if not self.fftbackend.is_traced(gpa):
            self._validate_gaussian_power(gpa)
        return self._xp.where(gpa < 0, 0.0, gpa)

    def _unclipped_gaussian_power_array(self):
        """Return the required Gaussian power spectrum without clipping negative modes."""
        # log(1 + xi) is NaN wherever xi < -1. That is reported, with an explanation, by
        # `_validate_gaussian_power`, so the bare floating-point warning is just noise.
        with np.errstate(invalid="ignore", divide="ignore"):
            gaussian_correlation = self.gaussian_correlation_array()
        full = dft.fft(
            gaussian_correlation,
            L=self.size,
            a=self.fourier_a,
            b=self.fourier_b,
            backend=self.fftbackend,
        )[0]
        gpa = self._full_spectrum_to_rfft(full.real) / (self.volume * self.synthesis_norm)
        return self._xp.where(self.k() == 0, 0.0, gpa)

    def _validate_gaussian_power(self, gaussian_power) -> None:
        r"""Check that the required Gaussian power spectrum is positive semi-definite.

        Although :math:`\xi` is positive semi-definite by construction, its log-transform
        :math:`\ln(1 + \xi)` need not be. This happens for fields of large variance, and
        means the requested power spectrum is not realizable as a lognormal field on this
        grid at all -- there is no Gaussian field whose exponential has that power. It is
        a genuine limitation of the Coles & Jones (1991) construction, not a numerical
        problem, so it is reported rather than silently worked around.
        """
        if not bool(self._xp.all(self._xp.isfinite(gaussian_power))):
            # log(1 + xi) is undefined wherever xi <= -1, which poisons every mode of the
            # transform, so there is no meaningful "most negative mode" to report.
            raise ValueError(
                "The correlation function implied by the requested power spectrum reaches "
                f"{float(self._xp.min(self.correlation_array())):.4g}, at or below -1, where "
                "log(1 + xi) is undefined. log(1 + xi) is therefore not a valid correlation "
                "function on this grid, so no lognormal field with this power spectrum exists "
                f"here. The field variance implied by the input power is xi(0) = "
                f"{self.variance:.4g}, and the construction generally fails once that "
                f"approaches or exceeds unity. {_UNREALIZABLE_ADVICE}"
            )

        largest = float(self._xp.max(gaussian_power))
        smallest = float(self._xp.min(gaussian_power))
        if smallest >= 0:
            return

        negative = gaussian_power < 0
        n_negative = int(self._xp.sum(negative))
        n_modes = int(gaussian_power.size)
        depth = smallest / largest
        negative_share = float(
            self._xp.sum(self._xp.abs(gaussian_power[negative]))
            / self._xp.sum(self._xp.abs(gaussian_power))
        )

        description = (
            f"{n_negative} of {n_modes} modes ({n_negative / n_modes:.3%}) of the Gaussian "
            "power spectrum required to produce the requested lognormal field are "
            f"negative; the most negative is {depth:.3g} times the largest mode, and "
            f"negative modes hold {negative_share:.3g} of the total power. The field "
            f"variance implied by the input power is xi(0) = {self.variance:.4g}."
        )

        if depth > -_GAUSSIAN_POWER_MAX_DEPTH:
            warnings.warn(
                f"{description} This is a small violation of the positive-definiteness "
                "that the lognormal construction requires, of the size expected from "
                "discretising a realistic power spectrum, and these modes are set to zero. "
                "It is reported because the generated field is then not exactly the "
                "requested one.",
                stacklevel=3,
            )
            return

        raise ValueError(
            f"{description} log(1 + xi) is therefore not a valid correlation function on "
            "this grid, so no lognormal field with this power spectrum exists here: the "
            "construction generally fails once xi(0) approaches or exceeds unity. "
            f"{_UNREALIZABLE_ADVICE}"
        )

    def _mode_amplitudes(self):
        """Return the mode amplitudes of the *latent Gaussian* field."""
        return self._xp.sqrt(self.gaussian_power_array())

    def _transform_field(self, field):
        r"""Exponentiate the latent Gaussian field into a lognormal over-density.

        The :math:`-\sigma_G^2/2` term makes the field zero-mean in expectation, and uses
        the theoretical latent variance rather than that of the realization, so the field
        realizes the specified model rather than a realization-dependent variant of it.
        """
        return self._xp.exp(field - self.gaussian_variance / 2) - 1
