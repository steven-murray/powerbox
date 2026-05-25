"""Classes that can create arbitrary-dimensional fields with given power spectra.

One such function produces *Gaussian* fields, and the other *LogNormal* fields.

In principle, these may be extended to other 1-point density distributions by
subclassing :class:`PowerBox` and over-writing the same methods as are over-written in
:class:`LogNormalPowerBox`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from . import dft
from ._fft_layout import full_spectrum_to_rfft, irfft_to_field
from ._geometry import tuplify_type
from ._hermitianity import hermitianize_rfft_array
from .tools import _magnitude_grid


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

    def __init__(
        self,
        N: int | Sequence[int],
        pk,
        dim: int = 2,
        boxlength: float | Sequence[float] = 1.0,
        ensure_physical: bool = False,
        a: float = 1.0,
        b: float = 1.0,
        vol_normalised_power: bool = True,
        seed: int | None = None,
        nthreads: int | None = None,
    ) -> None:
        self.dim = dim
        self.N = tuplify_type(int, N, dim, "N")
        self.boxlength = tuplify_type(float, boxlength, dim, "boxlength")
        self.L = self.boxlength
        self.fourier_a = a
        self.fourier_b = b
        self.vol_normalised_power = vol_normalised_power
        self.V = float(np.prod(self.boxlength))
        self.fftbackend = dft.get_fft_backend(nthreads)

        if self.vol_normalised_power:
            self.pk = lambda k: pk(k) / self.V
        else:
            self.pk = pk

        self.ensure_physical = ensure_physical
        self.Ntot = int(np.prod(self.N))

        self.seed = seed
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            # Keep seeded realizations close to historical behavior while using
            # the Generator API required by modern NumPy.
            self.rng = np.random.Generator(np.random.MT19937(seed))

        self.dx = tuple(
            length / axis_n for length, axis_n in zip(self.boxlength, self.N, strict=True)
        )

    @property
    def x(self) -> tuple[np.ndarray, ...]:
        """The co-ordinates of the grid along each axis."""
        return tuple(
            np.arange(-length / 2, length / 2, axis_dx)[:axis_n]
            for length, axis_dx, axis_n in zip(self.boxlength, self.dx, self.N, strict=True)
        )

    @property
    def kvec(self) -> tuple[np.ndarray, ...]:
        """The reduced wavenumber vectors for the half-Hermitian spectrum."""
        axes = [
            self.fftbackend.fftfreq(axis_n, d=axis_dx, b=self.fourier_b)
            for axis_n, axis_dx in zip(self.N[:-1], self.dx[:-1], strict=True)
        ]
        axes.append(self.fftbackend.rfftfreq(self.N[-1], d=self.dx[-1], b=self.fourier_b))
        return tuple(axes)

    @property
    def _rfft_shape(self) -> tuple[int, ...]:
        """Shape of the half-Hermitian spectrum compatible with ``irfftn``."""
        return (*self.N[:-1], self.N[-1] // 2 + 1)

    def _irfft_to_field(self, spectrum, scale: float):
        """Transform a reduced half-spectrum into a real-space field."""
        return irfft_to_field(
            spectrum,
            scale=scale,
            irfft_function=dft.irfft,
            L=self.boxlength,
            a=self.fourier_a,
            b=self.fourier_b,
            N=self.N,
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

        hermitianize_rfft_array(modes, has_nyquist=self.N[-1] % 2 == 0)

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
        dk = self._irfft_to_field(dk, scale=self.V)

        if self.ensure_physical:
            np.clip(dk, -1, np.inf, dk)

        return dk

    def create_discrete_sample(
        self,
        nbar: float,
        randomise_in_cell: bool = True,
        min_at_zero: bool = False,
        store_pos: bool = False,
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
        store_pos : bool, optional
            Whether to store the sample of tracers as an instance variable
            ``tracer_positions``.
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

        self.n_per_cell = self.rng.poisson(n)

        # Get all source positions
        args = self.x
        X = np.meshgrid(*args, indexing="ij")

        tracer_positions = np.array([x.flatten() for x in X]).T
        tracer_positions = tracer_positions.repeat(self.n_per_cell.flatten(), axis=0)

        if randomise_in_cell:
            tracer_positions += self.rng.uniform(
                size=(np.sum(self.n_per_cell), self.dim)
            ) * np.asarray(self.dx)

        if min_at_zero:
            tracer_positions += np.asarray(self.boxlength) / 2.0

        if store_pos:
            self.tracer_positions = tracer_positions

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

    def correlation_array(self):
        """Return the correlation function from the input power on the grid."""
        pa = self.power_array()
        return self._irfft_to_field(pa, scale=self.V)

    def gaussian_correlation_array(self):
        """Correlation required for a Gaussian field to produce the input power."""
        return np.log(1 + self.correlation_array())

    def gaussian_power_array(self):
        """Power spectrum required for a Gaussian field to produce the input power."""
        gca = self.fftbackend.empty(self.N)
        gca[...] = self.gaussian_correlation_array()
        gpa = np.abs(
            self._full_spectrum_to_rfft(
                dft.fft(
                    gca,
                    L=self.boxlength,
                    a=self.fourier_a,
                    b=self.fourier_b,
                    backend=self.fftbackend,
                )[0]
            )
        )
        gpa[self.k() == 0] = 0
        return gpa

    def delta_k(self):
        """
        Return a realization of ``delta_k``.

        i.e. the gaussianised square root of the unitless power spectrum
        (i.e. the Fourier co-efficients)
        """
        p = self.gaussian_power_array()
        gh = self.gauss_hermitian()
        gh[...] = np.sqrt(p) * gh
        return gh

    def delta_x(self):
        """Return the real-space over-density field from the input power spectrum."""
        dk = self.delta_k()
        dk = self._irfft_to_field(dk, scale=np.sqrt(self.V))

        sg = np.var(dk)
        return np.exp(dk - sg / 2) - 1
