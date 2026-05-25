"""JAX-backed random field generators."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

from .._fft_layout import full_spectrum_to_rfft, irfft_to_field
from .._geometry import tuplify_type
from ..dft_backend import JaxFFT
from . import dft
from .tools import _magnitude_grid

DEFAULT_JIT_NTOT_THRESHOLD = 65536


def _sample_gaussian_hermitian_fft(axis_lengths: tuple[int, ...], key: jax.Array) -> jax.Array:
    """Sample a full Hermitian Gaussian array via an FFT of real white noise."""
    if not axis_lengths:
        return jax.random.normal(key, ())

    noise = jax.random.normal(key, shape=axis_lengths)
    return jnp.fft.fftshift(jnp.fft.fftn(noise)) / jnp.sqrt(jnp.prod(jnp.array(axis_lengths)))


class PowerBox:
    r"""Generate JAX-backed Gaussian fields with a given isotropic power spectrum.

    Parameters
    ----------
    N : int or sequence of int
        Number of grid points on each axis.
    pk : callable
        Callable returning the isotropic input power spectrum as a function of ``k``.
    dim : int, default 2
        Number of spatial dimensions.
    boxlength : float or sequence of float, default 1.0
        Physical side length of the box along each axis.
    ensure_physical : bool, optional
        If ``True``, clip the returned real-space field to values greater than ``-1``.
    a, b : float, optional
        Fourier convention parameters. See :mod:`powerbox.dft`.
    vol_normalised_power : bool, optional
        Whether the input power spectrum is volume-weighted.
    nthreads : int, optional
        Accepted for API compatibility. The JAX backend ignores this value.
    key : jax.Array, optional
        PRNG key used to generate realizations. Methods such as :meth:`delta_x` accept
        a per-call ``key=`` override. If no key is supplied either here or at call time,
        a :class:`ValueError` is raised.
    usejit : bool, optional
        Whether to use the cached JIT-compiled ``delta_x`` path. If omitted, a simple
        heuristic selects JIT for large ``Ntot`` and eager execution for smaller boxes.

    Notes
    -----
    Geometry is normalized to per-axis tuples, matching the NumPy implementation's
    reduced-spectrum API. Unlike the NumPy implementation, random-state handling is
    explicit through JAX PRNG keys. ``delta_x()`` always uses the public JIT policy,
    while a private eager path remains available for benchmarking and comparison.
    """

    def __init__(
        self,
        N: int | Sequence[int],
        pk: Callable[[jax.Array], jax.Array],
        dim: int = 2,
        boxlength: float | Sequence[float] = 1.0,
        ensure_physical: bool = False,
        a: float = 1.0,
        b: float = 1.0,
        vol_normalised_power: bool = True,
        nthreads: int | None = None,
        key: jax.Array | None = None,
        usejit: bool | None = None,
    ) -> None:
        del nthreads

        self.dim = dim
        self.N = tuplify_type(int, N, dim, "N")
        self.boxlength = tuplify_type(float, boxlength, dim, "boxlength")
        self.L = self.boxlength
        self.fourier_a = a
        self.fourier_b = b
        self.vol_normalised_power = vol_normalised_power
        self.V = float(np.prod(self.boxlength))
        self.fftbackend = JaxFFT()
        self.key = key
        if self.vol_normalised_power:
            self.pk = lambda k: pk(k) / self.V
        else:
            self.pk = pk

        self.ensure_physical = ensure_physical
        self.Ntot = int(np.prod(self.N))
        self._usejit_requested = usejit is not None
        self.usejit = self._resolve_usejit(usejit)
        self._delta_x_calls = 0
        self._delta_x_warning_emitted = False
        self.dx = tuple(
            length / axis_n for length, axis_n in zip(self.boxlength, self.N, strict=True)
        )

    @property
    def x(self) -> tuple[jax.Array, ...]:
        """The co-ordinates of the grid along each axis."""
        return tuple(
            jnp.arange(axis_n, dtype=float) * axis_dx - length / 2
            for length, axis_dx, axis_n in zip(self.boxlength, self.dx, self.N, strict=True)
        )

    @property
    def kvec(self) -> tuple[jax.Array, ...]:
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

    def _resolve_key(self, key: jax.Array | None) -> jax.Array:
        """Return the key used for the current realization."""
        if key is not None:
            return key
        if self.key is not None:
            return self.key
        raise ValueError(
            "A JAX PRNG key is required. Pass `key=` to the constructor or to the method call."
        )

    def _resolve_usejit(self, usejit: bool | None) -> bool:
        """Resolve the cached JIT policy for this instance."""
        if usejit is not None:
            return bool(usejit)
        return self.Ntot >= DEFAULT_JIT_NTOT_THRESHOLD

    def _power_array_rfft(self) -> jax.Array:
        """Return the input power spectrum on the reduced half-spectrum grid."""
        k = _magnitude_grid(list(self.kvec))
        mask = k != 0
        safe_k = jnp.where(mask, k, 1)
        return jnp.where(mask, self.pk(safe_k), 0)

    def _gaussian_modes_rfft(self, key: jax.Array | None = None) -> jax.Array:
        """Return reduced Hermitian Gaussian modes sampled directly in rFFT layout."""
        key = self._resolve_key(key)
        surface_indices = [0]
        if self.N[-1] % 2 == 0:
            surface_indices.append(self.N[-1] // 2)

        keys = jax.random.split(key, 2 + len(surface_indices))
        modes = (
            jax.random.normal(keys[0], shape=self._rfft_shape)
            + 1j * jax.random.normal(keys[1], shape=self._rfft_shape)
        ) / jnp.sqrt(2)

        for surface_index, surface_key in zip(surface_indices, keys[2:], strict=True):
            modes = modes.at[..., surface_index].set(
                _sample_gaussian_hermitian_fft(self.N[:-1], surface_key)
            )

        return modes

    def _full_spectrum_to_rfft(self, spectrum: jax.Array) -> jax.Array:
        """Convert a centred full spectrum to the reduced ``irfftn`` layout."""
        return full_spectrum_to_rfft(
            spectrum,
            dim=self.dim,
            rfft_last_axis_size=self._rfft_shape[-1],
            backend=self.fftbackend,
        )

    def _irfft_to_field(self, spectrum: jax.Array, scale: float) -> jax.Array:
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

    def k(self) -> jax.Array:
        """Return the full grid of wavenumber magnitudes."""
        return _magnitude_grid(list(self.kvec))

    @property
    def r(self) -> jax.Array:
        """The radial position of every point in the grid."""
        return _magnitude_grid(list(self.x))

    def gauss_hermitian(self, key: jax.Array | None = None) -> jax.Array:
        """Return reduced Hermitian Gaussian modes in rFFT layout."""
        return self._gaussian_modes_rfft(key=key)

    def power_array(self) -> jax.Array:
        """Return the volume-normalized power spectrum evaluated on reduced ``self.k()``."""
        return self._power_array_rfft()

    def delta_k(self, key: jax.Array | None = None) -> jax.Array:
        """Return a realization of the Fourier-space field."""
        power = self.power_array()
        if bool(jnp.any(power < 0)):
            raise ValueError("The power spectrum function has returned negative values.")
        return jnp.sqrt(power) * self.gauss_hermitian(key=key)

    def _delta_x_eager(self, key: jax.Array | None = None) -> jax.Array:
        """Return the realized real-space field without JIT compilation."""
        dk = jnp.sqrt(self._power_array_rfft()) * self._gaussian_modes_rfft(key=key)
        field = self._irfft_to_field(dk, scale=self.V)
        if self.ensure_physical:
            field = jnp.clip(field, -1, jnp.inf)
        return field

    @cached_property
    def _delta_x_kernel(self) -> Callable[[jax.Array], jax.Array]:
        """Return a cached JIT-compiled kernel for :meth:`delta_x`."""

        @jax.jit
        def _kernel(run_key: jax.Array) -> jax.Array:
            return self._delta_x_eager(key=run_key)

        return _kernel

    def delta_x(self, key: jax.Array | None = None) -> jax.Array:
        """Return the realized real-space field using the configured execution policy."""
        self._delta_x_calls += 1
        if (
            not self.usejit
            and not self._usejit_requested
            and self._delta_x_calls > 1
            and not self._delta_x_warning_emitted
        ):
            warnings.warn(
                "delta_x() is using eager execution by default for this box size. "
                "Repeated calls may be much slower than usejit=True.",
                stacklevel=2,
            )
            self._delta_x_warning_emitted = True

        run_key = self._resolve_key(key)
        if self.usejit:
            return self._delta_x_kernel(run_key)
        return self._delta_x_eager(run_key)

    def create_discrete_sample(
        self,
        nbar: float,
        randomise_in_cell: bool = True,
        min_at_zero: bool = False,
        store_pos: bool = False,
        delta_x: jax.Array | None = None,
    ) -> jax.Array:
        """Discrete tracer sampling is not yet implemented for JAX."""
        del nbar, randomise_in_cell, min_at_zero, store_pos, delta_x
        raise NotImplementedError(
            "powerbox.jax.PowerBox.create_discrete_sample is not implemented in milestone 1."
        )


class LogNormalPowerBox(PowerBox):
    r"""Generate JAX-backed lognormal density fields with a given power spectrum."""

    def correlation_array(self) -> jax.Array:
        """Return the correlation function from the input power on the grid."""
        return self._irfft_to_field(self.power_array(), scale=self.V)

    def gaussian_correlation_array(self) -> jax.Array:
        """Return the Gaussian correlation producing the target lognormal power."""
        return jnp.log1p(self.correlation_array())

    def gaussian_power_array(self) -> jax.Array:
        """Return the Gaussian power spectrum producing the target lognormal field."""
        gaussian_power = jnp.abs(
            self._full_spectrum_to_rfft(
                dft.fft(
                    self.gaussian_correlation_array(),
                    L=self.boxlength,
                    a=self.fourier_a,
                    b=self.fourier_b,
                )[0]
            )
        )
        return jnp.where(self.k() == 0, 0, gaussian_power)

    def delta_k(self, key: jax.Array | None = None) -> jax.Array:
        """Return a realization of the Gaussianized Fourier-space field."""
        return jnp.sqrt(self.gaussian_power_array()) * self.gauss_hermitian(key=key)

    def _delta_x_eager(self, key: jax.Array | None = None) -> jax.Array:
        """Return the realized lognormal over-density field without JIT compilation."""
        dk = jnp.sqrt(self.gaussian_power_array())
        dk = dk * self._gaussian_modes_rfft(key=key)
        field = self._irfft_to_field(dk, scale=jnp.sqrt(self.V))
        sigma_g = jnp.var(field)
        return jnp.exp(field - sigma_g / 2) - 1

    def delta_x(self, key: jax.Array | None = None) -> jax.Array:
        """Return the realized lognormal over-density field."""
        return super().delta_x(key=key)
