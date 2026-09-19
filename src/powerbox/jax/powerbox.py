"""JAX-backed random field generators."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import cached_property

import attrs
import jax
import jax.numpy as jnp

from .._fft_layout import full_spectrum_to_rfft, irfft_to_field
from ..dft_backend import JaxFFT
from ..powerbox import LogNormalPowerBox as _NumpyLogNormalPowerBox
from ..powerbox import PowerBox as _NumpyPowerBox
from . import dft
from .tools import _magnitude_grid

DEFAULT_JIT_NTOT_THRESHOLD = 65536


def _sample_gaussian_hermitian_fft(axis_lengths: tuple[int, ...], key: jax.Array) -> jax.Array:
    """Sample a full Hermitian Gaussian array via an FFT of real white noise."""
    if not axis_lengths:
        return jax.random.normal(key, ())

    noise = jax.random.normal(key, shape=axis_lengths)
    return jnp.fft.fftshift(jnp.fft.fftn(noise)) / jnp.sqrt(jnp.prod(jnp.array(axis_lengths)))


@attrs.define(frozen=True, slots=False, kw_only=True)
class PowerBox(_NumpyPowerBox):
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
    explicit through JAX PRNG keys. Instances are frozen after construction so cached
    JIT kernels remain valid. ``delta_x()`` uses the configured JIT policy, while a
    private eager path remains available for benchmarking and comparison.
    """

    _pk: Callable[[jax.Array], jax.Array] = attrs.field(repr=False)
    key: jax.Array | None = attrs.field(default=None)
    _usejit: bool | None = attrs.field(default=None)

    fftbackend: JaxFFT = attrs.field(init=False, repr=False, default=JaxFFT())
    _delta_x_keys: list[jax.Array] = attrs.field(init=False, repr=False, factory=list)

    @cached_property
    def usejit(self) -> bool:
        """Whether to use JIT compilation for ``delta_x()``."""
        if self._usejit is None:
            return self.total_ncells >= DEFAULT_JIT_NTOT_THRESHOLD
        return self._usejit

    def _resolve_key(self, key: jax.Array | None) -> jax.Array:
        """Return the key used for the current realization."""
        if key is not None:
            return key
        if self.key is not None:
            return self.key
        raise ValueError(
            "A JAX PRNG key is required. Pass `key=` to the constructor or to the method call."
        )

    def pk(self, k: jax.Array) -> jax.Array:
        """Return the input power spectrum evaluated at k."""
        if self.vol_normalised_power:
            return self._pk(k) / self.volume
        return self._pk(k)

    @property
    def x(self) -> tuple[jax.Array, ...]:
        """The co-ordinates of the grid along each axis."""
        return tuple(
            jnp.arange(axis_n, dtype=float) * axis_dx - length / 2
            for length, axis_dx, axis_n in zip(self.size, self.dx, self.shape, strict=True)
        )

    @property
    def kvec(self) -> tuple[jax.Array, ...]:
        """The reduced wavenumber vectors for the half-Hermitian spectrum."""
        axes = [
            self.fftbackend.fftfreq(axis_n, d=axis_dx, b=self.fourier_b)
            for axis_n, axis_dx in zip(self.shape[:-1], self.dx[:-1], strict=True)
        ]
        axes.append(self.fftbackend.rfftfreq(self.shape[-1], d=self.dx[-1], b=self.fourier_b))
        return tuple(axes)

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
        if self.shape[-1] % 2 == 0:
            surface_indices.append(self.shape[-1] // 2)

        keys = jax.random.split(key, 2 + len(surface_indices))
        modes = (
            jax.random.normal(keys[0], shape=self._rfft_shape)
            + 1j * jax.random.normal(keys[1], shape=self._rfft_shape)
        ) / jnp.sqrt(2)

        for surface_index, surface_key in zip(surface_indices, keys[2:], strict=True):
            modes = modes.at[..., surface_index].set(
                _sample_gaussian_hermitian_fft(self.shape[:-1], surface_key)
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
            L=self.size,
            a=self.fourier_a,
            b=self.fourier_b,
            N=self.shape,
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
        return self._mode_amplitudes() * self.gauss_hermitian(key=key)

    def _delta_x_eager(self, key: jax.Array | None = None) -> jax.Array:
        """Return the realized real-space field without JIT compilation.

        The mode amplitudes and the one-point transformation come from the shared
        implementation in :mod:`powerbox.powerbox`; only the PRNG handling is specific to
        this backend.
        """
        dk = self._mode_amplitudes() * self._gaussian_modes_rfft(key=key)
        return self._transform_field(self._irfft_to_field(dk, scale=self.volume))

    @cached_property
    def _delta_x_kernel(self) -> Callable[[jax.Array], jax.Array]:
        """Return a cached JIT-compiled kernel for :meth:`delta_x`."""

        @jax.jit
        def _kernel(run_key: jax.Array) -> jax.Array:
            return self._delta_x_eager(key=run_key)

        return _kernel

    def delta_x(self, key: jax.Array | None = None) -> jax.Array:
        """Return the realized real-space field using the configured execution policy."""
        if not self.usejit and self._usejit is None and len(self._delta_x_keys) == 1:
            # Note that the default chose eager *because* the box is small, so JIT is not
            # a foregone win here; the point is to surface the choice, not to recommend
            # overriding it blindly.
            warnings.warn(
                "delta_x() is using eager execution, chosen by the size heuristic for a box "
                f"of {self.total_ncells} cells. Compilation may still pay off over repeated "
                "calls at this size: pass usejit=True to find out, or usejit=False to keep "
                "eager execution and silence this warning.",
                stacklevel=2,
            )

        # The spectrum checks need a data-dependent branch, which is impossible on traced
        # values, so they run here -- outside any trace -- and are skipped when the
        # compiled kernel re-evaluates the same expression.
        _ = self._validated_spectrum

        run_key = self._resolve_key(key)
        self._delta_x_keys.append(run_key)
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
            "powerbox.jax.PowerBox.create_discrete_sample is not implemented yet."
        )


# The lognormal construction is written against the FFT backend's array namespace, so it
# operates on jax.Array without re-implementation; only the PRNG handling and JIT policy,
# inherited from the JAX PowerBox above, are specific to this backend.
class LogNormalPowerBox(_NumpyLogNormalPowerBox, PowerBox):
    """Generate JAX-backed lognormal density fields with a given power spectrum."""
