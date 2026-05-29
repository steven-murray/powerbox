"""JAX-backed ``powerbox`` interfaces.

This subpackage mirrors the supported ``powerbox`` API surface using JAX arrays and
FFT primitives. Milestone-1 support intentionally excludes interpolation-based power
estimation and discrete tracer sampling.
"""

from __future__ import annotations

try:
    import jax  # noqa: F401
except ImportError as err:  # pragma: no cover - exercised when jax is unavailable
    raise ImportError(
        "powerbox.jax requires the optional 'jax' dependency. Install it with "
        "`pip install powerbox[jax]` or `uv sync --extra jax`."
    ) from err

from ..dft_backend import JaxFFT
from .dft import fft, fftfreq, fftshift, get_fft_backend, ifft, ifftshift
from .powerbox import LogNormalPowerBox, PowerBox
from .tools import (
    PowerSpectrum,
    angular_average,
    angular_average_nd,
    get_power,
    ignore_zero_absk,
    ignore_zero_ki,
    power2delta,
)

__all__ = [
    "JaxFFT",
    "LogNormalPowerBox",
    "PowerBox",
    "PowerSpectrum",
    "angular_average",
    "angular_average_nd",
    "fft",
    "fftfreq",
    "fftshift",
    "get_fft_backend",
    "get_power",
    "ifft",
    "ifftshift",
    "ignore_zero_absk",
    "ignore_zero_ki",
    "power2delta",
]
