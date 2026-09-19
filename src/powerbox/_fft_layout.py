"""Shared helpers for reduced-spectrum FFT layout conversions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def full_spectrum_to_rfft(
    spectrum: Any,
    *,
    dim: int,
    rfft_last_axis_size: int,
    backend: Any,
) -> Any:
    """Convert a centred full spectrum to reduced rFFT layout."""
    reduced = backend.ifftshift(spectrum, axes=(-1,))
    return reduced[(slice(None),) * (dim - 1) + (slice(None, rfft_last_axis_size),)]


def irfft_to_field(
    spectrum: Any,
    *,
    scale: float,
    irfft_function: Callable[..., tuple[Any, Sequence[Any]]],
    L: float | Sequence[float],
    a: float,
    b: float,
    N: Sequence[int],
    backend: Any | None = None,
) -> Any:
    """Transform reduced half-spectrum data to real-space field values."""
    kwargs: dict[str, Any] = {"L": L, "a": a, "b": b, "N": N}
    if backend is not None:
        kwargs["backend"] = backend
    return scale * irfft_function(spectrum, **kwargs)[0]
