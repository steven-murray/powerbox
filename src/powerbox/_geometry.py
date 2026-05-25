"""Shared geometry normalization utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _is_scalar_like(value: object) -> bool:
    """Return whether *value* should be treated as a scalar input."""
    return np.isscalar(value) or np.ndim(value) == 0


def tuplify_type(
    tp: type[int | float],
    obj: int | float | Sequence[int | float],
    dim: int,
    name: str,
) -> tuple[int | float, ...]:
    """Normalize scalar or per-axis inputs to tuples with basic type validation."""
    if tp is int:

        def convert(value: int | float) -> int:
            if not float(value).is_integer():
                raise TypeError(f"{name} entries must be integers.")
            return int(value)

    else:

        def convert(value: int | float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} entries must be real numbers.") from exc

    if _is_scalar_like(obj):
        return (convert(obj),) * dim

    if len(obj) != dim:
        raise ValueError(f"{name} must be a scalar or have length {dim}.")

    return tuple(convert(value) for value in obj)
