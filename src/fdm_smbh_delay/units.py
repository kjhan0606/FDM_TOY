"""Strict parsing at the unit-aware public interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from astropy import units as u


class UnitValidationError(ValueError):
    """Raised when a dimensional configuration value has no usable unit."""


def parse_quantity(value: Any, target_unit: u.UnitBase, field: str) -> float:
    """Return a scalar in ``target_unit``, rejecting unitless public values."""

    if not isinstance(value, str):
        raise UnitValidationError(
            f"{field} must be a string with an explicit unit, got {value!r}"
        )
    try:
        quantity = u.Quantity(value)
    except (TypeError, ValueError) as exc:
        raise UnitValidationError(f"cannot parse {field}={value!r}") from exc
    if quantity.unit == u.dimensionless_unscaled:
        raise UnitValidationError(f"{field} must carry an explicit unit")
    if not quantity.isscalar:
        raise UnitValidationError(f"{field} must be scalar")
    try:
        return float(quantity.to_value(target_unit))
    except u.UnitConversionError as exc:
        raise UnitValidationError(
            f"{field}={value!r} is not convertible to {target_unit}"
        ) from exc


def parse_vector(
    values: Any, target_unit: u.UnitBase, field: str, *, length: int = 3
) -> np.ndarray:
    """Parse a fixed-length vector whose entries each have explicit units."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise UnitValidationError(f"{field} must be a {length}-element sequence")
    if len(values) != length:
        raise UnitValidationError(f"{field} must contain exactly {length} entries")
    return np.asarray(
        [parse_quantity(value, target_unit, f"{field}[{i}]") for i, value in enumerate(values)],
        dtype=float,
    )


def require_finite_positive(value: float, field: str, *, allow_zero: bool = False) -> float:
    """Validate a parsed scalar and return it unchanged."""

    valid_sign = value >= 0.0 if allow_zero else value > 0.0
    if not np.isfinite(value) or not valid_sign:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be finite and {qualifier}, got {value!r}")
    return value
