"""Conservation-based decomposition of subgrid exchange into wave modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WaveModeExchange:
    wave_power: float
    wave_torque: float
    rotating_pattern_power: float
    radial_residual_power: float
    effective_pattern_speed_myr_inverse: float | None
    power_over_orbital_frequency_times_torque: float | None
    nonnegative_radial_residual: bool


def decompose_wave_mode_exchange(
    *,
    orbital_power: float,
    orbital_torque: float,
    orbital_frequency_myr_inverse: float,
    zero_tolerance: float = 1.0e-30,
) -> WaveModeExchange:
    """Split wave power into a rotating-pattern term and an ``m=0`` residual.

    A pattern rotating at the orbital frequency receives power equal to its
    angular frequency times its torque.  The remaining power has zero angular
    momentum in this minimal decomposition.  A negative residual indicates
    that one orbital-frequency pattern plus a positive-energy radial mode is
    insufficient.
    """

    values = np.asarray(
        [
            orbital_power,
            orbital_torque,
            orbital_frequency_myr_inverse,
            zero_tolerance,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(values)):
        raise ValueError("mode-exchange inputs must be finite")
    if orbital_frequency_myr_inverse <= 0.0 or zero_tolerance < 0.0:
        raise ValueError("frequency must be positive and tolerance non-negative")

    wave_power = -orbital_power
    wave_torque = -orbital_torque
    rotating_power = orbital_frequency_myr_inverse * wave_torque
    radial_power = wave_power - rotating_power
    if abs(wave_torque) <= zero_tolerance:
        effective_pattern_speed = None
        ratio = None
    else:
        effective_pattern_speed = wave_power / wave_torque
        ratio = effective_pattern_speed / orbital_frequency_myr_inverse
    return WaveModeExchange(
        wave_power=float(wave_power),
        wave_torque=float(wave_torque),
        rotating_pattern_power=float(rotating_power),
        radial_residual_power=float(radial_power),
        effective_pattern_speed_myr_inverse=(
            None
            if effective_pattern_speed is None
            else float(effective_pattern_speed)
        ),
        power_over_orbital_frequency_times_torque=(
            None if ratio is None else float(ratio)
        ),
        nonnegative_radial_residual=bool(radial_power >= -zero_tolerance),
    )
