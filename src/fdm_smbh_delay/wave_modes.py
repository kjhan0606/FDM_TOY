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


@dataclass(frozen=True)
class ResidualWaveExchange:
    target_wave_energy_increment: float
    target_wave_angular_momentum_increment: np.ndarray
    resolved_wave_energy_increment: float
    resolved_wave_angular_momentum_increment: np.ndarray
    residual_wave_energy_increment: float
    residual_wave_angular_momentum_increment: np.ndarray
    energy_closure_error: float
    angular_momentum_closure_error: np.ndarray


def residual_wave_exchange(
    *,
    orbital_energy_increment: float,
    orbital_angular_momentum_increment: np.ndarray,
    resolved_wave_energy_increment: float,
    resolved_wave_angular_momentum_increment: np.ndarray,
) -> ResidualWaveExchange:
    """Subtract resolved wave work and torque from the conserved target.

    The target wave increment is exactly opposite to the internal orbital
    increment. Work and torque already produced by the resolved multipole
    potential are removed before any additional wave-mode source is applied.
    """

    orbital_angular_momentum = np.asarray(
        orbital_angular_momentum_increment, dtype=float
    )
    resolved_angular_momentum = np.asarray(
        resolved_wave_angular_momentum_increment, dtype=float
    )
    scalar_values = np.asarray(
        [orbital_energy_increment, resolved_wave_energy_increment], dtype=float
    )
    if (
        orbital_angular_momentum.shape != (3,)
        or resolved_angular_momentum.shape != (3,)
        or np.any(~np.isfinite(orbital_angular_momentum))
        or np.any(~np.isfinite(resolved_angular_momentum))
        or np.any(~np.isfinite(scalar_values))
    ):
        raise ValueError("exchange increments must be finite scalars and vectors")
    target_energy = -float(orbital_energy_increment)
    target_angular_momentum = -orbital_angular_momentum
    residual_energy = target_energy - float(resolved_wave_energy_increment)
    residual_angular_momentum = (
        target_angular_momentum - resolved_angular_momentum
    )
    energy_closure = (
        float(orbital_energy_increment)
        + float(resolved_wave_energy_increment)
        + residual_energy
    )
    angular_momentum_closure = (
        orbital_angular_momentum
        + resolved_angular_momentum
        + residual_angular_momentum
    )
    return ResidualWaveExchange(
        target_wave_energy_increment=target_energy,
        target_wave_angular_momentum_increment=target_angular_momentum,
        resolved_wave_energy_increment=float(resolved_wave_energy_increment),
        resolved_wave_angular_momentum_increment=resolved_angular_momentum,
        residual_wave_energy_increment=residual_energy,
        residual_wave_angular_momentum_increment=residual_angular_momentum,
        energy_closure_error=energy_closure,
        angular_momentum_closure_error=angular_momentum_closure,
    )


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
