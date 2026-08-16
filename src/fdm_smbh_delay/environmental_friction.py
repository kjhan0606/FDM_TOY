"""Resolved-scale stellar, gaseous, and FDM dynamical-friction terms."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf

import numpy as np

from .constants import G_INTERNAL
from .galaxy_environment import FDMBackground, GasBackground, StellarBackground
from .wave_drag import DragEvaluation, evaluate_drag


@dataclass(frozen=True)
class FrictionEvaluation:
    acceleration_pc_myr2: np.ndarray
    density_msun_pc3: float
    relative_velocity_pc_myr: np.ndarray
    coefficient: float
    coulomb_logarithm: float | None
    stalled: bool
    reason: str


@dataclass(frozen=True)
class FDMFrictionEvaluation:
    acceleration_pc_myr2: np.ndarray
    analytic_applied: bool
    live_wake_resolved: bool
    drag: DragEvaluation | None


def maxwellian_slow_fraction(speed_pc_myr: float, sigma_pc_myr: float) -> float:
    """Fraction in the standard Maxwellian Chandrasekhar bracket."""

    if (
        not np.isfinite(speed_pc_myr)
        or not np.isfinite(sigma_pc_myr)
        or speed_pc_myr < 0.0
        or sigma_pc_myr <= 0.0
    ):
        raise ValueError("speed and velocity dispersion are invalid")
    x = speed_pc_myr / (np.sqrt(2.0) * sigma_pc_myr)
    return float(erf(x) - 2.0 * x * np.exp(-x * x) / np.sqrt(np.pi))


def stellar_dynamical_friction(
    *,
    perturber_mass_msun: float,
    position_pc: np.ndarray,
    velocity_pc_myr: np.ndarray,
    background: StellarBackground,
    minimum_impact_parameter_pc: float,
    maximum_impact_parameter_pc: float | None = None,
    core_stall_enclosed_mass_ratio: float = 0.0,
    velocity_floor_pc_myr: float = 1.0e-12,
) -> FrictionEvaluation:
    """Evaluate a conservative Maxwellian stellar-friction bracket.

    Setting ``core_stall_enclosed_mass_ratio`` to a positive value suppresses
    this term once the enclosed stellar mass is no larger than that multiple
    of the effective perturber mass.  This is an explicit bracket, not a claim
    that a Maxwellian closure resolves a depleted or anisotropic core.
    """

    position = np.asarray(position_pc, dtype=float)
    velocity = np.asarray(velocity_pc_myr, dtype=float)
    controls = np.asarray(
        [
            perturber_mass_msun,
            minimum_impact_parameter_pc,
            core_stall_enclosed_mass_ratio,
            velocity_floor_pc_myr,
        ],
        dtype=float,
    )
    if (
        position.shape != (3,)
        or velocity.shape != (3,)
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(controls))
        or perturber_mass_msun <= 0.0
        or minimum_impact_parameter_pc <= 0.0
        or core_stall_enclosed_mass_ratio < 0.0
        or velocity_floor_pc_myr <= 0.0
    ):
        raise ValueError("stellar-friction inputs are invalid")
    radius = float(np.linalg.norm(position))
    relative_velocity = velocity - background.bulk_velocity_pc_myr
    speed = float(np.linalg.norm(relative_velocity))
    density = float(background.profile.density(radius))
    enclosed = float(background.profile.enclosed_mass(radius))
    if maximum_impact_parameter_pc is None:
        bmax = radius
    else:
        if not np.isfinite(maximum_impact_parameter_pc) or maximum_impact_parameter_pc <= 0.0:
            raise ValueError("maximum impact parameter must be finite and positive")
        bmax = min(radius, maximum_impact_parameter_pc)
    focusing = G_INTERNAL * perturber_mass_msun / (
        speed**2 + background.velocity_dispersion_pc_myr**2
    )
    bmin = max(minimum_impact_parameter_pc, focusing)
    coulomb_log = float(np.log(bmax / bmin)) if bmax > bmin else 0.0
    core_stalled = (
        core_stall_enclosed_mass_ratio > 0.0
        and enclosed
        <= core_stall_enclosed_mass_ratio * perturber_mass_msun
    )
    if radius == 0.0 or speed <= velocity_floor_pc_myr:
        return FrictionEvaluation(
            np.zeros(3),
            density,
            relative_velocity,
            0.0,
            coulomb_log,
            False,
            "relative velocity is below the friction floor",
        )
    if core_stalled:
        return FrictionEvaluation(
            np.zeros(3),
            density,
            relative_velocity,
            0.0,
            coulomb_log,
            True,
            "enclosed stellar mass reached the explicit core-stalling bracket",
        )
    slow_fraction = maxwellian_slow_fraction(
        speed, background.velocity_dispersion_pc_myr
    )
    coefficient = max(coulomb_log * slow_fraction, 0.0)
    acceleration = (
        -4.0
        * np.pi
        * G_INTERNAL**2
        * perturber_mass_msun
        * density
        * coefficient
        * relative_velocity
        / speed**3
    )
    return FrictionEvaluation(
        acceleration,
        density,
        relative_velocity,
        coefficient,
        coulomb_log,
        False,
        "Maxwellian stellar dynamical friction",
    )


def ostriker_gas_coefficient(
    mach_number: float,
    coulomb_logarithm: float,
    *,
    sonic_regularization: float = 1.0e-3,
) -> float:
    """Finite-time gaseous-wake coefficient with a regularized sonic point."""

    values = np.asarray(
        [mach_number, coulomb_logarithm, sonic_regularization], dtype=float
    )
    if (
        np.any(~np.isfinite(values))
        or mach_number < 0.0
        or coulomb_logarithm < 0.0
        or not 0.0 < sonic_regularization < 1.0
    ):
        raise ValueError("gaseous-drag controls are invalid")

    def subsonic(mach: float) -> float:
        if mach == 0.0:
            return 0.0
        return 0.5 * np.log((1.0 + mach) / (1.0 - mach)) - mach

    def supersonic(mach: float) -> float:
        return 0.5 * np.log(1.0 - mach**-2) + coulomb_logarithm

    lower = 1.0 - sonic_regularization
    upper = 1.0 + sonic_regularization
    if mach_number < lower:
        coefficient = subsonic(mach_number)
    elif mach_number > upper:
        coefficient = supersonic(mach_number)
    else:
        weight = (mach_number - lower) / (upper - lower)
        coefficient = (1.0 - weight) * subsonic(lower) + weight * supersonic(upper)
    return float(max(coefficient, 0.0))


def gaseous_dynamical_friction(
    *,
    perturber_mass_msun: float,
    position_pc: np.ndarray,
    velocity_pc_myr: np.ndarray,
    total_enclosed_mass_msun: float,
    background: GasBackground,
    coulomb_logarithm: float,
    velocity_floor_pc_myr: float = 1.0e-12,
) -> FrictionEvaluation:
    position = np.asarray(position_pc, dtype=float)
    velocity = np.asarray(velocity_pc_myr, dtype=float)
    controls = np.asarray(
        [
            perturber_mass_msun,
            total_enclosed_mass_msun,
            coulomb_logarithm,
            velocity_floor_pc_myr,
        ],
        dtype=float,
    )
    if (
        position.shape != (3,)
        or velocity.shape != (3,)
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(controls))
        or perturber_mass_msun <= 0.0
        or total_enclosed_mass_msun < 0.0
        or coulomb_logarithm < 0.0
        or velocity_floor_pc_myr <= 0.0
    ):
        raise ValueError("gaseous-friction inputs are invalid")
    radius = float(np.linalg.norm(position))
    bulk_velocity = background.bulk_velocity(position, total_enclosed_mass_msun)
    relative_velocity = velocity - bulk_velocity
    speed = float(np.linalg.norm(relative_velocity))
    density = float(background.profile.density(radius))
    if speed <= velocity_floor_pc_myr:
        return FrictionEvaluation(
            np.zeros(3),
            density,
            relative_velocity,
            0.0,
            coulomb_logarithm,
            False,
            "relative gas velocity is below the friction floor",
        )
    mach = speed / background.sound_speed_pc_myr
    coefficient = ostriker_gas_coefficient(mach, coulomb_logarithm)
    acceleration = (
        -4.0
        * np.pi
        * G_INTERNAL**2
        * perturber_mass_msun
        * density
        * coefficient
        * relative_velocity
        / speed**3
    )
    return FrictionEvaluation(
        acceleration,
        density,
        relative_velocity,
        coefficient,
        coulomb_logarithm,
        False,
        "Mach-dependent gaseous dynamical friction",
    )


def fdm_dynamical_friction(
    *,
    perturber_mass_msun: float,
    position_pc: np.ndarray,
    velocity_pc_myr: np.ndarray,
    background: FDMBackground,
) -> FDMFrictionEvaluation:
    """Apply analytic FDM drag only when a live wake is not resolved."""

    if background.mode == "live_resolved":
        return FDMFrictionEvaluation(
            acceleration_pc_myr2=np.zeros(3),
            analytic_applied=False,
            live_wake_resolved=True,
            drag=None,
        )
    radius = float(np.linalg.norm(position_pc))
    drag = evaluate_drag(
        mass_msun=perturber_mass_msun,
        position_pc=np.asarray(position_pc, dtype=float),
        velocity_pc_myr=np.asarray(velocity_pc_myr, dtype=float),
        separation_pc=radius,
        soliton=background.soliton,
        m_fdm_ev=background.particle_mass_ev,
        alpha_df=background.alpha_df,
        bulk_velocity_pc_myr=background.bulk_velocity_pc_myr,
        velocity_floor_pc_myr=background.velocity_floor_pc_myr,
    )
    return FDMFrictionEvaluation(
        acceleration_pc_myr2=drag.acceleration_pc_myr2,
        analytic_applied=True,
        live_wake_resolved=False,
        drag=drag,
    )
