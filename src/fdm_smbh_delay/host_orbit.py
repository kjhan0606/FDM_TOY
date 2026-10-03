"""Bound-orbit diagnostics for a spherical, static host potential.

The turning-point eccentricity here describes the secondary's orbit in the
host plus the primary--secondary reflex point mass.  It is not the SMBH-only
Keplerian eccentricity used after binary binding.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq

from .constants import G_INTERNAL
from .fdm_orbital_response import (
    FDMProjectedOrbitalResponse,
    project_fdm_response_family_to_orbit,
)
from .fdm_outer_response_family import FDMOuterResponseFamily
from .galaxy_environment import CompositePotential
from .profile_table import ProfileSupportError


@dataclass(frozen=True)
class HostOrbitTurningPoints:
    status: str
    pericentre_pc: float | None
    apocentre_pc: float | None
    eccentricity: float | None
    specific_energy_pc2_myr2: float | None
    specific_angular_momentum_pc2_myr: float | None
    reason: str


@dataclass(frozen=True)
class FDMHostResponseDecision:
    status: str
    host_orbit: HostOrbitTurningPoints | None
    projected_response: FDMProjectedOrbitalResponse | None
    reason: str


def spherical_host_turning_points(
    *,
    host_potential: CompositePotential,
    position_pc: np.ndarray,
    velocity_pc_myr: np.ndarray,
    secondary_bh_mass_msun: float,
    radial_support_pc: tuple[float, float],
) -> HostOrbitTurningPoints:
    """Find both radial turning points without extending profile support.

    The relative equation includes the secondary's reflex acceleration.  The
    host must be spherical, static, and supply a potential with a consistent
    zero.  A finite support bracket is mandatory even for analytic profiles;
    an unbracketed or non-bound orbit returns ``censored``.
    """

    position = np.asarray(position_pc, dtype=float)
    velocity = np.asarray(velocity_pc_myr, dtype=float)
    if (
        position.shape != (3,) or velocity.shape != (3,)
        or np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity))
        or not math.isfinite(secondary_bh_mass_msun)
        or secondary_bh_mass_msun <= 0.0
        or len(radial_support_pc) != 2
    ):
        raise ValueError("spherical host orbit inputs are invalid")
    minimum, maximum = (float(value) for value in radial_support_pc)
    radius = float(np.linalg.norm(position))
    if (
        not math.isfinite(minimum) or not math.isfinite(maximum)
        or not 0.0 < minimum < radius < maximum
    ):
        raise ValueError("host radial support must strictly enclose a positive radius")

    def censored(reason: str, energy: float | None = None, angular: float | None = None) -> HostOrbitTurningPoints:
        return HostOrbitTurningPoints("censored", None, None, None, energy, angular, reason)

    angular = float(np.linalg.norm(np.cross(position, velocity)))
    if angular == 0.0:
        return censored("radial host orbit has no positive inner turning point", angular=angular)

    def effective_potential(sample_radius: float) -> float:
        return (
            float(host_potential.potential(sample_radius))
            - G_INTERNAL * secondary_bh_mass_msun / sample_radius
            + 0.5 * angular**2 / sample_radius**2
        )

    try:
        energy = 0.5 * float(np.dot(velocity, velocity)) + (
            float(host_potential.potential(radius))
            - G_INTERNAL * secondary_bh_mass_msun / radius
        )
        radial_speed = float(np.dot(position, velocity)) / radius
        tangential_speed2 = angular**2 / radius**2
        circular_speed2 = (
            G_INTERNAL
            * (float(host_potential.enclosed_mass(radius)) + secondary_bh_mass_msun)
            / radius
        )
        speed_scale2 = max(tangential_speed2, circular_speed2)
        if (
            radial_speed**2 <= 1.0e-12 * speed_scale2
            and abs(tangential_speed2 - circular_speed2) <= 1.0e-12 * speed_scale2
        ):
            return HostOrbitTurningPoints(
                "bound_candidate_static_host", radius, radius, 0.0,
                energy, angular,
                "circular static spherical host diagnostic only; no physical delay inferred",
            )

        def radial_energy(sample_radius: float) -> float:
            if sample_radius == radius:
                # A state exactly at peri/apo has zero radial speed.  Give
                # each root search a positive interior endpoint, otherwise
                # Brent returns the current radius for *both* roots.
                return max(0.5 * radial_speed**2, np.finfo(float).tiny)
            return energy - effective_potential(sample_radius)

        inner_value = radial_energy(minimum)
        outer_value = radial_energy(maximum)
        if not all(math.isfinite(value) for value in (energy, inner_value, outer_value)):
            return censored("host turning-point energy is non-finite", energy, angular)
        if inner_value > 0.0 or outer_value > 0.0:
            return censored("host orbit turning point lies outside declared radial support", energy, angular)
        peri = float(brentq(radial_energy, minimum, radius))
        apo = float(brentq(radial_energy, radius, maximum))
    except (ProfileSupportError, ValueError) as error:
        return censored(f"host profile cannot support turning-point search: {error}", angular=angular)
    if not 0.0 < peri <= radius <= apo or not math.isfinite(apo):
        return censored("host turning-point roots are invalid", energy, angular)
    eccentricity = (apo - peri) / (apo + peri)
    if not 0.0 <= eccentricity < 1.0:
        return censored("host turning-point eccentricity is invalid", energy, angular)
    return HostOrbitTurningPoints(
        "bound_candidate_static_host", peri, apo, eccentricity,
        energy, angular,
        "static spherical host diagnostic only; no merger torque or physical delay inferred",
    )


def project_fdm_response_family_in_host(
    *,
    family: FDMOuterResponseFamily,
    host_potential: CompositePotential,
    position_pc: np.ndarray,
    velocity_pc_myr: np.ndarray,
    secondary_bh_mass_msun: float,
    radial_support_pc: tuple[float, float],
) -> FDMHostResponseDecision:
    """Bind the q/e response query to a static-host orbit diagnostic.

    The host's central point mass is the primary SMBH, so q is derived rather
    than supplied independently.  The host turning-point eccentricity is
    meaningful only where the static spherical approximation applies; a
    galaxy-merger zoom must validate that approximation before physical use.
    """

    primary = host_potential.central_point_mass_msun
    if primary <= 0.0 or not 0.0 < secondary_bh_mass_msun <= primary:
        return FDMHostResponseDecision(
            "censored", None, None,
            "host response requires a central primary SMBH and 0 < q <= 1",
        )
    host_orbit = spherical_host_turning_points(
        host_potential=host_potential,
        position_pc=position_pc,
        velocity_pc_myr=velocity_pc_myr,
        secondary_bh_mass_msun=secondary_bh_mass_msun,
        radial_support_pc=radial_support_pc,
    )
    if host_orbit.status == "censored":
        return FDMHostResponseDecision("censored", host_orbit, None, host_orbit.reason)
    assert host_orbit.eccentricity is not None
    projected = project_fdm_response_family_to_orbit(
        family,
        position_pc=position_pc,
        velocity_pc_myr=velocity_pc_myr,
        mass_ratio_q=secondary_bh_mass_msun / primary,
        eccentricity=host_orbit.eccentricity,
    )
    if projected.status == "censored":
        return FDMHostResponseDecision("censored", host_orbit, None, projected.reason)
    return FDMHostResponseDecision(
        "candidate_pending_zoom_validation", host_orbit, projected,
        "static-host q/e/r projection only; no physical delay inferred",
    )
