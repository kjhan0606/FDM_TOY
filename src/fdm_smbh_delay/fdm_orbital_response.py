"""Frame-aware, no-extrapolation projection of an outer FDM response.

This is an applicability and coordinate contract, not a stochastic orbit
solver or a physical delay calibration.  A radius-only response cannot be
rotated into an orbit until its component frame has been declared.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .fdm_outer_response import FDMOuterResponseTable


@dataclass(frozen=True)
class FDMOrbitalResponseSupport:
    minimum_mass_ratio_q: float
    maximum_mass_ratio_q: float
    minimum_eccentricity: float
    maximum_eccentricity: float

    def __post_init__(self) -> None:
        values = (
            self.minimum_mass_ratio_q, self.maximum_mass_ratio_q,
            self.minimum_eccentricity, self.maximum_eccentricity,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or not 0.0 < self.minimum_mass_ratio_q <= self.maximum_mass_ratio_q <= 1.0
            or not 0.0 <= self.minimum_eccentricity <= self.maximum_eccentricity < 1.0
        ):
            raise ValueError("FDM orbital q/e support is invalid")


@dataclass(frozen=True)
class FDMProjectedOrbitalResponse:
    status: str
    radius_pc: float
    mass_ratio_q: float
    eccentricity: float
    drift_acceleration_pc_myr2: np.ndarray | None
    diffusion_tensor_pc2_myr3: np.ndarray | None
    reason: str


def project_fdm_outer_response_to_orbit(
    table: FDMOuterResponseTable,
    support: FDMOrbitalResponseSupport,
    *,
    position_pc: np.ndarray,
    velocity_pc_myr: np.ndarray,
    mass_ratio_q: float,
    eccentricity: float,
    minimum_orbital_sine: float = 1.0e-8,
) -> FDMProjectedOrbitalResponse:
    """Rotate RTN drift/diffusion into Cartesian space within q/e/r support.

    RTN is the right-handed radial, tangential, normal frame of the supplied
    orbit.  A nearly radial state has no stable normal axis and is censored.
    The returned response remains a *candidate* until an independently
    verified calibration and stochastic evolution consume it.
    """

    position = np.asarray(position_pc, dtype=float)
    velocity = np.asarray(velocity_pc_myr, dtype=float)
    if (
        position.shape != (3,) or velocity.shape != (3,)
        or np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity))
        or not math.isfinite(mass_ratio_q) or not 0.0 < mass_ratio_q <= 1.0
        or not math.isfinite(eccentricity) or not 0.0 <= eccentricity < 1.0
        or not math.isfinite(minimum_orbital_sine)
        or not 0.0 < minimum_orbital_sine < 1.0
    ):
        raise ValueError("FDM orbital response state or frame control is invalid")
    radius = float(np.linalg.norm(position))

    def censored(reason: str) -> FDMProjectedOrbitalResponse:
        return FDMProjectedOrbitalResponse(
            status="censored", radius_pc=radius,
            mass_ratio_q=mass_ratio_q, eccentricity=eccentricity,
            drift_acceleration_pc_myr2=None,
            diffusion_tensor_pc2_myr3=None, reason=reason,
        )

    if table.component_frame != "orbital_rtn":
        return censored("FDM response components lack an orbital RTN frame")
    if not (support.minimum_mass_ratio_q <= mass_ratio_q <= support.maximum_mass_ratio_q):
        return censored("FDM response mass ratio lies outside calibrated support")
    if not (support.minimum_eccentricity <= eccentricity <= support.maximum_eccentricity):
        return censored("FDM response eccentricity lies outside calibrated support")
    decision = table.decision(radius)
    if decision["status"] != "available":
        return censored(str(decision["reason"]))
    speed = float(np.linalg.norm(velocity))
    angular = np.cross(position, velocity)
    angular_norm = float(np.linalg.norm(angular))
    if (
        radius == 0.0 or speed == 0.0
        or angular_norm / (radius * speed) < minimum_orbital_sine
    ):
        return censored("FDM response orbital RTN frame is undefined for a radial state")
    radial = position / radius
    normal = angular / angular_norm
    tangential = np.cross(normal, radial)
    basis = np.column_stack((radial, tangential, normal))
    drift = basis @ decision["drift_acceleration_pc_myr2"]
    diffusion = basis @ decision["diffusion_tensor_pc2_myr3"] @ basis.T
    diffusion = 0.5 * (diffusion + diffusion.T)
    if (
        np.any(~np.isfinite(drift)) or np.any(~np.isfinite(diffusion))
        or np.min(np.linalg.eigvalsh(diffusion)) < -1.0e-12
    ):
        return censored("FDM response projection violated finite/PSD contract")
    return FDMProjectedOrbitalResponse(
        status="projected_candidate_pending_source_calibration",
        radius_pc=radius, mass_ratio_q=mass_ratio_q, eccentricity=eccentricity,
        drift_acceleration_pc_myr2=drift,
        diffusion_tensor_pc2_myr3=diffusion,
        reason="frame and q/e/r support only; no physical delay inferred",
    )
