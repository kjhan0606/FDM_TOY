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
from .fdm_outer_response_family import FDMOuterResponseFamily


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
    source_sha256: tuple[str, ...] = ()
    diffusion_convention: str = "unspecified"


def _orbital_basis(
    position: np.ndarray, velocity: np.ndarray, minimum_orbital_sine: float
) -> np.ndarray | None:
    radius = float(np.linalg.norm(position))
    speed = float(np.linalg.norm(velocity))
    angular = np.cross(position, velocity)
    angular_norm = float(np.linalg.norm(angular))
    if (
        not math.isfinite(radius) or not math.isfinite(speed)
        or not math.isfinite(angular_norm) or radius == 0.0 or speed == 0.0
        or angular_norm / (radius * speed) < minimum_orbital_sine
    ):
        return None
    radial = position / radius
    normal = angular / angular_norm
    tangential = np.cross(normal, radial)
    return np.column_stack((radial, tangential, normal))


def _project_values(
    position: np.ndarray,
    velocity: np.ndarray,
    minimum_orbital_sine: float,
    drift_rtn: np.ndarray,
    diffusion_rtn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    basis = _orbital_basis(position, velocity, minimum_orbital_sine)
    if basis is None:
        return None
    drift = basis @ drift_rtn
    diffusion = basis @ diffusion_rtn @ basis.T
    diffusion = 0.5 * (diffusion + diffusion.T)
    if (
        np.any(~np.isfinite(drift)) or np.any(~np.isfinite(diffusion))
        or np.min(np.linalg.eigvalsh(diffusion)) < -1.0e-12
    ):
        return None
    return drift, diffusion


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
    projected = _project_values(
        position, velocity, minimum_orbital_sine,
        decision["drift_acceleration_pc_myr2"],
        decision["diffusion_tensor_pc2_myr3"],
    )
    if projected is None:
        return censored("FDM response orbital frame or finite/PSD projection is invalid")
    drift, diffusion = projected
    return FDMProjectedOrbitalResponse(
        status="projected_candidate_pending_source_calibration",
        radius_pc=radius, mass_ratio_q=mass_ratio_q, eccentricity=eccentricity,
        drift_acceleration_pc_myr2=drift,
        diffusion_tensor_pc2_myr3=diffusion,
        reason="frame and q/e/r support only; no physical delay inferred",
        diffusion_convention=table.diffusion_convention,
    )


def project_fdm_response_family_to_orbit(
    family: FDMOuterResponseFamily,
    *,
    position_pc: np.ndarray,
    velocity_pc_myr: np.ndarray,
    mass_ratio_q: float,
    eccentricity: float,
    minimum_orbital_sine: float = 1.0e-8,
) -> FDMProjectedOrbitalResponse:
    """Project a source-bound q/e response candidate into Cartesian space."""

    position = np.asarray(position_pc, dtype=float)
    velocity = np.asarray(velocity_pc_myr, dtype=float)
    if (
        position.shape != (3,) or velocity.shape != (3,)
        or np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity))
        or not math.isfinite(minimum_orbital_sine)
        or not 0.0 < minimum_orbital_sine < 1.0
    ):
        raise ValueError("FDM orbital response state or frame control is invalid")
    radius = float(np.linalg.norm(position))
    decision = family.decision(mass_ratio_q, eccentricity, radius)

    def censored(reason: str) -> FDMProjectedOrbitalResponse:
        return FDMProjectedOrbitalResponse(
            status="censored", radius_pc=radius,
            mass_ratio_q=mass_ratio_q, eccentricity=eccentricity,
            drift_acceleration_pc_myr2=None,
            diffusion_tensor_pc2_myr3=None, reason=reason,
        )

    if decision["status"] == "censored":
        return censored(str(decision["reason"]))
    projected = _project_values(
        position, velocity, minimum_orbital_sine,
        decision["drift_acceleration_pc_myr2"],
        decision["diffusion_tensor_pc2_myr3"],
    )
    if projected is None:
        return censored("FDM response orbital frame or finite/PSD projection is invalid")
    drift, diffusion = projected
    return FDMProjectedOrbitalResponse(
        status="projected_candidate_pending_physical_validation",
        radius_pc=radius, mass_ratio_q=mass_ratio_q, eccentricity=eccentricity,
        drift_acceleration_pc_myr2=drift,
        diffusion_tensor_pc2_myr3=diffusion,
        reason="q/e/r mixture and orbital projection only; no physical delay inferred",
        source_sha256=decision["source_sha256"],
        diffusion_convention=decision["diffusion_convention"],
    )
