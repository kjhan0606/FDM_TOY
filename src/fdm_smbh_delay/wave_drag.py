"""Orbit-averaged fuzzy-dark-matter wave drag."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import sici

from .constants import G_INTERNAL, KM_S_TO_PC_MYR, Q_PER_EV_PC2_PER_MYR
from .soliton import SphericalSoliton


SMALL_Q_THRESHOLD = 1.0e-2


def drag_coefficient(q: float | np.ndarray) -> float | np.ndarray:
    """Return Hui's point-perturber coefficient with a stable small-q branch."""

    q_array = np.asarray(q, dtype=float)
    if np.any(q_array < 0.0) or np.any(~np.isfinite(q_array)):
        raise ValueError("q must be finite and non-negative")
    result = np.empty_like(q_array)
    small = q_array < SMALL_Q_THRESHOLD
    q2 = q_array[small] ** 2
    result[small] = q2 / 3.0 - q2**2 / 30.0 + 2.0 * q2**3 / 945.0
    if np.any(~small):
        x = 2.0 * q_array[~small]
        _, cosine_integral = sici(x)
        cin = np.euler_gamma + np.log(x) - cosine_integral
        result[~small] = cin + np.sin(x) / x - 1.0
    # Roundoff can make a theoretically non-negative coefficient slightly low.
    result = np.maximum(result, 0.0)
    if np.ndim(q) == 0:
        return float(result)
    return result


def wave_parameter(m_fdm_ev: float, speed_pc_myr: float, radius_pc: float) -> float:
    """Dimensionless de Broglie response parameter q."""

    return float(Q_PER_EV_PC2_PER_MYR * m_fdm_ev * speed_pc_myr * radius_pc)


def nonlinear_parameter(mass_msun: float, m_fdm_ev: float, speed_pc_myr: float) -> float:
    """Lancaster et al. nonlinear-response estimate."""

    if speed_pc_myr <= 0.0:
        return float("inf")
    hundred_kms = 100.0 * KM_S_TO_PC_MYR
    return float((mass_msun / 1.0e9) * (m_fdm_ev / 1.0e-22) * (hundred_kms / speed_pc_myr))


@dataclass(frozen=True)
class DragEvaluation:
    acceleration_pc_myr2: np.ndarray
    force_msun_pc_myr2: np.ndarray
    relative_velocity_pc_myr: np.ndarray
    density_msun_pc3: float
    q: float
    eta_nl: float
    enclosed_to_bh_mass: float
    used_velocity_floor: bool


def evaluate_drag(
    *,
    mass_msun: float,
    position_pc: np.ndarray,
    velocity_pc_myr: np.ndarray,
    separation_pc: float,
    soliton: SphericalSoliton,
    m_fdm_ev: float,
    alpha_df: float,
    bulk_velocity_pc_myr: np.ndarray,
    velocity_floor_pc_myr: float,
) -> DragEvaluation:
    """Evaluate force and diagnostics for one SMBH."""

    position = np.asarray(position_pc, dtype=float)
    relative_velocity = np.asarray(velocity_pc_myr, dtype=float) - np.asarray(
        bulk_velocity_pc_myr, dtype=float
    )
    radius = float(np.linalg.norm(position))
    speed = float(np.linalg.norm(relative_velocity))
    density = float(soliton.density(radius))
    enclosed_ratio = float(soliton.enclosed_mass(radius) / mass_msun)

    if speed <= velocity_floor_pc_myr:
        zeros = np.zeros(3, dtype=float)
        return DragEvaluation(
            zeros,
            zeros,
            relative_velocity,
            density,
            0.0,
            nonlinear_parameter(mass_msun, m_fdm_ev, speed),
            enclosed_ratio,
            True,
        )

    effective_radius = alpha_df * min(0.5 * separation_pc, soliton.core_radius_pc)
    q = wave_parameter(m_fdm_ev, speed, effective_radius)
    coefficient = drag_coefficient(q)
    force = (
        -4.0
        * np.pi
        * G_INTERNAL**2
        * mass_msun**2
        * density
        * coefficient
        * relative_velocity
        / speed**3
    )
    return DragEvaluation(
        acceleration_pc_myr2=force / mass_msun,
        force_msun_pc_myr2=force,
        relative_velocity_pc_myr=relative_velocity,
        density_msun_pc3=density,
        q=q,
        eta_nl=nonlinear_parameter(mass_msun, m_fdm_ev, speed),
        enclosed_to_bh_mass=enclosed_ratio,
        used_velocity_floor=False,
    )
