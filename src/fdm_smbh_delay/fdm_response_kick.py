"""Numerical velocity kick from a candidate outer FDM response.

This samples one local white-noise increment.  It neither evolves an orbit
nor establishes that the measured wake is Markovian or physically calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .fdm_orbital_response import FDMProjectedOrbitalResponse


@dataclass(frozen=True)
class FDMResponseKick:
    status: str
    velocity_increment_pc_myr: np.ndarray | None
    deterministic_increment_pc_myr: np.ndarray | None
    stochastic_increment_pc_myr: np.ndarray | None
    reason: str
    source_sha256: tuple[str, ...]


def sample_candidate_fdm_velocity_kick(
    response: FDMProjectedOrbitalResponse,
    *,
    time_step_myr: float,
    random_seed: int,
    completed_steps: int,
) -> FDMResponseKick:
    """Sample ``Cov[delta v] = D_v * dt`` with a restart-stable stream.

    The seed and accepted-step index select one draw independently of prior
    calls, so a resumed trajectory can reproduce its next numerical kick.
    The caller must separately justify a white-noise limit from measured
    coherence times; this function never returns a physical delay.
    """

    if not math.isfinite(time_step_myr) or time_step_myr <= 0.0:
        raise ValueError("FDM kick time step must be finite and positive")
    if (
        type(random_seed) is not int or not 0 <= random_seed <= 0xFFFFFFFF
        or type(completed_steps) is not int or not 0 <= completed_steps <= 0xFFFFFFFF
    ):
        raise ValueError("FDM kick seed and accepted-step index must be uint32 integers")

    def censored(reason: str) -> FDMResponseKick:
        return FDMResponseKick("censored", None, None, None, reason, response.source_sha256)

    if response.status != "projected_candidate_pending_physical_validation":
        return censored("FDM response lacks a source-bound q/e orbital projection")
    if not response.source_sha256:
        return censored("FDM response has no source identifiers")
    if response.diffusion_convention != "velocity_covariance_rate":
        return censored("FDM velocity-diffusion covariance convention is unspecified")
    drift = response.drift_acceleration_pc_myr2
    diffusion = response.diffusion_tensor_pc2_myr3
    if drift is None or diffusion is None:
        return censored("FDM projected drift or diffusion is missing")
    drift = np.asarray(drift, dtype=float)
    diffusion = np.asarray(diffusion, dtype=float)
    if (
        drift.shape != (3,) or diffusion.shape != (3, 3)
        or np.any(~np.isfinite(drift)) or np.any(~np.isfinite(diffusion))
        or not np.allclose(diffusion, diffusion.T, rtol=0.0, atol=1.0e-10)
    ):
        return censored("FDM projected drift/diffusion is not finite and symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (diffusion + diffusion.T))
    if np.min(eigenvalues) < -1.0e-12:
        return censored("FDM projected velocity diffusion is not positive semidefinite")
    normal = np.random.default_rng(
        np.random.SeedSequence([random_seed, completed_steps])
    ).standard_normal(3)
    stochastic = eigenvectors @ (np.sqrt(np.maximum(eigenvalues, 0.0) * time_step_myr) * normal)
    deterministic = drift * time_step_myr
    increment = deterministic + stochastic
    if np.any(~np.isfinite(increment)):
        return censored("FDM velocity increment overflowed")
    return FDMResponseKick(
        "candidate_pending_coherence_and_physical_validation",
        increment, deterministic, stochastic,
        "local white-noise numerical sample only; no physical delay inferred",
        response.source_sha256,
    )
