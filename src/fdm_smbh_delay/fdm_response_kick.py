"""Numerical velocity kick from a candidate outer FDM response.

This samples one local white-noise increment.  It neither evolves an orbit
nor establishes that the measured wake is Markovian or physically calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .fdm_orbital_response import FDMProjectedOrbitalResponse
from .fdm_outer_halo import FDMOuterHaloClosure


MIN_WHITE_NOISE_COHERENCE_RATIO = 10.0


@dataclass(frozen=True)
class FDMResponseKick:
    status: str
    velocity_increment_pc_myr: np.ndarray | None
    deterministic_increment_pc_myr: np.ndarray | None
    stochastic_increment_pc_myr: np.ndarray | None
    reason: str
    source_sha256: tuple[str, ...]
    maximum_coherence_time_myr: float | None = None


def sample_candidate_fdm_velocity_kick(
    response: FDMProjectedOrbitalResponse,
    *,
    time_step_myr: float,
    random_seed: int,
    completed_steps: int,
    closure_by_source_sha256: Mapping[str, FDMOuterHaloClosure],
) -> FDMResponseKick:
    """Sample ``Cov[delta v] = D_v * dt`` with a restart-stable stream.

    The seed and accepted-step index select one draw independently of prior
    calls, so a resumed trajectory can reproduce its next numerical kick.
    Every source corner must supply a calibrated, co-supported closure.  A
    step shorter than ten times the largest corner coherence time is censored
    instead of assuming independent Gaussian increments.  Passing this
    conservative numerical gate is *not* a proof of a white-noise limit or a
    physical delay.
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
    if not isinstance(closure_by_source_sha256, Mapping):
        raise ValueError("FDM source closure mapping is required")
    maximum_coherence = 0.0
    for source in dict.fromkeys(response.source_sha256):
        closure = closure_by_source_sha256.get(source)
        if not isinstance(closure, FDMOuterHaloClosure):
            return censored(f"FDM source {source} lacks a measured coherence closure")
        if closure.closure_status != "calibrated":
            return censored(f"FDM source {source} coherence closure is not calibrated")
        try:
            coherence = float(closure.evaluate(response.radius_pc)["coherence_time_myr"])
        except ValueError as error:
            return censored(f"FDM source {source} coherence support: {error}")
        if not math.isfinite(coherence) or coherence <= 0.0:
            return censored(f"FDM source {source} coherence time is invalid")
        maximum_coherence = max(maximum_coherence, coherence)
    if time_step_myr < MIN_WHITE_NOISE_COHERENCE_RATIO * maximum_coherence:
        return censored(
            "FDM step is shorter than ten measured coherence times; "
            "a colored-noise model is required"
        )
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
        "candidate_pending_physical_validation",
        increment, deterministic, stochastic,
        "coherence-gated numerical sample only; white-noise validity and physical delay unproven",
        response.source_sha256,
        maximum_coherence,
    )
