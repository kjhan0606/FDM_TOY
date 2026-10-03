"""Censor-first numerical orbit path for candidate outer FDM responses.

This split integrator couples a static spherical host, stellar/gas friction,
tidal envelope truncation, and one local FDM drift/diffusion kick per accepted
step.  It does not calibrate galaxy-merger torques or publish a physical delay.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .constants import G_INTERNAL
from .fdm_outer_halo import FDMOuterHaloClosure
from .fdm_outer_response_family import FDMOuterResponseFamily
from .fdm_response_kick import sample_candidate_fdm_velocity_kick
from .host_orbit import project_fdm_response_family_in_host
from .kpc_inspiral import (
    DualNucleusState,
    KpcIntegrationConfig,
    KpcInspiralModel,
    _advance_phase_aware_rk4,
    force_budget,
)
from .profile_table import ProfileSupportError


@dataclass(frozen=True)
class CandidateFDMOuterOrbitResult:
    status: str
    final_state: DualNucleusState
    accepted_states: tuple[DualNucleusState, ...]
    used_source_sha256: tuple[str, ...]
    reason: str


def _time_step(
    state: DualNucleusState,
    model: KpcInspiralModel,
    config: KpcIntegrationConfig,
) -> float:
    radius = state.radius_pc
    speed = float(np.linalg.norm(state.velocity_pc_myr))
    enclosed = float(model.host_potential.enclosed_mass(radius))
    dynamical = math.sqrt(
        radius**3 / (G_INTERNAL * (enclosed + model.secondary_bh_mass_msun))
    )
    crossing = math.inf if speed == 0.0 else radius / speed
    return min(
        config.maximum_step_myr,
        config.timestep_fraction * dynamical,
        config.timestep_fraction * crossing,
        config.maximum_time_myr - state.elapsed_myr,
    )


def _project(
    state: DualNucleusState,
    model: KpcInspiralModel,
    family: FDMOuterResponseFamily,
    radial_support_pc: tuple[float, float],
):
    return project_fdm_response_family_in_host(
        family=family,
        host_potential=model.host_potential,
        position_pc=state.position_pc,
        velocity_pc_myr=state.velocity_pc_myr,
        secondary_bh_mass_msun=model.secondary_bh_mass_msun,
        radial_support_pc=radial_support_pc,
    )


def integrate_candidate_fdm_outer_orbit(
    *,
    initial_state: DualNucleusState,
    model: KpcInspiralModel,
    config: KpcIntegrationConfig,
    response_family: FDMOuterResponseFamily,
    closure_by_source_sha256: Mapping[str, FDMOuterHaloClosure],
    radial_support_pc: tuple[float, float],
    random_seed: int,
    step_budget: int | None = None,
) -> CandidateFDMOuterOrbitResult:
    """Evolve a bounded *candidate* outer orbit with exact step-index restart.

    One deterministic half step, one midpoint FDM velocity kick, and another
    deterministic half step form each accepted update.  A failed midpoint or
    endpoint applicability check discards the whole proposed step.  This
    path intentionally has no ``DelaySegment`` or hard-binary handoff.
    """

    if model.fdm_background is not None:
        raise ValueError("candidate FDM wake cannot be combined with analytic FDM drag")
    if step_budget is not None and (type(step_budget) is not int or step_budget < 1):
        raise ValueError("candidate step budget must be a positive integer")
    if initial_state.elapsed_myr > config.maximum_time_myr:
        raise ValueError("candidate initial state lies beyond maximum_time_myr")
    if type(random_seed) is not int or not 0 <= random_seed <= 0xFFFFFFFF:
        raise ValueError("candidate random seed must be a uint32 integer")

    state = initial_state
    accepted = [state]
    sources: list[str] = []

    def result(status: str, reason: str) -> CandidateFDMOuterOrbitResult:
        return CandidateFDMOuterOrbitResult(
            status, state, tuple(accepted), tuple(dict.fromkeys(sources)), reason
        )

    steps_this_call = 0
    while state.completed_steps < config.maximum_steps:
        if state.radius_pc <= config.target_radius_pc:
            return result(
                "candidate_geometric_target",
                "geometric target reached in a numerical candidate; no physical delay inferred",
            )
        if state.elapsed_myr >= config.maximum_time_myr:
            return result("censored", "candidate orbit exhausted available time")
        if step_budget is not None and steps_this_call >= step_budget:
            return result("checkpoint", "bounded step budget reached; resume from final_state")
        try:
            # Profile/force evaluation at the accepted point must succeed
            # before a new tentative step can be attempted.
            force_budget(state, model)
            time_step = _time_step(state, model, config)
            if not math.isfinite(time_step) or time_step <= 0.0:
                return result("invalid", "candidate adaptive time step is invalid")
            start_query = _project(state, model, response_family, radial_support_pc)
            if start_query.status == "censored":
                return result("censored", f"candidate start unsupported: {start_query.reason}")
            assert start_query.projected_response is not None
            start_check = sample_candidate_fdm_velocity_kick(
                start_query.projected_response,
                time_step_myr=time_step,
                random_seed=random_seed,
                completed_steps=state.completed_steps,
                closure_by_source_sha256=closure_by_source_sha256,
            )
            if start_check.status == "censored":
                return result("censored", f"candidate start kick unsupported: {start_check.reason}")
            midpoint = _advance_phase_aware_rk4(state, model, 0.5 * time_step)
            midpoint_query = _project(midpoint, model, response_family, radial_support_pc)
            if midpoint_query.status == "censored":
                return result("censored", f"candidate midpoint unsupported: {midpoint_query.reason}")
            assert midpoint_query.projected_response is not None
            kick = sample_candidate_fdm_velocity_kick(
                midpoint_query.projected_response,
                time_step_myr=time_step,
                random_seed=random_seed,
                completed_steps=state.completed_steps,
                closure_by_source_sha256=closure_by_source_sha256,
            )
            if kick.status == "censored":
                return result("censored", f"candidate FDM kick unsupported: {kick.reason}")
            assert kick.velocity_increment_pc_myr is not None
            kicked_midpoint = DualNucleusState(
                elapsed_myr=midpoint.elapsed_myr,
                position_pc=midpoint.position_pc,
                velocity_pc_myr=midpoint.velocity_pc_myr + kick.velocity_increment_pc_myr,
                envelope_truncation_radius_pc=midpoint.envelope_truncation_radius_pc,
                completed_steps=state.completed_steps,
            )
            tentative = _advance_phase_aware_rk4(kicked_midpoint, model, 0.5 * time_step)
            tentative = DualNucleusState(
                elapsed_myr=tentative.elapsed_myr,
                position_pc=tentative.position_pc,
                velocity_pc_myr=tentative.velocity_pc_myr,
                envelope_truncation_radius_pc=tentative.envelope_truncation_radius_pc,
                completed_steps=state.completed_steps + 1,
            )
            endpoint_query = _project(tentative, model, response_family, radial_support_pc)
            if endpoint_query.status == "censored":
                return result("censored", f"candidate endpoint unsupported: {endpoint_query.reason}")
            assert endpoint_query.projected_response is not None
            # Reuse the applicability check at the endpoint, but discard its
            # deterministic keyed draw; only the midpoint kick enters motion.
            endpoint_check = sample_candidate_fdm_velocity_kick(
                endpoint_query.projected_response,
                time_step_myr=time_step,
                random_seed=random_seed,
                completed_steps=state.completed_steps,
                closure_by_source_sha256=closure_by_source_sha256,
            )
            if endpoint_check.status == "censored":
                return result("censored", f"candidate endpoint kick unsupported: {endpoint_check.reason}")
            force_budget(tentative, model)
        except ProfileSupportError as error:
            return result("censored", f"candidate RK stage outside measured profile: {error}")
        except ValueError as error:
            return result("invalid", f"candidate orbit left its valid domain: {error}")
        state = tentative
        accepted.append(state)
        sources.extend(start_query.projected_response.source_sha256)
        sources.extend(kick.source_sha256)
        sources.extend(endpoint_query.projected_response.source_sha256)
        steps_this_call += 1
    return result("censored", "candidate orbit exhausted maximum step count")
