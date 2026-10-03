"""Phase and a--e diagnostics for a numerical outer-FDM candidate orbit.

These labels reuse the static-host geometric/binding tests but do not turn a
candidate stochastic trajectory into an accepted physical delay or handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .fdm_candidate_orbit import (
    CandidateFDMOuterOrbitResult, integrate_candidate_fdm_outer_orbit,
)
from .fdm_outer_halo import FDMOuterHaloClosure
from .fdm_outer_response_family import FDMOuterResponseFamily
from .fdm_response_sources import VerifiedFDMResponseSourceBundle
from .kpc_inspiral import (
    DualNucleusState,
    KpcIntegrationConfig,
    KpcInspiralModel,
    KpcToHardConfig,
    KpcToHardState,
    _advance_physical_phase,
    _replace_dynamics,
    _selected_hard_binary_radius_pc,
    _static_host_scope_error,
    _terminal_phase_state,
)
from .kpc_to_pc import InspiralPhase
from .orbital_exchange import keplerian_elements_from_relative_state


@dataclass(frozen=True)
class CandidateBinaryElementSample:
    elapsed_myr: float
    separation_pc: float
    specific_two_body_energy_pc2_myr2: float
    specific_angular_momentum_pc2_myr: float
    semimajor_axis_pc: float | None
    eccentricity: float | None


@dataclass(frozen=True)
class CandidateFDMPhaseAssessment:
    status: str
    final_state: KpcToHardState
    element_samples: tuple[CandidateBinaryElementSample, ...]
    reason: str


@dataclass(frozen=True)
class CandidateFDMKpcToHardResult:
    status: str
    final_state: KpcToHardState
    element_samples: tuple[CandidateBinaryElementSample, ...]
    used_source_sha256: tuple[str, ...]
    reason: str
    source_manifest_sha256: str | None = None


def _same_dynamics(first: DualNucleusState, second: DualNucleusState) -> bool:
    return (
        first.elapsed_myr == second.elapsed_myr
        and first.completed_steps == second.completed_steps
        and first.envelope_truncation_radius_pc == second.envelope_truncation_radius_pc
        and np.array_equal(first.position_pc, second.position_pc)
        and np.array_equal(first.velocity_pc_myr, second.velocity_pc_myr)
    )


def _sample(
    dynamics: DualNucleusState,
    model: KpcInspiralModel,
    config: KpcToHardConfig,
) -> CandidateBinaryElementSample:
    elements = keplerian_elements_from_relative_state(
        total_mass=config.primary_bh_mass_msun + model.secondary_bh_mass_msun,
        displacement=dynamics.position_pc,
        relative_velocity=dynamics.velocity_pc_myr,
    )
    inside = dynamics.radius_pc <= config.common_nucleus_radius_pc
    bound = inside and elements.semimajor_axis is not None and 0.0 <= elements.eccentricity < 1.0
    return CandidateBinaryElementSample(
        elapsed_myr=dynamics.elapsed_myr,
        separation_pc=elements.separation,
        specific_two_body_energy_pc2_myr2=elements.specific_energy,
        specific_angular_momentum_pc2_myr=float(
            np.linalg.norm(elements.specific_angular_momentum)
        ),
        semimajor_axis_pc=elements.semimajor_axis if bound else None,
        eccentricity=elements.eccentricity if bound else None,
    )


def assess_candidate_fdm_outer_phases(
    *,
    orbit: CandidateFDMOuterOrbitResult,
    initial_state: KpcToHardState,
    model: KpcInspiralModel,
    config: KpcToHardConfig,
) -> CandidateFDMPhaseAssessment:
    """Classify accepted states only; preserve MULTIPLE and censor boundaries.

    The returned semimajor axis and eccentricity are instantaneous two-body
    diagnostics inside the common nucleus.  Host-orbit eccentricity used by
    the outer response remains a different quantity.  No ``DelaySegment`` or
    binary-evolution initial condition is emitted.
    """

    if orbit.status not in {
        "checkpoint", "candidate_geometric_target", "censored", "invalid"
    }:
        raise ValueError("unsupported candidate orbit result status")
    if not orbit.accepted_states or not _same_dynamics(
        orbit.accepted_states[0], initial_state.dynamical_state
    ) or not _same_dynamics(orbit.accepted_states[-1], orbit.final_state):
        raise ValueError("candidate orbit and phase assessment states are not aligned")
    for before, after in zip(orbit.accepted_states, orbit.accepted_states[1:]):
        if (
            after.completed_steps != before.completed_steps + 1
            or after.elapsed_myr <= before.elapsed_myr
        ):
            raise ValueError("candidate orbit accepted-state sequence is invalid")
    if initial_state.inspiral_state.phase is InspiralPhase.MULTIPLE:
        return CandidateFDMPhaseAssessment(
            "multiple", initial_state, (),
            "MULTIPLE capture is preserved; no binary phase or delay inferred",
        )
    if initial_state.inspiral_state.terminal:
        raise ValueError("terminal candidate phase cannot consume a binary orbit")
    if initial_state.inspiral_state.phase not in {
        InspiralPhase.DUAL_NUCLEUS, InspiralPhase.COMMON_NUCLEUS_UNBOUND,
        InspiralPhase.BOUND_BINARY, InspiralPhase.HARD_BINARY,
    }:
        raise ValueError("candidate phase must be classified at the starting boundary")
    scope_error = _static_host_scope_error(model, config)
    try:
        _selected_hard_binary_radius_pc(model, config)
    except ValueError as error:
        scope_error = str(error)
    if scope_error is not None:
        raise ValueError(scope_error)

    state = initial_state
    samples: list[CandidateBinaryElementSample] = []
    for dynamics in orbit.accepted_states:
        if state.inspiral_state.phase is InspiralPhase.HARD_BINARY:
            break
        if dynamics is not orbit.accepted_states[0]:
            state = _replace_dynamics(state, dynamics)
            state, terminal_status = _advance_physical_phase(state, model, config)
            if terminal_status == "outside":
                samples.append(_sample(dynamics, model, config))
                return CandidateFDMPhaseAssessment(
                    "censored", state, tuple(samples), state.inspiral_state.reason
                )
        samples.append(_sample(dynamics, model, config))
    if state.inspiral_state.phase is InspiralPhase.HARD_BINARY:
        return CandidateFDMPhaseAssessment(
            "candidate_hard_boundary", state, tuple(samples),
            "hard boundary reached only in an unvalidated outer-FDM candidate",
        )
    if orbit.status in {"censored", "invalid"}:
        target = (
            InspiralPhase.CENSORED if orbit.status == "censored"
            else InspiralPhase.INVALID
        )
        state = _terminal_phase_state(state, target, orbit.reason)
        return CandidateFDMPhaseAssessment(
            orbit.status, state, tuple(samples), orbit.reason
        )
    return CandidateFDMPhaseAssessment(
        orbit.status, state, tuple(samples),
        "phase/elements remain numerical candidates; no physical delay inferred",
    )


def integrate_candidate_fdm_to_hard_boundary(
    *,
    initial_state: KpcToHardState,
    model: KpcInspiralModel,
    outer_config: KpcIntegrationConfig,
    phase_config: KpcToHardConfig,
    response_family: FDMOuterResponseFamily,
    closure_by_source_sha256: Mapping[str, FDMOuterHaloClosure],
    radial_support_pc: tuple[float, float],
    random_seed: int,
    step_budget: int | None = None,
) -> CandidateFDMKpcToHardResult:
    """Stop at the first candidate hard boundary, censor, or outer seam.

    Each accepted stochastic step is classified immediately; no subsequent
    outer-response step is taken after a hard-binary transition.  This is a
    numerical boundary finder only and deliberately returns no physical
    ``DelaySegment`` or binary-evolution initial state.
    """

    if step_budget is not None and (type(step_budget) is not int or step_budget < 1):
        raise ValueError("candidate phase step budget must be a positive integer")
    if outer_config.maximum_time_myr != phase_config.maximum_time_myr:
        raise ValueError("candidate outer and phase time budgets must agree")
    if initial_state.inspiral_state.phase is InspiralPhase.MULTIPLE:
        return CandidateFDMKpcToHardResult(
            "multiple", initial_state, (), (),
            "MULTIPLE capture is preserved; no binary integration attempted",
        )
    if initial_state.inspiral_state.phase is InspiralPhase.HARD_BINARY:
        return CandidateFDMKpcToHardResult(
            "candidate_hard_boundary", initial_state, (), (),
            "initial state is already at a candidate hard boundary",
        )

    state = initial_state
    samples: list[CandidateBinaryElementSample] = []
    sources: list[str] = []
    steps = 0

    def result(status: str, reason: str) -> CandidateFDMKpcToHardResult:
        return CandidateFDMKpcToHardResult(
            status, state, tuple(samples), tuple(dict.fromkeys(sources)), reason
        )

    while True:
        if step_budget is not None and steps >= step_budget:
            return result("checkpoint", "bounded candidate phase step budget reached")
        orbit = integrate_candidate_fdm_outer_orbit(
            initial_state=state.dynamical_state,
            model=model, config=outer_config, response_family=response_family,
            closure_by_source_sha256=closure_by_source_sha256,
            radial_support_pc=radial_support_pc, random_seed=random_seed,
            step_budget=1,
        )
        assessment = assess_candidate_fdm_outer_phases(
            orbit=orbit, initial_state=state, model=model, config=phase_config,
        )
        if not samples:
            samples.extend(assessment.element_samples)
        else:
            samples.extend(assessment.element_samples[1:])
        sources.extend(orbit.used_source_sha256)
        state = assessment.final_state
        steps += state.dynamical_state.completed_steps - orbit.accepted_states[0].completed_steps
        if assessment.status != "checkpoint":
            return result(assessment.status, assessment.reason)


def integrate_verified_candidate_fdm_to_hard_boundary(
    *,
    source_bundle: VerifiedFDMResponseSourceBundle,
    initial_state: KpcToHardState,
    model: KpcInspiralModel,
    outer_config: KpcIntegrationConfig,
    phase_config: KpcToHardConfig,
    radial_support_pc: tuple[float, float],
    random_seed: int,
    step_budget: int | None = None,
) -> CandidateFDMKpcToHardResult:
    """Run the numerical candidate only while all named source bytes match.

    File identity is checked both before and after integration; it is not a
    substitute for convergence, force conservation, or phase-replica audits.
    """

    source_bundle.verify_current_files()
    result = integrate_candidate_fdm_to_hard_boundary(
        initial_state=initial_state,
        model=model,
        outer_config=outer_config,
        phase_config=phase_config,
        response_family=source_bundle.response_family,
        closure_by_source_sha256=source_bundle.closure_by_source_sha256,
        radial_support_pc=radial_support_pc,
        random_seed=random_seed,
        step_budget=step_budget,
    )
    source_bundle.verify_current_files()
    return replace(result, source_manifest_sha256=source_bundle.manifest_sha256)
