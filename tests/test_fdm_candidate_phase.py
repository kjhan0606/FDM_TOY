from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import fdm_smbh_delay.fdm_candidate_checkpoint as candidate_checkpoint
from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.fdm_candidate_orbit import CandidateFDMOuterOrbitResult
from fdm_smbh_delay.fdm_candidate_orbit import integrate_candidate_fdm_outer_orbit
from fdm_smbh_delay.fdm_candidate_phase import (
    assess_candidate_fdm_outer_phases,
    integrate_candidate_fdm_to_hard_boundary,
)
from fdm_smbh_delay.fdm_outer_halo import FDMOuterHaloClosure
from fdm_smbh_delay.fdm_outer_response import FDMOuterResponseTable
from fdm_smbh_delay.fdm_outer_response_family import FDMOuterResponseFamily
from fdm_smbh_delay.galaxy_environment import CompositePotential
from fdm_smbh_delay.kpc_inspiral import (
    DualNucleusState, KpcInspiralModel, KpcToHardConfig,
    KpcIntegrationConfig, KpcToHardState, initial_kpc_to_hard_state,
)
from fdm_smbh_delay.kpc_to_pc import InspiralPhase, transition_state


def _case():
    model = KpcInspiralModel(
        host_potential=CompositePotential((), central_point_mass_msun=1.0e8),
        secondary_bh_mass_msun=2.0e7,
    )
    config = KpcToHardConfig(
        primary_bh_mass_msun=1.0e8, common_nucleus_radius_pc=100.0,
        sigma_pc_myr=100.0, maximum_time_myr=1.0,
        maximum_step_myr=0.1, hard_binary_radius_pc=2.0,
    )
    states = (
        DualNucleusState(0.0, np.array([200.0, 0.0, 0.0]), np.array([0.0, 40.0, 0.0]), None, 0),
        DualNucleusState(0.1, np.array([80.0, 0.0, 0.0]), np.array([0.0, 100.0, 0.0]), None, 1),
        DualNucleusState(0.2, np.array([2.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0]), None, 2),
    )
    initial = initial_kpc_to_hard_state(
        event_uid="capture-1", dynamical_state=states[0], model=model, config=config,
    )
    return model, config, states, initial


def _orbit(states, status="candidate_geometric_target"):
    return CandidateFDMOuterOrbitResult(
        status=status, final_state=states[-1], accepted_states=tuple(states),
        used_source_sha256=("a" * 64,), reason="synthetic candidate boundary",
    )


def _source(upper_radius: float = 250.0):
    table = FDMOuterResponseTable(
        radii_pc=np.array([10.0, upper_radius]),
        drift_acceleration_pc_myr2=np.zeros((2, 3)),
        diffusion_tensor_pc2_myr3=np.zeros((2, 3, 3)),
        response_status="calibrated", component_frame="orbital_rtn",
        diffusion_convention="velocity_covariance_rate",
    )
    family = FDMOuterResponseFamily(
        mass_ratios_q=(0.2,), eccentricities=(0.0, 0.9),
        tables=((table, table),), source_sha256=(("a" * 64, "a" * 64),),
    )
    closure = FDMOuterHaloClosure(
        radii_pc=np.array([10.0, upper_radius]),
        mass_current_msun_pc2_myr=np.zeros((2, 3)),
        coherence_time_myr=np.array([1.0e-6, 1.0e-6]),
        de_broglie_wavelength_pc=np.ones(2),
        velocity_diffusion_pc2_myr3=np.zeros(2),
        density_gradient_scale_pc=np.ones(2),
        closure_status="calibrated",
    )
    return family, {"a" * 64: closure}


def test_candidate_phase_tracks_bound_hard_and_joint_a_e_without_delay() -> None:
    model, config, states, initial = _case()
    assessment = assess_candidate_fdm_outer_phases(
        orbit=_orbit(states), initial_state=initial, model=model, config=config,
    )
    assert assessment.status == "candidate_hard_boundary"
    assert [item.phase for item in assessment.final_state.transition_history] == [
        InspiralPhase.NUMERICAL_CAPTURE, InspiralPhase.DUAL_NUCLEUS,
        InspiralPhase.BOUND_BINARY, InspiralPhase.HARD_BINARY,
    ]
    assert len(assessment.element_samples) == 3
    assert assessment.element_samples[0].semimajor_axis_pc is None
    for item in assessment.element_samples[1:]:
        assert item.semimajor_axis_pc is not None and item.semimajor_axis_pc > 0.0
        assert item.eccentricity is not None and 0.0 <= item.eccentricity < 1.0
    assert assessment.element_samples[2].semimajor_axis_pc <= config.hard_binary_radius_pc
    assert not hasattr(assessment, "delay_segment")


def test_candidate_phase_history_is_identical_across_step_budget_restart() -> None:
    model, config, states, initial = _case()
    full = assess_candidate_fdm_outer_phases(
        orbit=_orbit(states), initial_state=initial, model=model, config=config,
    )
    first = assess_candidate_fdm_outer_phases(
        orbit=_orbit(states[:2], "checkpoint"),
        initial_state=initial, model=model, config=config,
    )
    resumed = assess_candidate_fdm_outer_phases(
        orbit=_orbit(states[1:]),
        initial_state=first.final_state, model=model, config=config,
    )
    assert first.status == "checkpoint"
    assert resumed.status == full.status
    assert resumed.final_state.transition_history == full.final_state.transition_history
    assert resumed.final_state.dynamical_state is states[-1]


def test_candidate_phase_preserves_multiple_and_censors_missing_interval() -> None:
    model, config, states, initial = _case()
    multiple = transition_state(
        initial.inspiral_state, InspiralPhase.MULTIPLE,
        elapsed_myr=0.0, reason="third SMBH remains in the capture group",
    )
    multiple_state = KpcToHardState(
        initial.dynamical_state, multiple, initial.transition_history + (multiple,),
    )
    multiple_assessment = assess_candidate_fdm_outer_phases(
        orbit=_orbit(states), initial_state=multiple_state,
        model=model, config=config,
    )
    assert multiple_assessment.status == "multiple"
    assert multiple_assessment.final_state is multiple_state
    assert multiple_assessment.element_samples == ()

    censored = assess_candidate_fdm_outer_phases(
        orbit=_orbit(states[:1], "censored"),
        initial_state=initial, model=model, config=config,
    )
    assert censored.status == "censored"
    assert censored.final_state.inspiral_state.phase is InspiralPhase.CENSORED
    assert censored.final_state.dynamical_state is states[0]


def test_candidate_phase_refuses_unmatched_state_and_host_mass() -> None:
    model, config, states, initial = _case()
    with pytest.raises(ValueError, match="not aligned"):
        assess_candidate_fdm_outer_phases(
            orbit=_orbit(states[1:]), initial_state=initial,
            model=model, config=config,
        )
    wrong_host = KpcInspiralModel(
        host_potential=CompositePotential((), central_point_mass_msun=2.0e8),
        secondary_bh_mass_msun=2.0e7,
    )
    with pytest.raises(ValueError, match="central_point_mass_msun"):
        assess_candidate_fdm_outer_phases(
            orbit=_orbit(states), initial_state=initial,
            model=wrong_host, config=config,
        )


def test_integrated_candidate_crosses_common_nucleus_but_not_hard_boundary() -> None:
    primary, secondary = 1.0e8, 2.0e7
    mu = G_INTERNAL * (primary + secondary)
    axis, eccentricity, radius = 100.0, 0.5, 105.0
    angular = np.sqrt(mu * axis * (1.0 - eccentricity**2))
    tangential = angular / radius
    radial = -np.sqrt(mu * (2.0 / radius - 1.0 / axis) - tangential**2)
    model = KpcInspiralModel(
        host_potential=CompositePotential((), central_point_mass_msun=primary),
        secondary_bh_mass_msun=secondary,
    )
    initial = DualNucleusState(
        0.0, np.array([radius, 0.0, 0.0]),
        np.array([radial, tangential, 0.0]), None, 0,
    )
    family, closures = _source()
    outer_config = KpcIntegrationConfig(95.0, 1.0, 0.01, maximum_steps=200)
    orbit = integrate_candidate_fdm_outer_orbit(
        initial_state=initial, model=model,
        config=outer_config,
        response_family=family,
        closure_by_source_sha256=closures,
        radial_support_pc=(5.0, 400.0), random_seed=7,
    )
    assert orbit.status == "candidate_geometric_target"
    phase_config = KpcToHardConfig(
        primary_bh_mass_msun=primary, common_nucleus_radius_pc=100.0,
        sigma_pc_myr=100.0, maximum_time_myr=1.0,
        maximum_step_myr=0.01, hard_binary_radius_pc=2.0,
    )
    phase_initial = initial_kpc_to_hard_state(
        event_uid="capture-integrated", dynamical_state=initial,
        model=model, config=phase_config,
    )
    assessment = assess_candidate_fdm_outer_phases(
        orbit=orbit, initial_state=phase_initial,
        model=model, config=phase_config,
    )
    assert assessment.status == "candidate_geometric_target"
    assert assessment.final_state.inspiral_state.phase is InspiralPhase.BOUND_BINARY
    assert assessment.element_samples[-1].semimajor_axis_pc == pytest.approx(axis, rel=1.0e-8)
    assert assessment.element_samples[-1].eccentricity == pytest.approx(eccentricity, rel=1.0e-8)
    coupled = integrate_candidate_fdm_to_hard_boundary(
        initial_state=phase_initial, model=model, outer_config=outer_config,
        phase_config=phase_config, response_family=family,
        closure_by_source_sha256=closures,
        radial_support_pc=(5.0, 400.0), random_seed=7,
    )
    assert coupled.status == "candidate_geometric_target"
    assert coupled.final_state.inspiral_state.phase is InspiralPhase.BOUND_BINARY
    assert np.array_equal(coupled.final_state.dynamical_state.position_pc, orbit.final_state.position_pc)
    assert coupled.final_state.transition_history == assessment.final_state.transition_history


def test_coupled_candidate_phase_driver_restarts_without_losing_transitions(tmp_path) -> None:
    model, phase_config, states, initial = _case()
    family, closures = _source()
    kwargs = dict(
        model=model,
        outer_config=KpcIntegrationConfig(50.0, 1.0, 0.005, maximum_steps=100),
        phase_config=phase_config,
        response_family=family, closure_by_source_sha256=closures,
        radial_support_pc=(5.0, 400.0), random_seed=7,
    )
    full = integrate_candidate_fdm_to_hard_boundary(
        initial_state=initial, step_budget=4, **kwargs,
    )
    first = integrate_candidate_fdm_to_hard_boundary(
        initial_state=initial, step_budget=2, **kwargs,
    )
    resumed = integrate_candidate_fdm_to_hard_boundary(
        initial_state=first.final_state, step_budget=2, **kwargs,
    )
    assert full.status == first.status == resumed.status == "checkpoint"
    assert full.final_state.transition_history == resumed.final_state.transition_history
    assert np.array_equal(
        full.final_state.dynamical_state.position_pc,
        resumed.final_state.dynamical_state.position_pc,
    )
    assert full.used_source_sha256 == resumed.used_source_sha256 == ("a" * 64,)
    assert len(full.element_samples) == 5
    assert not hasattr(full, "delay_segment")

    path = tmp_path / "candidate_phase.json"
    candidate_checkpoint.write_candidate_fdm_phase_checkpoint(
        path, first.final_state, **kwargs,
    )
    restored = candidate_checkpoint.read_candidate_fdm_phase_checkpoint(path, **kwargs)
    disk_resumed = integrate_candidate_fdm_to_hard_boundary(
        initial_state=restored, step_budget=2, **kwargs,
    )
    assert disk_resumed.final_state.transition_history == full.final_state.transition_history
    assert np.array_equal(
        disk_resumed.final_state.dynamical_state.velocity_pc_myr,
        full.final_state.dynamical_state.velocity_pc_myr,
    )
    with pytest.raises(ValueError, match="physics fingerprint"):
        candidate_checkpoint.read_candidate_fdm_phase_checkpoint(
            path, **{**kwargs, "phase_config": replace(phase_config, sigma_pc_myr=101.0)},
        )
    raw = json.loads(path.read_text())
    raw["state"]["transition_history"][-1]["reason"] = "tampered phase"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="state checksum"):
        candidate_checkpoint.read_candidate_fdm_phase_checkpoint(path, **kwargs)

    already_hard = initial_kpc_to_hard_state(
        event_uid="hard-initial", dynamical_state=states[-1],
        model=model, config=phase_config,
    )
    stopped = integrate_candidate_fdm_to_hard_boundary(
        initial_state=already_hard, **kwargs,
    )
    assert stopped.status == "candidate_hard_boundary"
    assert stopped.final_state is already_hard
    assert stopped.used_source_sha256 == ()

    multiple_phase = transition_state(
        initial.inspiral_state, InspiralPhase.MULTIPLE,
        elapsed_myr=0.0, reason="third SMBH remains in the capture group",
    )
    multiple_state = KpcToHardState(
        initial.dynamical_state, multiple_phase,
        initial.transition_history + (multiple_phase,),
    )
    multiple_result = integrate_candidate_fdm_to_hard_boundary(
        initial_state=multiple_state,
        **{**kwargs, "closure_by_source_sha256": {}},
    )
    assert multiple_result.status == "multiple"
    assert multiple_result.final_state is multiple_state
    assert multiple_result.used_source_sha256 == ()
    with pytest.raises(ValueError, match="time budgets must agree"):
        integrate_candidate_fdm_to_hard_boundary(
            initial_state=initial,
            **{**kwargs, "phase_config": replace(phase_config, maximum_time_myr=2.0)},
        )
