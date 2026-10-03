from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import fdm_smbh_delay.kpc_inspiral as kpc_inspiral
from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.galaxy_environment import (
    CompositePotential,
    DehnenProfile,
    FDMBackground,
    NuclearEnvelope,
    StellarBackground,
)
from fdm_smbh_delay.soliton import SchiveSoliton
from fdm_smbh_delay.profile_table import TabulatedSphericalProfile
from fdm_smbh_delay.kpc_inspiral import (
    KpcInspiralModel,
    KpcIntegrationConfig,
    KpcToHardConfig,
    KpcToHardState,
    StellarFrictionConfig,
    DualNucleusState,
    force_budget,
    initial_dual_nucleus_state,
    initial_kpc_to_hard_state,
    integrate_dual_nucleus,
    integrate_dual_nucleus_to_hard,
    tidal_truncation_radius_pc,
)
from fdm_smbh_delay.kpc_to_pc import InspiralPhase


def _point_mass_model(*, envelope=None) -> KpcInspiralModel:
    return KpcInspiralModel(
        host_potential=CompositePotential((), central_point_mass_msun=1.0e10),
        secondary_bh_mass_msun=1.0e8,
        nuclear_envelope=envelope,
    )


def _bounded_profile_model() -> KpcInspiralModel:
    profile = TabulatedSphericalProfile(
        radii_pc=np.array([10.0, 20.0, 30.0]),
        density_msun_pc3=np.array([100.0, 50.0, 20.0]),
        enclosed_mass_msun=np.array([1.0e6, 2.0e6, 3.0e6]),
        potential_pc2_myr2=np.array([-100.0, -80.0, -60.0]),
    )
    return KpcInspiralModel(
        host_potential=CompositePotential((profile,), central_point_mass_msun=1.0e10),
        secondary_bh_mass_msun=1.0e8,
    )


def test_phase_aware_initial_state_outside_profile_is_censored() -> None:
    model = _bounded_profile_model()
    config = KpcToHardConfig(
        primary_bh_mass_msun=1.0e10,
        common_nucleus_radius_pc=15.0,
        sigma_pc_myr=100.0,
        maximum_time_myr=1.0,
        maximum_step_myr=0.1,
        hard_binary_radius_pc=1.0,
    )
    dynamics = DualNucleusState(0.0, np.array([35.0, 0.0, 0.0]), np.zeros(3), None)
    initial = initial_kpc_to_hard_state(
        event_uid="profile-outside", dynamical_state=dynamics,
        model=model, config=config,
    )
    result = integrate_dual_nucleus_to_hard(initial_state=initial, model=model, config=config)
    assert result.status == "outside"
    assert result.final_state.inspiral_state.phase is InspiralPhase.CENSORED
    assert result.final_state.dynamical_state.completed_steps == 0
    assert result.samples == ()
    assert result.delay_segment.status == "timeout"


def test_phase_aware_trial_outside_profile_preserves_last_valid_state(monkeypatch) -> None:
    model = _bounded_profile_model()
    config = KpcToHardConfig(
        primary_bh_mass_msun=1.0e10,
        common_nucleus_radius_pc=15.0,
        sigma_pc_myr=100.0,
        maximum_time_myr=1.0,
        maximum_step_myr=0.1,
        hard_binary_radius_pc=1.0,
    )
    dynamics = DualNucleusState(0.0, np.array([25.0, 0.0, 0.0]), np.zeros(3), None)
    initial = initial_kpc_to_hard_state(
        event_uid="profile-trial", dynamical_state=dynamics,
        model=model, config=config,
    )

    def unsupported_trial(state, _model, time_step):
        return DualNucleusState(
            state.elapsed_myr + time_step,
            np.array([31.0, 0.0, 0.0]), state.velocity_pc_myr,
            state.envelope_truncation_radius_pc, state.completed_steps + 1,
        )

    monkeypatch.setattr(kpc_inspiral, "_advance_phase_aware_rk4", unsupported_trial)
    result = integrate_dual_nucleus_to_hard(initial_state=initial, model=model, config=config)
    assert result.status == "outside"
    assert result.final_state.inspiral_state.phase is InspiralPhase.CENSORED
    assert result.final_state.dynamical_state.completed_steps == 0
    assert result.final_state.dynamical_state.radius_pc == pytest.approx(25.0)
    assert len(result.samples) == 1
    assert result.delay_segment.status == "timeout"


def test_geometric_kpc_path_censors_profile_extrapolation() -> None:
    model = _bounded_profile_model()
    initial = DualNucleusState(0.0, np.array([35.0, 0.0, 0.0]), np.zeros(3), None)
    result = integrate_dual_nucleus(
        initial_state=initial, model=model,
        config=KpcIntegrationConfig(1.0, 1.0, 0.1),
    )
    assert result.status == "outside"
    assert result.delay_segment.status == "timeout"
    assert result.samples == ()


def test_static_kpc_model_refuses_unprovided_live_fdm_wake_force() -> None:
    live = FDMBackground(
        soliton=SchiveSoliton.from_mass(1.0e9, 100.0, "total_profile"),
        particle_mass_ev=1.0e-21,
        alpha_df=0.341,
        bulk_velocity_pc_myr=np.zeros(3),
        mode="live_resolved",
    )
    with pytest.raises(ValueError, match="no source-bound live FDM wake force"):
        KpcInspiralModel(
            host_potential=CompositePotential((), central_point_mass_msun=1.0e10),
            secondary_bh_mass_msun=1.0e8,
            fdm_background=live,
        )


def _circular_state(model: KpcInspiralModel, radius: float = 100.0):
    speed = np.sqrt(G_INTERNAL * model.host_potential.enclosed_mass(radius) / radius)
    return initial_dual_nucleus_state(
        position_pc=np.array([radius, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, speed, 0.0]),
        model=model,
    )


def test_conservative_orbit_closes_energy_without_drag() -> None:
    model = _point_mass_model()
    state = _circular_state(model)
    period = 2.0 * np.pi * state.radius_pc / np.linalg.norm(state.velocity_pc_myr)
    result = integrate_dual_nucleus(
        initial_state=state,
        model=model,
        config=KpcIntegrationConfig(
            target_radius_pc=1.0,
            maximum_time_myr=period,
            maximum_step_myr=period / 100.0,
            timestep_fraction=0.02,
            sample_interval_steps=10,
        ),
    )
    assert result.status == "timeout"
    initial_energy = result.samples[0].specific_orbital_energy_pc2_myr2
    final_energy = result.samples[-1].specific_orbital_energy_pc2_myr2
    assert (final_energy - initial_energy) / initial_energy == pytest.approx(
        0.0, abs=2.0e-8
    )
    assert all(sample.drag_work_rate_pc2_myr3 == 0.0 for sample in result.samples)


def test_tidal_stripping_is_monotonic() -> None:
    envelope = NuclearEnvelope(5.0e8, 10.0, 100.0)
    model = _point_mass_model(envelope=envelope)
    state = _circular_state(model, radius=100.0)
    truncation = tidal_truncation_radius_pc(state, model)
    assert truncation is not None
    assert 0.0 < truncation <= envelope.initial_truncation_radius_pc
    assert envelope.mass_within(truncation) <= envelope.initial_bound_mass_msun


def test_bounded_resume_matches_uninterrupted_state() -> None:
    model = _point_mass_model()
    initial = _circular_state(model)
    period = 2.0 * np.pi * initial.radius_pc / np.linalg.norm(initial.velocity_pc_myr)
    config = KpcIntegrationConfig(
        target_radius_pc=1.0,
        maximum_time_myr=0.2 * period,
        maximum_step_myr=period / 100.0,
        timestep_fraction=0.02,
    )
    uninterrupted = integrate_dual_nucleus(
        initial_state=initial, model=model, config=config
    )
    partial = integrate_dual_nucleus(
        initial_state=initial, model=model, config=config, step_budget=7
    )
    assert partial.status == "checkpoint"
    assert partial.delay_segment.status == "missing"
    resumed = integrate_dual_nucleus(
        initial_state=partial.final_state, model=model, config=config
    )
    assert resumed.status == uninterrupted.status == "timeout"
    assert resumed.final_state.elapsed_myr == uninterrupted.final_state.elapsed_myr
    assert np.array_equal(
        resumed.final_state.position_pc, uninterrupted.final_state.position_pc
    )
    assert np.array_equal(
        resumed.final_state.velocity_pc_myr,
        uninterrupted.final_state.velocity_pc_myr,
    )


def test_core_stall_is_a_censored_physical_result() -> None:
    stars = DehnenProfile(1.0e8, 1000.0, 0.0)
    model = KpcInspiralModel(
        host_potential=CompositePotential((stars,), central_point_mass_msun=1.0e8),
        secondary_bh_mass_msun=1.0e8,
        stellar_background=StellarBackground(stars, 100.0, np.zeros(3)),
        stellar_friction=StellarFrictionConfig(
            minimum_impact_parameter_pc=0.1,
            core_stall_enclosed_mass_ratio=1.0,
        ),
    )
    state = initial_dual_nucleus_state(
        position_pc=np.array([10.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 100.0, 0.0]),
        model=model,
    )
    assert force_budget(state, model).stellar.stalled
    result = integrate_dual_nucleus(
        initial_state=state,
        model=model,
        config=KpcIntegrationConfig(1.0, 100.0, 0.1),
    )
    assert result.status == "stalled"
    assert result.delay_segment.status == "timeout"


def test_stellar_friction_reaches_common_nucleus_boundary() -> None:
    stars = DehnenProfile(1.0e11, 500.0, 1.0)
    host = CompositePotential((stars,), central_point_mass_msun=1.0e8)
    model = KpcInspiralModel(
        host_potential=host,
        secondary_bh_mass_msun=1.0e8,
        stellar_background=StellarBackground(stars, 100.0, np.zeros(3)),
        stellar_friction=StellarFrictionConfig(minimum_impact_parameter_pc=0.1),
    )
    radius = 100.0
    speed = np.sqrt(G_INTERNAL * host.enclosed_mass(radius) / radius)
    state = initial_dual_nucleus_state(
        position_pc=np.array([radius, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, speed, 0.0]),
        model=model,
    )
    result = integrate_dual_nucleus(
        initial_state=state,
        model=model,
        config=KpcIntegrationConfig(50.0, 2.0, 0.01),
    )
    assert result.status == "reached_common_nucleus"
    assert result.final_state.radius_pc <= 50.0
    assert result.samples[-1].elapsed_myr == result.final_state.elapsed_myr
    assert result.delay_segment.status == "complete"


def _phase_aware_model() -> KpcInspiralModel:
    return KpcInspiralModel(
        host_potential=CompositePotential((), central_point_mass_msun=1.0e8),
        secondary_bh_mass_msun=1.0e8,
    )


def _phase_aware_config() -> KpcToHardConfig:
    return KpcToHardConfig(
        primary_bh_mass_msun=1.0e8,
        common_nucleus_radius_pc=20.0,
        sigma_pc_myr=100.0,
        hard_binary_radius_pc=8.0,
        maximum_time_myr=1.0,
        maximum_step_myr=1.0e-3,
    )


def _tangential_velocity_for_axis(axis_pc: float, radius_pc: float = 10.0) -> np.ndarray:
    gravitational_mass = G_INTERNAL * 2.0e8
    speed_squared = gravitational_mass * (2.0 / radius_pc - 1.0 / axis_pc)
    return np.array([0.0, np.sqrt(speed_squared), 0.0])


def _inside_unbound_state(model: KpcInspiralModel) -> DualNucleusState:
    radius = 10.0
    escape_speed = np.sqrt(
        2.0 * G_INTERNAL * 2.0e8 / radius
    )
    return initial_dual_nucleus_state(
        position_pc=np.array([radius, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 1.01 * escape_speed, 0.0]),
        model=model,
    )


def test_phase_aware_timestep_uses_total_mutual_mass() -> None:
    model = _phase_aware_model()
    radius = 100.0
    state = initial_dual_nucleus_state(
        position_pc=np.array([radius, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 1.0e-6, 0.0]),
        model=model,
    )
    config = KpcToHardConfig(
        primary_bh_mass_msun=1.0e8,
        common_nucleus_radius_pc=20.0,
        sigma_pc_myr=100.0,
        hard_binary_radius_pc=1.0,
        maximum_time_myr=10.0,
        maximum_step_myr=10.0,
        timestep_fraction=0.02,
    )
    time_step = kpc_inspiral._phase_aware_adaptive_step(state, model, config)
    expected = config.timestep_fraction * np.sqrt(
        radius**3 / (G_INTERNAL * 2.0e8)
    )
    host_only = config.timestep_fraction * np.sqrt(
        radius**3 / (G_INTERNAL * 1.0e8)
    )
    assert time_step == pytest.approx(expected)
    assert time_step < host_only


def test_phase_aware_relative_orbit_uses_total_smbh_mass() -> None:
    model = _phase_aware_model()
    radius = 100.0
    speed = np.sqrt(G_INTERNAL * 2.0e8 / radius)
    dynamics = initial_dual_nucleus_state(
        position_pc=np.array([radius, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, speed, 0.0]),
        model=model,
    )
    period = 2.0 * np.pi * radius / speed
    config = KpcToHardConfig(
        primary_bh_mass_msun=1.0e8,
        common_nucleus_radius_pc=20.0,
        sigma_pc_myr=100.0,
        hard_binary_radius_pc=1.0,
        maximum_time_myr=period,
        maximum_step_myr=period / 100.0,
        timestep_fraction=0.02,
        sample_interval_steps=10,
    )
    initial = initial_kpc_to_hard_state(
        event_uid="capture-relative-orbit",
        dynamical_state=dynamics,
        model=model,
        config=config,
    )
    result = integrate_dual_nucleus_to_hard(
        initial_state=initial, model=model, config=config
    )
    assert result.status == "timeout"
    initial_energy = result.samples[0].specific_orbital_energy_pc2_myr2
    final_energy = result.samples[-1].specific_orbital_energy_pc2_myr2
    assert (final_energy - initial_energy) / initial_energy == pytest.approx(
        0.0, abs=2.0e-8
    )


def test_common_nucleus_entry_is_not_physical_completion(monkeypatch) -> None:
    model = _phase_aware_model()
    config = _phase_aware_config()
    initial = initial_kpc_to_hard_state(
        event_uid="capture-unbound",
        dynamical_state=_inside_unbound_state(model),
        model=model,
        config=config,
    )
    assert initial.inspiral_state.phase is InspiralPhase.COMMON_NUCLEUS_UNBOUND

    def remain_unbound(state, _model, time_step):
        return DualNucleusState(
            state.elapsed_myr + time_step,
            state.position_pc,
            state.velocity_pc_myr,
            state.envelope_truncation_radius_pc,
            state.completed_steps + 1,
        )

    monkeypatch.setattr(kpc_inspiral, "_advance_phase_aware_rk4", remain_unbound)
    result = integrate_dual_nucleus_to_hard(
        initial_state=initial,
        model=model,
        config=config,
        step_budget=1,
    )
    assert result.status == "checkpoint"
    assert result.binary_initial_state is None
    assert result.final_state.inspiral_state.phase is InspiralPhase.COMMON_NUCLEUS_UNBOUND


def test_phase_history_and_binary_handoff_survive_checkpoint_resume(monkeypatch) -> None:
    model = _phase_aware_model()
    config = _phase_aware_config()
    initial = initial_kpc_to_hard_state(
        event_uid="capture-resume",
        dynamical_state=_inside_unbound_state(model),
        model=model,
        config=config,
    )

    def prescribed_capture(state, _model, time_step):
        axis = 15.0 if state.completed_steps == 0 else 7.0
        return DualNucleusState(
            state.elapsed_myr + time_step,
            np.array([10.0, 0.0, 0.0]),
            _tangential_velocity_for_axis(axis),
            state.envelope_truncation_radius_pc,
            state.completed_steps + 1,
        )

    monkeypatch.setattr(kpc_inspiral, "_advance_phase_aware_rk4", prescribed_capture)
    uninterrupted = integrate_dual_nucleus_to_hard(
        initial_state=initial, model=model, config=config
    )
    partial = integrate_dual_nucleus_to_hard(
        initial_state=initial, model=model, config=config, step_budget=1
    )
    assert partial.status == "checkpoint"
    assert partial.final_state.inspiral_state.phase is InspiralPhase.BOUND_BINARY
    resumed = integrate_dual_nucleus_to_hard(
        initial_state=partial.final_state, model=model, config=config
    )

    assert uninterrupted.status == resumed.status == "reached_hard_binary"
    assert resumed.final_state.transition_history == uninterrupted.final_state.transition_history
    assert resumed.final_state.inspiral_state == uninterrupted.final_state.inspiral_state
    assert resumed.final_state.dynamical_state.elapsed_myr == pytest.approx(
        uninterrupted.final_state.dynamical_state.elapsed_myr, rel=0.0, abs=0.0
    )
    assert np.array_equal(
        resumed.final_state.dynamical_state.position_pc,
        uninterrupted.final_state.dynamical_state.position_pc,
    )
    assert np.array_equal(
        resumed.final_state.dynamical_state.velocity_pc_myr,
        uninterrupted.final_state.dynamical_state.velocity_pc_myr,
    )
    assert [state.phase for state in resumed.final_state.transition_history] == [
        InspiralPhase.NUMERICAL_CAPTURE,
        InspiralPhase.COMMON_NUCLEUS_UNBOUND,
        InspiralPhase.BOUND_BINARY,
        InspiralPhase.HARD_BINARY,
    ]
    handoff = resumed.binary_initial_state
    assert handoff is not None
    assert handoff.elapsed_myr == 0.0
    assert handoff.semimajor_axis_pc == pytest.approx(
        resumed.final_state.inspiral_state.semimajor_axis_pc
    )
    assert handoff.eccentricity_squared == pytest.approx(
        resumed.final_state.inspiral_state.eccentricity**2
    )


def test_resume_payload_rejects_malformed_transition_history() -> None:
    model = _phase_aware_model()
    config = _phase_aware_config()
    valid = initial_kpc_to_hard_state(
        event_uid="capture-history",
        dynamical_state=_inside_unbound_state(model),
        model=model,
        config=config,
    )
    wrong_event = replace(valid.transition_history[0], event_uid="other-capture")
    with pytest.raises(ValueError, match="event_uid values must agree"):
        KpcToHardState(
            valid.dynamical_state,
            valid.inspiral_state,
            (wrong_event,) + valid.transition_history[1:],
        )

    time_reversed = replace(valid.transition_history[0], elapsed_myr=0.1)
    with pytest.raises(ValueError, match="elapsed times must be non-decreasing"):
        KpcToHardState(
            valid.dynamical_state,
            valid.inspiral_state,
            (time_reversed,) + valid.transition_history[1:],
        )


def test_unbound_exit_is_censored_outside_static_host_domain(monkeypatch) -> None:
    model = _phase_aware_model()
    config = _phase_aware_config()
    initial = initial_kpc_to_hard_state(
        event_uid="capture-flyby",
        dynamical_state=_inside_unbound_state(model),
        model=model,
        config=config,
    )

    def leave_common_nucleus(state, _model, time_step):
        return DualNucleusState(
            state.elapsed_myr + time_step,
            np.array([21.0, 0.0, 0.0]),
            state.velocity_pc_myr,
            state.envelope_truncation_radius_pc,
            state.completed_steps + 1,
        )

    monkeypatch.setattr(
        kpc_inspiral, "_advance_phase_aware_rk4", leave_common_nucleus
    )
    result = integrate_dual_nucleus_to_hard(
        initial_state=initial, model=model, config=config
    )
    assert result.status == "outside"
    assert result.final_state.inspiral_state.phase is InspiralPhase.CENSORED
    assert result.delay_segment.status == "timeout"


@pytest.mark.parametrize("initially_bound", [False, True])
def test_bound_energy_outside_common_nucleus_cannot_be_promoted(
    monkeypatch, initially_bound: bool,
) -> None:
    model = _phase_aware_model()
    config = _phase_aware_config()
    if initially_bound:
        starting = initial_dual_nucleus_state(
            position_pc=np.array([10.0, 0.0, 0.0]),
            velocity_pc_myr=_tangential_velocity_for_axis(15.0),
            model=model,
        )
    else:
        starting = _inside_unbound_state(model)
    initial = initial_kpc_to_hard_state(
        event_uid="capture-bound-energy-exit",
        dynamical_state=starting,
        model=model,
        config=config,
    )
    assert initial.inspiral_state.phase is (
        InspiralPhase.BOUND_BINARY if initially_bound
        else InspiralPhase.COMMON_NUCLEUS_UNBOUND
    )

    def leave_with_bound_energy(state, _model, time_step):
        return DualNucleusState(
            state.elapsed_myr + time_step,
            np.array([21.0, 0.0, 0.0]),
            _tangential_velocity_for_axis(30.0, radius_pc=21.0),
            state.envelope_truncation_radius_pc,
            state.completed_steps + 1,
        )

    monkeypatch.setattr(
        kpc_inspiral, "_advance_phase_aware_rk4", leave_with_bound_energy
    )
    result = integrate_dual_nucleus_to_hard(
        initial_state=initial, model=model, config=config,
    )
    assert result.status == "outside"
    assert result.final_state.inspiral_state.phase is InspiralPhase.CENSORED
    assert result.binary_initial_state is None
    assert result.delay_segment.status == "timeout"


def test_primary_mass_must_match_static_host_central_mass() -> None:
    model = _phase_aware_model()
    config = KpcToHardConfig(
        primary_bh_mass_msun=2.0e8,
        common_nucleus_radius_pc=20.0,
        sigma_pc_myr=100.0,
        hard_binary_radius_pc=8.0,
        maximum_time_myr=1.0,
        maximum_step_myr=1.0e-3,
    )
    initial = initial_kpc_to_hard_state(
        event_uid="capture-invalid-host",
        dynamical_state=_inside_unbound_state(model),
        model=model,
        config=config,
    )
    result = integrate_dual_nucleus_to_hard(
        initial_state=initial, model=model, config=config
    )
    assert result.status == "invalid"
    assert result.final_state.inspiral_state.phase is InspiralPhase.INVALID
    assert result.delay_segment.status == "invalid"


def test_hard_radius_outside_common_nucleus_is_invalid_not_complete() -> None:
    model = _phase_aware_model()
    config = KpcToHardConfig(
        primary_bh_mass_msun=1.0e8,
        common_nucleus_radius_pc=20.0,
        sigma_pc_myr=100.0,
        hard_binary_radius_pc=30.0,
        maximum_time_myr=1.0,
        maximum_step_myr=1.0e-3,
    )
    initial = initial_kpc_to_hard_state(
        event_uid="capture-invalid-hard-radius",
        dynamical_state=_inside_unbound_state(model),
        model=model,
        config=config,
    )
    assert initial.inspiral_state.phase is InspiralPhase.INVALID
    result = integrate_dual_nucleus_to_hard(
        initial_state=initial, model=model, config=config
    )
    assert result.status == "invalid"
    assert result.binary_initial_state is None
