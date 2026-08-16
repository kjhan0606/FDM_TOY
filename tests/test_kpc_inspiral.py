from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.galaxy_environment import (
    CompositePotential,
    DehnenProfile,
    NuclearEnvelope,
    StellarBackground,
)
from fdm_smbh_delay.kpc_inspiral import (
    KpcInspiralModel,
    KpcIntegrationConfig,
    StellarFrictionConfig,
    force_budget,
    initial_dual_nucleus_state,
    integrate_dual_nucleus,
    tidal_truncation_radius_pc,
)


def _point_mass_model(*, envelope=None) -> KpcInspiralModel:
    return KpcInspiralModel(
        host_potential=CompositePotential((), central_point_mass_msun=1.0e10),
        secondary_bh_mass_msun=1.0e8,
        nuclear_envelope=envelope,
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
