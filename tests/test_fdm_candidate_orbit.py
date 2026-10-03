from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.fdm_candidate_orbit import integrate_candidate_fdm_outer_orbit
import fdm_smbh_delay.fdm_candidate_checkpoint as candidate_checkpoint
from fdm_smbh_delay.fdm_outer_halo import FDMOuterHaloClosure
from fdm_smbh_delay.fdm_outer_response import FDMOuterResponseTable
from fdm_smbh_delay.fdm_outer_response_family import FDMOuterResponseFamily
from fdm_smbh_delay.galaxy_environment import (
    CompositePotential, DehnenProfile, FDMBackground, GasBackground,
    NuclearEnvelope, StellarBackground,
)
from fdm_smbh_delay.kpc_inspiral import (
    GasFrictionConfig, KpcInspiralModel, KpcIntegrationConfig,
    StellarFrictionConfig, force_budget, initial_dual_nucleus_state,
)
from fdm_smbh_delay.soliton import SchiveSoliton


def _inputs(*, upper_radius: float = 200.0, diffusion: float = 0.01):
    primary = 1.0e8
    secondary = 2.0e7
    model = KpcInspiralModel(
        host_potential=CompositePotential((), central_point_mass_msun=primary),
        secondary_bh_mass_msun=secondary,
    )
    speed = np.sqrt(G_INTERNAL * (primary + secondary) * 1.5 / 50.0)
    initial = initial_dual_nucleus_state(
        position_pc=np.array([50.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, speed, 0.0]), model=model,
    )
    table = FDMOuterResponseTable(
        radii_pc=np.array([10.0, upper_radius]),
        drift_acceleration_pc_myr2=np.zeros((2, 3)),
        diffusion_tensor_pc2_myr3=np.array([diffusion * np.eye(3)] * 2),
        response_status="calibrated", component_frame="orbital_rtn",
        diffusion_convention="velocity_covariance_rate",
    )
    family = FDMOuterResponseFamily(
        mass_ratios_q=(0.2,), eccentricities=(0.0, 0.9),
        tables=((table, table),),
        source_sha256=(("a" * 64, "a" * 64),),
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
    config = KpcIntegrationConfig(
        target_radius_pc=10.0, maximum_time_myr=1.0,
        maximum_step_myr=0.005, maximum_steps=100,
    )
    kwargs = dict(
        model=model, config=config, response_family=family,
        closure_by_source_sha256={"a" * 64: closure},
        radial_support_pc=(5.0, 400.0), random_seed=29,
    )
    return initial, kwargs


def test_candidate_orbit_restart_matches_uninterrupted_and_tracks_sources() -> None:
    initial, kwargs = _inputs()
    full = integrate_candidate_fdm_outer_orbit(
        initial_state=initial, step_budget=4, **kwargs,
    )
    first = integrate_candidate_fdm_outer_orbit(
        initial_state=initial, step_budget=2, **kwargs,
    )
    resumed = integrate_candidate_fdm_outer_orbit(
        initial_state=first.final_state, step_budget=2, **kwargs,
    )
    assert full.status == first.status == resumed.status == "checkpoint"
    assert full.final_state.completed_steps == 4
    assert full.final_state.elapsed_myr == resumed.final_state.elapsed_myr
    assert np.array_equal(full.final_state.position_pc, resumed.final_state.position_pc)
    assert np.array_equal(full.final_state.velocity_pc_myr, resumed.final_state.velocity_pc_myr)
    assert full.used_source_sha256 == first.used_source_sha256 == resumed.used_source_sha256
    assert len(full.accepted_states) == 5


def test_zero_response_candidate_preserves_kepler_energy_and_angular_momentum() -> None:
    initial, kwargs = _inputs(diffusion=0.0)
    result = integrate_candidate_fdm_outer_orbit(
        initial_state=initial, step_budget=8, **kwargs,
    )
    assert result.status == "checkpoint"
    mu = G_INTERNAL * (1.0e8 + 2.0e7)

    def invariants(state):
        energy = 0.5 * np.dot(state.velocity_pc_myr, state.velocity_pc_myr) - mu / state.radius_pc
        angular = np.linalg.norm(np.cross(state.position_pc, state.velocity_pc_myr))
        return energy, angular

    assert invariants(result.final_state) == pytest.approx(invariants(initial), rel=1.0e-9)


def test_candidate_orbit_censors_unsupported_endpoint_without_accepting_it() -> None:
    initial, kwargs = _inputs(upper_radius=50.0007, diffusion=0.0)
    result = integrate_candidate_fdm_outer_orbit(
        initial_state=initial, step_budget=1, **kwargs,
    )
    assert result.status == "censored"
    assert "endpoint" in result.reason
    assert result.final_state is initial
    assert result.final_state.completed_steps == 0
    assert len(result.accepted_states) == 1


def test_candidate_orbit_never_enters_from_unsupported_start() -> None:
    initial, kwargs = _inputs(upper_radius=49.9, diffusion=0.0)
    result = integrate_candidate_fdm_outer_orbit(
        initial_state=initial, step_budget=1, **kwargs,
    )
    assert result.status == "censored"
    assert "start unsupported" in result.reason
    assert result.final_state is initial
    assert result.used_source_sha256 == ()


def test_candidate_orbit_keeps_stellar_gas_and_tidal_channels() -> None:
    initial, kwargs = _inputs(diffusion=0.0)
    stars = DehnenProfile(1.0e8, 100.0, 1.0)
    gas = DehnenProfile(1.0e8, 100.0, 1.0)
    host = CompositePotential((stars, gas), central_point_mass_msun=1.0e8)
    base = replace(kwargs["model"], host_potential=host)
    rich = replace(
        base,
        nuclear_envelope=NuclearEnvelope(1.0e6, 5.0, 100.0),
        stellar_background=StellarBackground(stars, 10.0, np.zeros(3)),
        stellar_friction=StellarFrictionConfig(1.0),
        gas_background=GasBackground(gas, 10.0, 0.2),
        gas_friction=GasFrictionConfig(1.0),
    )
    rich_initial = initial_dual_nucleus_state(
        position_pc=initial.position_pc, velocity_pc_myr=initial.velocity_pc_myr,
        model=rich,
    )
    budget = force_budget(rich_initial, rich)
    assert budget.stellar is not None and budget.gas is not None
    assert budget.fdm is None
    rich_result = integrate_candidate_fdm_outer_orbit(
        initial_state=rich_initial, step_budget=1, **{**kwargs, "model": rich},
    )
    plain_result = integrate_candidate_fdm_outer_orbit(
        initial_state=initial, step_budget=1, **{**kwargs, "model": base},
    )
    assert rich_result.status == plain_result.status == "checkpoint"
    assert rich_result.final_state.envelope_truncation_radius_pc <= 100.0
    assert not np.array_equal(
        rich_result.final_state.velocity_pc_myr,
        plain_result.final_state.velocity_pc_myr,
    )


def test_candidate_orbit_rejects_analytic_fdm_double_counting() -> None:
    initial, kwargs = _inputs()
    analytic = FDMBackground(
        soliton=SchiveSoliton.from_mass(1.0e8, 100.0, "total_profile"),
        particle_mass_ev=1.0e-21, alpha_df=0.341,
        bulk_velocity_pc_myr=np.zeros(3),
    )
    model = replace(kwargs["model"], fdm_background=analytic)
    with pytest.raises(ValueError, match="cannot be combined"):
        integrate_candidate_fdm_outer_orbit(
            initial_state=initial, step_budget=1, **{**kwargs, "model": model},
        )


def test_candidate_orbit_censors_colored_noise_step_before_state_change() -> None:
    initial, kwargs = _inputs()
    closure = kwargs["closure_by_source_sha256"]["a" * 64]
    slow_closure = replace(closure, coherence_time_myr=np.array([1.0, 1.0]))
    result = integrate_candidate_fdm_outer_orbit(
        initial_state=initial, step_budget=1,
        **{**kwargs, "closure_by_source_sha256": {"a" * 64: slow_closure}},
    )
    assert result.status == "censored"
    assert "colored-noise model" in result.reason
    assert result.final_state is initial
    assert result.used_source_sha256 == ()


def test_candidate_disk_checkpoint_restarts_and_rejects_changed_inputs(
    tmp_path, monkeypatch,
) -> None:
    initial, kwargs = _inputs()
    full = integrate_candidate_fdm_outer_orbit(
        initial_state=initial, step_budget=4, **kwargs,
    )
    first = integrate_candidate_fdm_outer_orbit(
        initial_state=initial, step_budget=2, **kwargs,
    )
    path = tmp_path / "candidate.json"
    candidate_checkpoint.write_candidate_fdm_checkpoint(path, first.final_state, **kwargs)
    restored = candidate_checkpoint.read_candidate_fdm_checkpoint(path, **kwargs)
    resumed = integrate_candidate_fdm_outer_orbit(
        initial_state=restored, step_budget=2, **kwargs,
    )
    assert np.array_equal(full.final_state.position_pc, resumed.final_state.position_pc)
    assert np.array_equal(full.final_state.velocity_pc_myr, resumed.final_state.velocity_pc_myr)
    with pytest.raises(ValueError, match="physics fingerprint"):
        candidate_checkpoint.read_candidate_fdm_checkpoint(
            path, **{**kwargs, "random_seed": 30},
        )
    with pytest.raises(ValueError, match="physics fingerprint"):
        candidate_checkpoint.read_candidate_fdm_checkpoint(
            path, **{**kwargs, "closure_by_source_sha256": {}},
        )
    monkeypatch.setattr(candidate_checkpoint, "candidate_implementation_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="implementation fingerprint"):
        candidate_checkpoint.read_candidate_fdm_checkpoint(path, **kwargs)
    monkeypatch.undo()
    raw = json.loads(path.read_text())
    raw["state"]["position_pc"][0] += 1.0
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="state checksum"):
        candidate_checkpoint.read_candidate_fdm_checkpoint(path, **kwargs)
