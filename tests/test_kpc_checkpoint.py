from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import fdm_smbh_delay.kpc_checkpoint as kpc_checkpoint
from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.galaxy_environment import CompositePotential, DehnenProfile
from fdm_smbh_delay.kpc_checkpoint import (
    kpc_physics_sha256,
    kpc_implementation_sha256,
    read_kpc_to_hard_checkpoint,
    write_kpc_to_hard_checkpoint,
)
from fdm_smbh_delay.kpc_inspiral import (
    DualNucleusState, KpcInspiralModel, KpcToHardConfig,
    initial_kpc_to_hard_state, integrate_dual_nucleus_to_hard,
)


def _case():
    model = KpcInspiralModel(
        host_potential=CompositePotential((), central_point_mass_msun=1.0e10),
        secondary_bh_mass_msun=1.0e8,
    )
    radius = 100.0
    speed = np.sqrt(G_INTERNAL * 1.01e10 / radius)
    dynamics = DualNucleusState(0.0, np.array([radius, 0.0, 0.0]),
                                np.array([0.0, speed, 0.0]), None)
    config = KpcToHardConfig(
        primary_bh_mass_msun=1.0e10,
        common_nucleus_radius_pc=10.0,
        sigma_pc_myr=100.0,
        maximum_time_myr=0.1,
        maximum_step_myr=0.01,
        hard_binary_radius_pc=1.0,
    )
    initial = initial_kpc_to_hard_state(
        event_uid="checkpoint-orbit", dynamical_state=dynamics,
        model=model, config=config,
    )
    return model, config, initial


def test_disk_restart_matches_uninterrupted_phase_aware_orbit(tmp_path) -> None:
    model, config, initial = _case()
    uninterrupted = integrate_dual_nucleus_to_hard(
        initial_state=initial, model=model, config=config,
    )
    partial = integrate_dual_nucleus_to_hard(
        initial_state=initial, model=model, config=config, step_budget=7,
    )
    assert partial.status == "checkpoint"
    path = tmp_path / "kpc-checkpoint.json"
    write_kpc_to_hard_checkpoint(path, partial.final_state, model, config)
    loaded = read_kpc_to_hard_checkpoint(path, model, config)
    assert loaded.transition_history == partial.final_state.transition_history
    assert np.array_equal(
        loaded.dynamical_state.position_pc, partial.final_state.dynamical_state.position_pc
    )
    resumed = integrate_dual_nucleus_to_hard(
        initial_state=loaded, model=model, config=config,
    )
    assert resumed.status == uninterrupted.status == "timeout"
    assert resumed.final_state.transition_history == uninterrupted.final_state.transition_history
    assert np.array_equal(
        resumed.final_state.dynamical_state.position_pc,
        uninterrupted.final_state.dynamical_state.position_pc,
    )
    assert np.array_equal(
        resumed.final_state.dynamical_state.velocity_pc_myr,
        uninterrupted.final_state.dynamical_state.velocity_pc_myr,
    )


def test_checkpoint_rejects_changed_force_or_integration_controls(tmp_path) -> None:
    model, config, initial = _case()
    path = tmp_path / "kpc-checkpoint.json"
    write_kpc_to_hard_checkpoint(path, initial, model, config)
    other_model = replace(
        model, host_potential=CompositePotential((DehnenProfile(1.0e8, 50.0, 1.0),),
                                                 central_point_mass_msun=1.0e10),
    )
    assert kpc_physics_sha256(model, config) != kpc_physics_sha256(other_model, config)
    with pytest.raises(ValueError, match="physics/configuration identity changed"):
        read_kpc_to_hard_checkpoint(path, other_model, config)
    with pytest.raises(ValueError, match="physics/configuration identity changed"):
        read_kpc_to_hard_checkpoint(path, model, replace(config, maximum_step_myr=0.005))


def test_checkpoint_rejects_changed_implementation_identity(tmp_path, monkeypatch) -> None:
    model, config, initial = _case()
    path = tmp_path / "kpc-checkpoint.json"
    write_kpc_to_hard_checkpoint(path, initial, model, config)
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == 2
    assert saved["implementation_sha256"] == kpc_implementation_sha256()
    assert len(saved["implementation_sha256"]) == 64
    monkeypatch.setattr(kpc_checkpoint, "kpc_implementation_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="physics implementation identity changed"):
        read_kpc_to_hard_checkpoint(path, model, config)


def test_checkpoint_rejects_state_corruption_and_out_of_budget_state(tmp_path) -> None:
    model, config, initial = _case()
    path = tmp_path / "kpc-checkpoint.json"
    write_kpc_to_hard_checkpoint(path, initial, model, config)
    record = json.loads(path.read_text())
    record["state"]["dynamics"]["position_pc"][0] = 999.0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="state digest disagrees"):
        read_kpc_to_hard_checkpoint(path, model, config)
    overrun = replace(initial.dynamical_state, completed_steps=config.maximum_steps + 1)
    write_kpc_to_hard_checkpoint(path, replace(initial, dynamical_state=overrun), model, config)
    with pytest.raises(ValueError, match="maximum step count"):
        read_kpc_to_hard_checkpoint(path, model, config)
