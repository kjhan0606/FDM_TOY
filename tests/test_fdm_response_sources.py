from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import fdm_smbh_delay.fdm_candidate_checkpoint as candidate_checkpoint
from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.fdm_candidate_phase import integrate_verified_candidate_fdm_to_hard_boundary
from fdm_smbh_delay.fdm_outer_halo import FDMOuterHaloClosure
from fdm_smbh_delay.fdm_outer_response import FDMOuterResponseTable
from fdm_smbh_delay.fdm_response_sources import read_verified_fdm_response_sources
from fdm_smbh_delay.galaxy_environment import CompositePotential
from fdm_smbh_delay.kpc_inspiral import (
    DualNucleusState, KpcInspiralModel, KpcIntegrationConfig,
    KpcToHardConfig, initial_kpc_to_hard_state,
)


def _files(tmp_path):
    entries = []
    for q in (0.2, 0.5):
        for e in (0.0, 0.8):
            table = FDMOuterResponseTable(
                radii_pc=np.array([10.0, 20.0]),
                drift_acceleration_pc_myr2=np.array([[q, -e, 0.0]] * 2),
                diffusion_tensor_pc2_myr3=np.array([np.eye(3)] * 2),
                response_status="calibrated", component_frame="orbital_rtn",
                diffusion_convention="velocity_covariance_rate",
            )
            closure = FDMOuterHaloClosure(
                radii_pc=np.array([10.0, 20.0]),
                mass_current_msun_pc2_myr=np.zeros((2, 3)),
                coherence_time_myr=np.array([1.0e-6, 1.0e-6]),
                de_broglie_wavelength_pc=np.ones(2),
                velocity_diffusion_pc2_myr3=np.ones(2),
                density_gradient_scale_pc=np.ones(2),
                closure_status="calibrated",
            )
            path = tmp_path / f"q{q}-e{e}.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "mass_ratio_q": q,
                "eccentricity": e,
                "source_case_id": f"case-q{q}-e{e}",
                "response": table.as_dict(),
                "closure": closure.as_dict(),
            }, sort_keys=True))
            entries.append({
                "mass_ratio_q": q, "eccentricity": e,
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, entries)
    return manifest, entries


def _manifest(path, entries):
    path.write_text(json.dumps({
        "schema_version": 1,
        "status": "candidate_outer_fdm_response_sources",
        "nodes": entries,
    }, sort_keys=True))


def test_source_loader_verifies_all_qe_files_and_rechecks_current_bytes(tmp_path) -> None:
    manifest, entries = _files(tmp_path)
    bundle = read_verified_fdm_response_sources(manifest)
    assert bundle.status == "source_files_verified_physics_pending"
    assert bundle.response_family.mass_ratios_q == (0.2, 0.5)
    assert bundle.response_family.eccentricities == (0.0, 0.8)
    decision = bundle.response_family.decision(0.35, 0.4, 15.0)
    assert decision["status"] == "interpolated_candidate_pending_physical_validation"
    assert set(decision["source_sha256"]) == set(bundle.closure_by_source_sha256)
    bundle.verify_current_files()

    node_path = tmp_path / entries[0]["path"]
    node_path.write_bytes(node_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after loading"):
        bundle.verify_current_files()


def test_source_loader_censors_missing_corner_and_coordinate_mismatch(tmp_path) -> None:
    manifest, entries = _files(tmp_path)
    _manifest(manifest, entries[:-1])
    with pytest.raises(ValueError, match="grid is incomplete"):
        read_verified_fdm_response_sources(manifest)

    changed = [dict(entry) for entry in entries]
    changed[0]["eccentricity"] = 0.1
    _manifest(manifest, changed)
    with pytest.raises(ValueError, match="coordinates disagree"):
        read_verified_fdm_response_sources(manifest)


def test_source_loader_rejects_wrong_hash_and_oversized_json(tmp_path, monkeypatch) -> None:
    manifest, entries = _files(tmp_path)
    changed = [dict(entry) for entry in entries]
    changed[0]["sha256"] = "0" * 64
    _manifest(manifest, changed)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_verified_fdm_response_sources(manifest)

    _manifest(manifest, entries)
    import fdm_smbh_delay.fdm_response_sources as sources
    monkeypatch.setattr(sources, "_NODE_LIMIT_BYTES", 100)
    with pytest.raises(ValueError, match="exceeds 100 bytes"):
        read_verified_fdm_response_sources(manifest)


def test_source_bundle_detects_changed_manifest(tmp_path) -> None:
    manifest, entries = _files(tmp_path)
    bundle = read_verified_fdm_response_sources(manifest)
    _manifest(manifest, list(reversed(entries)))
    with pytest.raises(ValueError, match="manifest changed"):
        bundle.verify_current_files()


def test_verified_candidate_run_and_phase_checkpoint_bind_manifest(tmp_path) -> None:
    manifest, entries = _files(tmp_path)
    bundle = read_verified_fdm_response_sources(manifest)
    primary, secondary, radius = 1.0e8, 2.0e7, 15.0
    speed = np.sqrt(G_INTERNAL * (primary + secondary) / radius)
    model = KpcInspiralModel(
        host_potential=CompositePotential((), central_point_mass_msun=primary),
        secondary_bh_mass_msun=secondary,
    )
    dynamics = DualNucleusState(
        0.0, np.array([radius, 0.0, 0.0]),
        np.array([0.0, speed, 0.0]), None, 0,
    )
    outer_config = KpcIntegrationConfig(10.0, 0.1, 0.001, maximum_steps=10)
    phase_config = KpcToHardConfig(
        primary_bh_mass_msun=primary, common_nucleus_radius_pc=20.0,
        sigma_pc_myr=100.0, maximum_time_myr=0.1,
        maximum_step_myr=0.001, hard_binary_radius_pc=2.0,
    )
    initial = initial_kpc_to_hard_state(
        event_uid="verified-source-case", dynamical_state=dynamics,
        model=model, config=phase_config,
    )
    kwargs = dict(
        source_bundle=bundle, model=model, outer_config=outer_config,
        phase_config=phase_config, radial_support_pc=(5.0, 30.0),
        random_seed=17,
    )
    result = integrate_verified_candidate_fdm_to_hard_boundary(
        initial_state=initial, step_budget=1, **kwargs,
    )
    assert result.status == "checkpoint"
    assert result.source_manifest_sha256 == bundle.manifest_sha256
    checkpoint = tmp_path / "verified_phase.json"
    candidate_checkpoint.write_verified_candidate_fdm_phase_checkpoint(
        checkpoint, result.final_state, **kwargs,
    )
    restored = candidate_checkpoint.read_verified_candidate_fdm_phase_checkpoint(
        checkpoint, **kwargs,
    )
    assert restored.transition_history == result.final_state.transition_history
    with pytest.raises(ValueError, match="physics fingerprint"):
        candidate_checkpoint.read_candidate_fdm_phase_checkpoint(
            checkpoint, model=model, outer_config=outer_config,
            phase_config=phase_config, response_family=bundle.response_family,
            closure_by_source_sha256=bundle.closure_by_source_sha256,
            radial_support_pc=(5.0, 30.0), random_seed=17,
        )
    node_path = tmp_path / entries[0]["path"]
    node_path.write_bytes(node_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after loading"):
        candidate_checkpoint.read_verified_candidate_fdm_phase_checkpoint(
            checkpoint, **kwargs,
        )
