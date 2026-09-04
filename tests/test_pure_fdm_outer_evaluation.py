from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fdm_smbh_delay.fdm_outer_halo import FDMOuterHaloClosure
from fdm_smbh_delay.fdm_outer_wave_ledger import FDMOuterWaveLedger
from fdm_smbh_delay.pure_fdm_outer_evaluation import (
    evaluate_pure_fdm_outer_result_index,
    load_pure_fdm_outer_result_index,
)
from fdm_smbh_delay.pure_fdm_nested_registration import (
    build_pure_fdm_nested_registration_manifest,
)
from fdm_smbh_delay.pure_fdm_zoom import NestedZoomCheckpointContract
from fdm_smbh_delay.zoom_calibration import load_zoom_grid


def _closure() -> FDMOuterHaloClosure:
    return FDMOuterHaloClosure(
        radii_pc=np.array([10.0, 20.0]),
        mass_current_msun_pc2_myr=np.zeros((2, 3)),
        coherence_time_myr=np.array([1.0, 2.0]),
        de_broglie_wavelength_pc=np.array([2.0, 3.0]),
        velocity_diffusion_pc2_myr3=np.array([0.01, 0.02]),
        density_gradient_scale_pc=np.array([15.0, 25.0]),
    )


def _contract(case_id: str, manifest_sha256: str) -> NestedZoomCheckpointContract:
    return NestedZoomCheckpointContract(
        outer_manifest_sha256=manifest_sha256,
        outer_case_id=case_id,
        checkpoint_path=f"/gpfs/example/{case_id}/checkpoint",
        checkpoint_sha256="b" * 64,
        capture_event_uid="capture-1",
        force_ledger_sha256="c" * 64,
        wave_ledger_sha256="d" * 64,
        fdm_particle_mass_ev=1.0e-21,
        soliton_mass_msun=1.0e10,
        core_radius_pc=200.0,
        eta_sp=0.001,
        checkpoint_separation_pc=10.0,
        target_separation_pc=1.0,
        finest_cell_size_pc=0.25,
        minimum_softening_pc=0.125,
        maximum_wake_extent_pc=2.0,
        de_broglie_wavelength_pc=1.0,
        hjm_wave_seam_clearance_pc=3.0,
        boundary_clearance_pc=3.0,
    )


def _wave_ledger() -> FDMOuterWaveLedger:
    return FDMOuterWaveLedger(
        source_path="/gpfs/example/outer-wave-ledger.json",
        source_sha256="d" * 64,
        force_ledger_sha256="c" * 64,
        field_snapshot_index_sha256="e" * 64,
        profile_snapshot_index_sha256="f" * 64,
        sample_times_myr=np.array([0.0, 1.0]),
        core_centres_pc=np.zeros((2, 3)),
        fdm_current_msun_pc2_myr=np.ones((2, 3)),
        coherence_time_myr=np.array([0.1, 0.2]),
        de_broglie_wavelength_pc=np.array([1.0, 1.0]),
        granule_power=np.array([0.01, 0.02]),
        dipole_l1_real_imag=np.zeros((2, 2)),
        quadrupole_l2_real_imag=np.zeros((2, 2)),
        wake_extent_pc=np.array([2.0, 2.0]),
        hjm_wave_seam_clearance_pc=np.array([3.0, 3.0]),
        boundary_clearance_pc=np.array([3.0, 3.0]),
        maximum_relative_hamiltonian_error=1.0e-5,
        maximum_relative_mass_error=1.0e-5,
        maximum_relative_angular_momentum_error=2.0e-5,
    )


def _write_result(path: Path, case, manifest_sha256: str, common_time: float) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": case.case_id,
                "case": case.as_dict(),
                "manifest_sha256": manifest_sha256,
                "capture_event_uid": "capture-1",
                "stages": {
                    "outer_start": {
                        "status": "complete",
                        "elapsed_since_capture_myr": 0.0,
                        "separation_pc": 5000.0,
                    },
                    "common_core": {
                        "status": "complete",
                        "elapsed_since_capture_myr": common_time,
                        "separation_pc": 100.0,
                    },
                    "nested_checkpoint": {
                        "status": "complete",
                        "elapsed_since_capture_myr": 15.0,
                        "separation_pc": 10.0,
                    },
                },
                "diagnostics": {
                    "maximum_relative_energy_error": 1.0e-5,
                    "maximum_relative_angular_momentum_error": 2.0e-5,
                    "minimum_de_broglie_resolution_cells": 8.0,
                    "minimum_wake_resolution_cells": 8.0,
                },
                "outer_closure": _closure().as_dict(),
                "outer_wave_ledger": _wave_ledger().as_dict(),
                "nested_checkpoint": _contract(case.case_id, manifest_sha256).as_dict(),
            }
        ),
        encoding="utf-8",
    )


def _write_baseline_index(tmp_path: Path, *, omit_case_id: str | None = None) -> tuple[Path, object]:
    grid = load_zoom_grid("configs/pure_fdm_outer_zoom_grid.yaml")
    baseline = grid.cases[0].physics.physics_id
    selected = [case for case in grid.cases if case.physics.physics_id == baseline]
    paths: dict[str, str] = {}
    for case in selected:
        if case.case_id == omit_case_id:
            continue
        common_time = 10.0 + 2.0 * case.replicate
        if case.numerics.finest_cell_size_pc > 2.0:
            common_time += 0.5 if case.replicate == 0 else 0.1
        result_path = tmp_path / f"{case.case_id}.json"
        _write_result(result_path, case, grid.manifest_sha256, common_time)
        paths[case.case_id] = result_path.name
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_sha256": grid.manifest_sha256,
                "results": paths,
            }
        ),
        encoding="utf-8",
    )
    return index_path, grid


def test_evaluator_accepts_complete_phase_ensemble_but_not_a_delay(tmp_path: Path) -> None:
    index_path, grid = _write_baseline_index(tmp_path)
    evaluation = evaluate_pure_fdm_outer_result_index(
        grid, load_pure_fdm_outer_result_index(index_path, grid=grid)
    ).as_dict()

    accepted = [
        item
        for item in evaluation["physics_assessments"]
        if item["status"] == "accepted_outer_ensemble_uncalibrated"
    ]
    assert evaluation["status"] == "pure_fdm_outer_ensemble_evaluated"
    assert "no physical kpc-to-pc delay" in evaluation["interpretation"]
    assert len(accepted) == 1
    assert accepted[0]["common_core_delay_mean_myr"] == pytest.approx(11.0)
    assert len(accepted[0]["eligible_nested_outer_case_ids"]) == 2
    assert evaluation["accepted_outer_ensemble_count"] == 1


def test_evaluator_censors_missing_phase_result_without_imputing_zero(tmp_path: Path) -> None:
    index_path, grid = _write_baseline_index(tmp_path)
    record = json.loads(index_path.read_text(encoding="utf-8"))
    omitted = next(iter(record["results"]))
    index_path, grid = _write_baseline_index(tmp_path, omit_case_id=omitted)
    evaluation = evaluate_pure_fdm_outer_result_index(
        grid, load_pure_fdm_outer_result_index(index_path, grid=grid)
    ).as_dict()
    baseline = grid.cases[0].physics.physics_id
    assessment = next(
        item for item in evaluation["physics_assessments"] if item["physics_id"] == baseline
    )
    assert assessment["status"] == "censored"
    assert assessment["common_core_delay_mean_myr"] is None
    assert any("not registered" in reason for reason in assessment["reasons"])


def test_evaluator_censors_malformed_registered_result_without_crashing(tmp_path: Path) -> None:
    index_path, grid = _write_baseline_index(tmp_path)
    record = json.loads(index_path.read_text(encoding="utf-8"))
    result_path = tmp_path / next(iter(record["results"].values()))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["stages"]["common_core"] = []
    result_path.write_text(json.dumps(result), encoding="utf-8")
    evaluation = evaluate_pure_fdm_outer_result_index(
        grid, load_pure_fdm_outer_result_index(index_path, grid=grid)
    ).as_dict()
    baseline = grid.cases[0].physics.physics_id
    assessment = next(
        item for item in evaluation["physics_assessments"] if item["physics_id"] == baseline
    )
    assert assessment["status"] == "censored"
    assert any("unreadable or invalid" in reason for reason in assessment["reasons"])


def test_result_index_rejects_wrong_manifest(tmp_path: Path) -> None:
    index_path, grid = _write_baseline_index(tmp_path)
    record = json.loads(index_path.read_text(encoding="utf-8"))
    record["manifest_sha256"] = "0" * 64
    index_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the grid"):
        load_pure_fdm_outer_result_index(index_path, grid=grid)


def test_nested_registration_uses_only_accepted_exact_checkpoints(tmp_path: Path) -> None:
    index_path, grid = _write_baseline_index(tmp_path)
    evaluation = evaluate_pure_fdm_outer_result_index(
        grid, load_pure_fdm_outer_result_index(index_path, grid=grid)
    )
    registration = build_pure_fdm_nested_registration_manifest(evaluation).as_dict()

    ready = [
        item
        for item in registration["physics_registrations"]
        if item["status"] == "ready_for_operator_registration"
    ]
    assert registration["status"] == "pure_fdm_nested_zoom_registration_ready"
    assert registration["ready_registration_count"] == 2
    assert len(ready) == 1
    assert len(ready[0]["registrations"]) == 2
    assert all(
        item["execution"] == "operator registration only; no job submission"
        for item in ready[0]["registrations"]
    )


def test_nested_registration_censors_hand_edited_checkpoint_physics(tmp_path: Path) -> None:
    index_path, grid = _write_baseline_index(tmp_path)
    record = json.loads(index_path.read_text(encoding="utf-8"))
    fine_case_id = next(
        case.case_id
        for case in grid.cases
        if case.physics == grid.cases[0].physics
        and case.numerics.finest_cell_size_pc == 2.0
    )
    result_path = tmp_path / record["results"][fine_case_id]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["nested_checkpoint"]["fdm_particle_mass_ev"] = 3.0e-22
    result_path.write_text(json.dumps(result), encoding="utf-8")

    evaluation = evaluate_pure_fdm_outer_result_index(
        grid, load_pure_fdm_outer_result_index(index_path, grid=grid)
    )
    registration = build_pure_fdm_nested_registration_manifest(evaluation).as_dict()
    baseline = grid.cases[0].physics.physics_id
    item = next(
        entry
        for entry in registration["physics_registrations"]
        if entry["physics_id"] == baseline
    )
    assert item["status"] == "censored"
    assert item["registrations"] == []
    assert any("fdm particle mass" in reason for reason in item["reasons"])


def test_cli_writes_atomic_evaluation_record(tmp_path: Path) -> None:
    index_path, _ = _write_baseline_index(tmp_path)
    output = tmp_path / "output" / "evaluation.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_pure_fdm_outer_ensemble.py",
            "configs/pure_fdm_outer_zoom_grid.yaml",
            str(index_path),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "pure_fdm_outer_ensemble_evaluated"
    assert json.loads(completed.stdout)["manifest_sha256"] == record["manifest_sha256"]


def test_nested_registration_cli_writes_no_submit_record(tmp_path: Path) -> None:
    index_path, _ = _write_baseline_index(tmp_path)
    output = tmp_path / "output" / "registration.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_pure_fdm_nested_zoom_registration.py",
            "configs/pure_fdm_outer_zoom_grid.yaml",
            str(index_path),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "pure_fdm_nested_zoom_registration_ready"
    assert record["ready_registration_count"] == 2
