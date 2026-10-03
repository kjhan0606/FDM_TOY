from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fdm_smbh_delay.fdm_outer_halo import FDMOuterHaloClosure
from fdm_smbh_delay.fdm_outer_wave_ledger import FDMOuterWaveLedger
from fdm_smbh_delay.pure_fdm_outer_results import (
    PureFDMOuterRunResult,
    PureFDMOuterStage,
    assess_pure_fdm_outer_phase_ensemble,
    compare_pure_fdm_outer_resolution_pair,
    read_pure_fdm_outer_result,
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


def _result(case, manifest_sha256: str, *, common_time: float = 10.0, cells: float = 8.0):
    return PureFDMOuterRunResult(
        case=case,
        manifest_sha256=manifest_sha256,
        source_path=Path("outer-result.json"),
        source_sha256="a" * 64,
        capture_event_uid="capture-1",
        stages=(
            ("outer_start", PureFDMOuterStage("complete", 0.0, 5000.0)),
            ("common_core", PureFDMOuterStage("complete", common_time, 100.0)),
            ("nested_checkpoint", PureFDMOuterStage("complete", 15.0, 10.0)),
        ),
        maximum_relative_energy_error=1.0e-5,
        maximum_relative_angular_momentum_error=2.0e-5,
        minimum_de_broglie_resolution_cells=cells,
        minimum_wake_resolution_cells=cells,
        outer_closure=_closure(),
        outer_wave_ledger=_wave_ledger(),
        nested_checkpoint=_contract(case.case_id, manifest_sha256),
    )


def _pair():
    grid = load_zoom_grid("configs/pure_fdm_outer_zoom_grid.yaml")
    coarse, fine = grid.cases[:2]
    return fine, coarse, grid.manifest_sha256


def _pair_for_replicate(replicate: int):
    grid = load_zoom_grid("configs/pure_fdm_outer_zoom_grid.yaml")
    physics_id = grid.cases[0].physics.physics_id
    cases = [
        case
        for case in grid.cases
        if case.physics.physics_id == physics_id and case.replicate == replicate
    ]
    coarse, fine = sorted(cases, key=lambda case: case.numerics.finest_cell_size_pc, reverse=True)
    return fine, coarse, grid.manifest_sha256


def test_outer_resolution_pair_is_accepted_but_not_a_physical_delay() -> None:
    fine, coarse, manifest_sha = _pair()
    decision = compare_pure_fdm_outer_resolution_pair(
        _result(fine, manifest_sha, common_time=10.0),
        _result(coarse, manifest_sha, common_time=10.5),
    )
    assert decision.accepted_for_inner_registration
    assert decision.status == "accepted_outer_uncalibrated"
    assert decision.common_core_delay_systematic_fraction == pytest.approx(0.05)


def test_outer_resolution_pair_rejects_unresolved_wave() -> None:
    fine, coarse, manifest_sha = _pair()
    decision = compare_pure_fdm_outer_resolution_pair(
        _result(fine, manifest_sha),
        _result(coarse, manifest_sha, cells=3.0),
    )
    assert decision.status == "rejected"
    assert any("underresolved" in reason for reason in decision.reasons)


def test_phase_ensemble_keeps_stochastic_variation_separate() -> None:
    fine0, coarse0, manifest_sha = _pair_for_replicate(0)
    fine1, coarse1, _ = _pair_for_replicate(1)
    ensemble = assess_pure_fdm_outer_phase_ensemble(
        (
            compare_pure_fdm_outer_resolution_pair(
                _result(fine0, manifest_sha, common_time=10.0),
                _result(coarse0, manifest_sha, common_time=10.5),
            ),
            compare_pure_fdm_outer_resolution_pair(
                _result(fine1, manifest_sha, common_time=12.0),
                _result(coarse1, manifest_sha, common_time=12.1),
            ),
        )
    )
    assert ensemble.ready_for_overlap_evaluation
    assert ensemble.common_core_delay_mean_myr == pytest.approx(11.0)
    assert ensemble.common_core_delay_standard_deviation_myr == pytest.approx(np.sqrt(2.0))


def test_phase_ensemble_rejects_duplicate_replicate() -> None:
    fine, coarse, manifest_sha = _pair()
    decision = compare_pure_fdm_outer_resolution_pair(
        _result(fine, manifest_sha), _result(coarse, manifest_sha)
    )
    with pytest.raises(ValueError, match="duplicate replicate"):
        assess_pure_fdm_outer_phase_ensemble((decision, decision))


def test_reader_validates_nested_contract_provenance(tmp_path: Path) -> None:
    fine, _, manifest_sha = _pair()
    contract = _contract(fine.case_id, manifest_sha)
    record = {
        "schema_version": 1,
        "case_id": fine.case_id,
        "case": fine.as_dict(),
        "manifest_sha256": manifest_sha,
        "capture_event_uid": "capture-1",
        "stages": {
            "outer_start": {
                "status": "complete",
                "elapsed_since_capture_myr": 0.0,
                "separation_pc": 5000.0,
            },
            "common_core": {
                "status": "complete",
                "elapsed_since_capture_myr": 10.0,
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
        "nested_checkpoint": contract.as_dict(),
    }
    path = tmp_path / "outer.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    result = read_pure_fdm_outer_result(path, case=fine, manifest_sha256=manifest_sha)
    assert result.nested_checkpoint == contract

    record["nested_checkpoint"]["outer_case_id"] = "wrong-case"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="different outer case"):
        read_pure_fdm_outer_result(path, case=fine, manifest_sha256=manifest_sha)

    record["nested_checkpoint"] = contract.as_dict()
    record["nested_checkpoint"]["wave_ledger_sha256"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="wave ledger does not match"):
        read_pure_fdm_outer_result(path, case=fine, manifest_sha256=manifest_sha)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (("stages", "common_core"), [], "outer stage common_core must be an object"),
        (("diagnostics", "minimum_wake_resolution_cells"), True, "finite non-negative"),
        (("outer_closure",), None, "outer FDM closure is invalid"),
    ],
)
def test_reader_reports_malformed_records_as_value_errors(
    tmp_path: Path, field: tuple[str, ...], replacement: object, message: str
) -> None:
    fine, _, manifest_sha = _pair()
    contract = _contract(fine.case_id, manifest_sha)
    record = {
        "schema_version": 1,
        "case_id": fine.case_id,
        "case": fine.as_dict(),
        "manifest_sha256": manifest_sha,
        "capture_event_uid": "capture-1",
        "stages": {
            "outer_start": {
                "status": "complete",
                "elapsed_since_capture_myr": 0.0,
                "separation_pc": 5000.0,
            },
            "common_core": {
                "status": "complete",
                "elapsed_since_capture_myr": 10.0,
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
        "nested_checkpoint": contract.as_dict(),
    }
    cursor = record
    for key in field[:-1]:
        cursor = cursor[key]
    cursor[field[-1]] = replacement
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_pure_fdm_outer_result(path, case=fine, manifest_sha256=manifest_sha)
