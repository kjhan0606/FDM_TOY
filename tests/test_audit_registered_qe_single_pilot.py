from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from fdm_smbh_delay.qe_followup_design import read_verified_qe_followup_design
from scripts import audit_registered_qe_single_pilot as audit


def _paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    design_root = root / "results/wave_calibration_qe_followup_q100e030_a020"
    return (
        design_root / "design.json",
        root / "results/wave_calibration_qe_extension/physical_cases.csv",
        design_root / "run_manifest.csv",
    )


def _run(tmp_path: Path, *, identity: str = "valid") -> Path:
    design_path, cases, manifest = _paths()
    design, file_hash = read_verified_qe_followup_design(
        design_path, physical_cases=cases, run_manifest=manifest
    )
    run = tmp_path / "qe_q100_e030_a020_n256"
    run.mkdir()
    binding = {
        "status": "qe_prospective_design_bound_not_a_calibration_release",
        "design_sha256": design["design_sha256"],
        "file_sha256": file_hash,
        "role": "coarse",
    }
    if identity == "tampered":
        binding["design_sha256"] = "0" * 64
    metadata = {
        "run_id": run.name,
        "case_id": design["case_id"],
        "resolution": 256,
        "box_size_pc": 26.4,
        "duration_myr": 0.1,
    }
    if identity != "missing":
        metadata["qe_design_binding"] = binding
    (run / "fdm_adapter_metadata.json").write_text(json.dumps(metadata))
    return run


def _loaded(count: int) -> dict:
    dtype = [
        ("cycle", float),
        ("start_time_myr", float),
        ("end_time_myr", float),
        ("mean_time_myr", float),
        ("orbital_period_myr", float),
        ("mean_separation_pc", float),
        ("mean_separation_over_cell_size", float),
        ("mean_eccentricity_osculating", float),
    ]
    orbit = np.zeros(count, dtype=dtype)
    orbit["cycle"] = np.arange(count)
    orbit["start_time_myr"] = np.arange(count) * 0.01
    orbit["end_time_myr"] = np.arange(1, count + 1) * 0.01
    orbit["mean_time_myr"] = (np.arange(count) + 0.5) * 0.01
    orbit["orbital_period_myr"] = 0.01
    orbit["mean_separation_pc"] = np.linspace(0.431, 0.437, count)
    orbit["mean_separation_over_cell_size"] = 4.2
    orbit["mean_eccentricity_osculating"] = np.linspace(0.25, 0.31, count)
    return {
        "orbit_series": orbit,
        "orbit_artifact_provenance": {
            "status": "legacy_unverified_orbit_artifacts",
            "reason": "test archival fixture",
        },
        "orbit": {
            "status": "orbit_averaged",
            "complete_orbits": count,
            "start_time_myr": 0.0,
            "end_time_myr": count * 0.01,
        },
        "conservation": {"initial_spatially_resolved_duration_myr": count * 0.01},
        "series": np.array(
            [(0.0,), (0.1,)], dtype=[("time_myr", float)]
        ),
    }


@pytest.mark.parametrize(("count", "passed"), [(7, False), (8, True)])
def test_registered_fixed_bin_n7_fails_n8_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, passed: bool
) -> None:
    run = _run(tmp_path)
    (run / "orbit_averaged_exchange_summary.json").write_text(
        json.dumps({
            "status": "orbit_averaged",
            "complete_orbits": count,
            "start_time_myr": 0.0,
            "end_time_myr": count * 0.01,
        })
    )
    monkeypatch.setattr(
        audit, "_require_pilot_complete",
        lambda *args, **kwargs: {
            "fdm_adapter_metadata.json": audit._sha256(
                run / "fdm_adapter_metadata.json"
            )
        },
    )
    monkeypatch.setattr(audit, "load_convergence_run", lambda *args: _loaded(count))
    result = audit.audit_registered_single_pilot(*_paths(), run)
    assert result["initially_resolved_complete_orbits_in_fixed_bin"] == count
    assert result["orbit_count_threshold_passed"] is passed
    assert result["measured_eccentricity_minimum"] == pytest.approx(0.25)
    assert result["measured_eccentricity_maximum"] == pytest.approx(0.31)
    assert result["measured_eccentricity_duration_weighted_mean"] == pytest.approx(0.28)
    assert result["release_status"] == "no_calibration_release"
    assert result["production_calibration_row_admitted"] is False
    assert result["orbit_artifact_provenance"]["status"] == (
        "legacy_unverified_orbit_artifacts"
    )


def test_registered_fixed_bin_rejects_mixed_orbit_summary_and_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    (run / "orbit_averaged_exchange_summary.json").write_text(json.dumps({
        "status": "orbit_averaged",
        "complete_orbits": 8,
        "start_time_myr": 0.0,
        "end_time_myr": 7.5,
    }))
    monkeypatch.setattr(
        audit, "_require_pilot_complete",
        lambda *args, **kwargs: {
            "fdm_adapter_metadata.json": audit._sha256(
                run / "fdm_adapter_metadata.json"
            )
        },
    )
    loaded = _loaded(8)
    loaded["orbit"]["end_time_myr"] = 7.5
    monkeypatch.setattr(audit, "load_convergence_run", lambda *args: loaded)
    with pytest.raises(ValueError, match="summary and table times"):
        audit.audit_registered_single_pilot(*_paths(), run)


@pytest.mark.parametrize(
    "defect",
    [
        "gap",
        "cycle",
        "midpoint",
        "nonfinite_time",
        "nonpositive_interval",
        "nonfinite_period",
        "period_mismatch",
        "conservation",
        "nonfinite_conservation",
        "nonmonotone_conservation",
        "resolved_duration_out_of_range",
    ],
)
def test_registered_fixed_bin_rejects_invalid_orbit_temporal_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    run = _run(tmp_path)
    loaded = _loaded(8)
    if defect == "gap":
        loaded["orbit_series"]["start_time_myr"][4] += 1.0e-4
    elif defect == "cycle":
        loaded["orbit_series"]["cycle"][4] = 5
    elif defect == "midpoint":
        loaded["orbit_series"]["mean_time_myr"][4] += 1.0e-4
    elif defect == "nonfinite_time":
        loaded["orbit_series"]["start_time_myr"][4] = np.nan
    elif defect == "nonpositive_interval":
        loaded["orbit_series"]["end_time_myr"][4] = 0.04
    elif defect == "nonfinite_period":
        loaded["orbit_series"]["orbital_period_myr"][4] = np.nan
    elif defect == "period_mismatch":
        loaded["orbit_series"]["orbital_period_myr"][4] = 0.02
    elif defect == "conservation":
        loaded["series"]["time_myr"][-1] = 0.07
    elif defect == "nonfinite_conservation":
        loaded["series"]["time_myr"][-1] = np.nan
    elif defect == "nonmonotone_conservation":
        loaded["series"] = np.array(
            [(0.0,), (0.1,), (0.09,)], dtype=[("time_myr", float)]
        )
    else:
        loaded["conservation"]["initial_spatially_resolved_duration_myr"] = 0.11
    monkeypatch.setattr(
        audit, "_require_pilot_complete",
        lambda *args, **kwargs: {
            "fdm_adapter_metadata.json": audit._sha256(
                run / "fdm_adapter_metadata.json"
            )
        },
    )
    monkeypatch.setattr(audit, "load_convergence_run", lambda *args: loaded)
    with pytest.raises(ValueError, match="summary and table times"):
        audit.audit_registered_single_pilot(*_paths(), run)


def test_registered_fixed_bin_rejects_empty_orbit_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    loaded = _loaded(1)
    loaded["orbit_series"] = loaded["orbit_series"][:0]
    loaded["orbit"]["complete_orbits"] = 0
    monkeypatch.setattr(
        audit, "_require_pilot_complete",
        lambda *args, **kwargs: {
            "fdm_adapter_metadata.json": audit._sha256(
                run / "fdm_adapter_metadata.json"
            )
        },
    )
    monkeypatch.setattr(audit, "load_convergence_run", lambda *args: loaded)
    with pytest.raises(ValueError, match="orbit table is empty"):
        audit.audit_registered_single_pilot(*_paths(), run)


def test_pilot_stage_does_not_require_wave_response_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    (run / "config.uldm").write_text("{}")
    (run / "torch_run_summary.json").write_text('{"status":"complete"}')
    (run / "conservation_summary.json").write_text(json.dumps({
        "status": "diagnosed", "case_id": "qe_q100_e030_a020",
        "resolution": 256, "duration_myr": 0.1,
    }))
    (run / "orbit_averaged_exchange_summary.json").write_text(
        '{"status":"orbit_averaged","complete_orbits":8}'
    )
    (run / "conservation_timeseries.csv").write_text("time_myr\n0\n")
    (run / "orbit_averaged_exchange.csv").write_text("start_time_myr\n0\n")
    monkeypatch.setattr(
        audit, "validate_torch_calibration_completion", lambda *args, **kwargs: ({}, {})
    )
    hashes = audit._require_pilot_complete(
        run, case_id="qe_q100_e030_a020", resolution=256, duration_myr=0.1
    )
    assert set(hashes) == set(audit.PILOT_INPUTS)
    assert not (run / "wave_response_summary.json").exists()
    assert not (run / "wave_response_timeseries.csv").exists()


def test_metadata_replacement_before_first_protected_hash_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)

    def replace_then_hash(*args, **kwargs):
        metadata_path = run / "fdm_adapter_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["run_id"] = "replacement"
        metadata_path.write_text(json.dumps(metadata))
        return {"fdm_adapter_metadata.json": audit._sha256(metadata_path)}

    monkeypatch.setattr(audit, "_require_pilot_complete", replace_then_hash)
    with pytest.raises(ValueError, match="metadata changed"):
        audit.audit_registered_single_pilot(*_paths(), run)


def test_manifest_replacement_during_run_id_lookup_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path, cases, manifest = _paths()
    copied_design = tmp_path / "design.json"
    copied_cases = tmp_path / "physical_cases.csv"
    copied_manifest = tmp_path / "run_manifest.csv"
    shutil.copyfile(design_path, copied_design)
    shutil.copyfile(cases, copied_cases)
    shutil.copyfile(manifest, copied_manifest)
    run = _run(tmp_path)
    original = audit._manifest_run_id

    def replace_after_lookup(*args, **kwargs):
        run_id = original(*args, **kwargs)
        copied_manifest.write_bytes(copied_manifest.read_bytes() + b"\n")
        return run_id

    monkeypatch.setattr(audit, "_manifest_run_id", replace_after_lookup)
    with pytest.raises(ValueError, match="manifest changed"):
        audit.audit_registered_single_pilot(
            copied_design, copied_cases, copied_manifest, run
        )


@pytest.mark.parametrize("identity", ["missing", "tampered"])
def test_registered_fixed_bin_rejects_missing_or_tampered_identity(
    tmp_path: Path, identity: str
) -> None:
    run = _run(tmp_path, identity=identity)
    with pytest.raises(ValueError, match="registered design identity"):
        audit.audit_registered_single_pilot(*_paths(), run)


def test_doubled_box_rejects_directory_run_id_mismatch(tmp_path: Path) -> None:
    design_path, cases, manifest = _paths()
    design, file_hash = read_verified_qe_followup_design(
        design_path, physical_cases=cases, run_manifest=manifest
    )
    run = tmp_path / "copied_doubled_box_run"
    run.mkdir()
    (run / "fdm_adapter_metadata.json").write_text(json.dumps({
        "run_id": "qe_q100_e030_a020_n768",
        "case_id": design["case_id"],
        "resolution": 768,
        "box_size_pc": 52.8,
        "duration_myr": 0.1,
        "qe_design_binding": {
            "status": "qe_prospective_design_bound_not_a_calibration_release",
            "design_sha256": design["design_sha256"],
            "file_sha256": file_hash,
            "role": "doubled_box_control",
        },
    }))
    with pytest.raises(ValueError, match="run ID disagrees"):
        audit.audit_registered_single_pilot(design_path, cases, manifest, run)
