"""Fail-closed common-time comparison of the q/e startup diagnostics."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.analyze_qe_startup_component_dt import (
    _COMPONENTS,
    summarize,
)


def _write_fixture(root: Path) -> None:
    for label, factor, steps in (("f100", 1.0, 1), ("f050", 0.5, 2), ("f025", 0.25, 4)):
        run = root / label
        run.mkdir(parents=True, exist_ok=True)
        metadata = {
            "case_id": "qe_q030_e030_a020",
            "resolution": 256,
            "backend": "pytorch_cuda",
            "time_step_factor": factor,
            "save_number": 58500,
            "actual_wave_steps": 58500 * steps,
            "diagnostic_stop_after_save": 10,
            "analytic_fdm_drag": False,
            "duration_myr": 0.1,
            "wave_time_step_code": 0.1 / (58500 * steps),
            "nbody_rk4_substeps_per_wave_step": 9,
            "box_size_pc": 26.4,
            "particle_mass_ev": 1e-21,
            "pyul_length_unit_m": 1.0,
            "pyul_time_unit_s": 1.0,
            "pyul_mass_unit_kg": 1.0,
            "pyul_energy_unit_j": 1.0,
            "adapter_revision": "adapter",
            "pyul_revision": "pyul",
            "qe_design_binding": {"design_sha256": "design"},
            "live_wave_force_on_smbhs": True,
            "smbh_force_on_live_wave": True,
            "reference_initial_wave_sha256": "a" * 64,
            "reference_initial_particle_sha256": "b" * 64,
        }
        metadata_path = run / "fdm_adapter_metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        config_path = run / "config.uldm"
        config_path.write_text(json.dumps({"Matter Particles": {
            "Condition": [[1.0], [0.3]], "Plummer Radius": 0.05,
        }}))
        (run / "torch_run_summary.json").write_text(json.dumps({
            "status": "diagnostic_partial",
            "saved_intervals": 10,
            "actual_wave_steps": 10 * steps,
            "planned_wave_steps": 58500 * steps,
            "duration_myr": 0.1 * 10 / 58500,
        }))
        (run / "conservation_summary.json").write_text(json.dumps({
            "status": "diagnostic_partial",
            "samples": 11,
            "case_id": "qe_q030_e030_a020",
            "resolution": 256,
            "duration_myr": 0.1,
            "initial_spatially_resolved_samples": 11,
            "maximum_initial_resolved_energy_error_over_transfer": 0.012,
            "maximum_energy_error_over_transfer_tolerance": 0.01,
            "initial_resolved_energy_conservation_passed": False,
        }))
        source_paths = (
            "scripts/run_torch_wave_case.py",
            "src/fdm_smbh_delay/torch_wave.py",
            "src/fdm_smbh_delay/pyul.py",
        )
        sources = []
        for relative in source_paths:
            source = run / "torch_solver_provenance/source" / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"frozen numerical source")
            sources.append({
                "path": relative,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            })
        (run / "torch_solver_provenance/manifest.json").write_text(json.dumps({
            "status": "source_snapshot",
            "run": str(run),
            "input_records": {
                "fdm_adapter_metadata_sha256": hashlib.sha256(
                    metadata_path.read_bytes()
                ).hexdigest(),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            },
            "source_files": sources,
        }))
        index = np.arange(11, dtype=float)
        components = (
            100.0 + index,
            -30.0 - 0.2 * index,
            -50.0 - 0.4 * index,
            20.0 + 0.3 * index,
            -10.0 - 0.1 * index,
        )
        total = sum(components)
        wave_intrinsic = components[0] + components[1]
        binary_orbital = 50.0 * index
        bh_com_kinetic = np.zeros_like(index)
        np.savetxt(
            run / "conservation_timeseries.csv",
            np.column_stack((0.1 * index / 58500, *components, total,
                             binary_orbital, bh_com_kinetic, wave_intrinsic,
                             np.where(index > 0, 0.012, 0.0))),
            delimiter=",",
            header=",".join(("time_myr", *_COMPONENTS, "combined_energy",
                             "binary_orbital_energy", "bh_com_kinetic_energy",
                             "wave_intrinsic_energy", "energy_error_over_transfer")),
            comments="",
        )


def test_common_time_components_require_matching_launch_and_source(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    result = summarize(tmp_path)
    assert result["status"] == "startup_component_diagnostic_not_a_calibration_release"
    assert [run["wave_steps_per_saved_interval"] for run in result["runs"]] == [1, 2, 4]
    assert result["runs"][0]["change_from_initial"]["combined_energy"][10] == pytest.approx(6.0)
    metadata_path = tmp_path / "f025/fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["reference_initial_wave_sha256"] = "c" * 64
    metadata_path.write_text(json.dumps(metadata))
    provenance_path = tmp_path / "f025/torch_solver_provenance/manifest.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["input_records"]["fdm_adapter_metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="identical launch-time states"):
        summarize(tmp_path)


def test_common_time_components_reject_source_or_time_tampering(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    source = tmp_path / "f050/torch_solver_provenance/source/scripts/run_torch_wave_case.py"
    source.write_bytes(b"altered")
    with pytest.raises(ValueError, match="frozen solver source hash differs"):
        summarize(tmp_path)
    source.write_bytes(b"frozen numerical source")
    path = tmp_path / "f050/conservation_timeseries.csv"
    data = np.genfromtxt(path, delimiter=",", names=True)
    data["time_myr"][5] += 1.0e-4
    np.savetxt(
        path,
        np.column_stack([data[name] for name in data.dtype.names or ()]),
        delimiter=",",
        header=",".join(data.dtype.names or ()),
        comments="",
    )
    with pytest.raises(ValueError, match="saved times are not aligned"):
        summarize(tmp_path)


def test_rejects_manifest_escape_and_component_or_peak_tampering(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    manifest_path = tmp_path / "f025/torch_solver_provenance/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_files"][0]["path"] = "../outside.py"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes snapshot"):
        summarize(tmp_path)

    _write_fixture(tmp_path)
    csv_path = tmp_path / "f050/conservation_timeseries.csv"
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    data["combined_energy"][4] += 1.0
    np.savetxt(csv_path, np.column_stack([data[name] for name in data.dtype.names or ()]),
               delimiter=",", header=",".join(data.dtype.names or ()), comments="")
    with pytest.raises(ValueError, match="do not sum to total"):
        summarize(tmp_path)

    _write_fixture(tmp_path)
    summary_path = tmp_path / "f100/conservation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["maximum_initial_resolved_energy_error_over_transfer"] = 0.02
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="does not match timeseries"):
        summarize(tmp_path)


def test_rejects_incomplete_and_refuses_existing_output(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    summary_path = tmp_path / "f100/torch_run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "complete"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="incomplete"):
        summarize(tmp_path)

    summary["status"] = "diagnostic_partial"
    summary_path.write_text(json.dumps(summary))
    output = tmp_path / "existing.json"
    output.write_text("preserve me")
    result = subprocess.run(
        [sys.executable, "scripts/analyze_qe_startup_component_dt.py", str(tmp_path),
         "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "refusing to replace" in result.stderr
    assert output.read_text() == "preserve me"


def test_rejects_different_physics_with_identical_initial_hashes(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    metadata_path = tmp_path / "f050/fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["pyul_mass_unit_kg"] = 2.0
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = tmp_path / "f050/torch_solver_provenance/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_records"]["fdm_adapter_metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identical physical setup"):
        summarize(tmp_path)

    _write_fixture(tmp_path)
    config_path = tmp_path / "f050/config.uldm"
    config = json.loads(config_path.read_text())
    config["Matter Particles"]["Plummer Radius"] = 0.06
    config_path.write_text(json.dumps(config))
    manifest = json.loads(manifest_path.read_text())
    manifest["input_records"]["config_sha256"] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identical physical setup"):
        summarize(tmp_path)


def test_rejects_falsified_energy_error_history(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    csv_path = tmp_path / "f025/conservation_timeseries.csv"
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    data["energy_error_over_transfer"][5] = 0.0
    np.savetxt(csv_path, np.column_stack([data[name] for name in data.dtype.names or ()]),
               delimiter=",", header=",".join(data.dtype.names or ()), comments="")
    with pytest.raises(ValueError, match="history differs from components"):
        summarize(tmp_path)


def test_writes_new_analysis_once(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    output = tmp_path / "analysis.json"
    result = subprocess.run(
        [sys.executable, "scripts/analyze_qe_startup_component_dt.py", str(tmp_path),
         "--output", str(output)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["status"] == (
        "startup_component_diagnostic_not_a_calibration_release"
    )
    assert not list(tmp_path.glob(".analysis.json.*.tmp"))
