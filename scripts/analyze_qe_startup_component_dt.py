#!/usr/bin/env python3
"""Compare launch-hashed q/e startup energy components at identical times."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


_COMPONENTS = (
    "wave_kinetic_energy",
    "wave_self_gravity_energy",
    "wave_bh_interaction_grid",
    "bh_total_kinetic_energy",
    "bh_mutual_gravity_energy",
)
_RUNS = (("f100", 1.0, 1), ("f050", 0.5, 2), ("f025", 0.25, 4))
_SAVED_INTERVALS = 58500
_DIAGNOSTIC_STOPS = 10
_PHYSICAL_METADATA = (
    "box_size_pc", "particle_mass_ev", "pyul_length_unit_m",
    "pyul_time_unit_s", "pyul_mass_unit_kg", "pyul_energy_unit_j",
    "nbody_rk4_substeps_per_wave_step", "adapter_revision", "pyul_revision",
    "qe_design_binding", "live_wave_force_on_smbhs", "smbh_force_on_live_wave",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_component_changes(series: np.ndarray) -> dict[str, list[float]]:
    if series.ndim != 1 or series.size != _DIAGNOSTIC_STOPS + 1:
        raise ValueError("startup diagnostic needs exactly eleven energy states")
    if series.dtype.names is None or any(
        field not in series.dtype.names
        for field in (*_COMPONENTS, "combined_energy", "time_myr", "energy_error_over_transfer")
    ):
        raise ValueError("startup energy series lacks Hamiltonian components")
    components = [np.asarray(series[field], dtype=float) for field in _COMPONENTS]
    total = np.asarray(series["combined_energy"], dtype=float)
    if not all(np.all(np.isfinite(item)) for item in (*components, total)):
        raise ValueError("startup Hamiltonian components are not finite")
    scale = np.maximum.reduce([np.abs(component) for component in components])
    if np.any(np.abs(sum(components) - total) > 1.0e-12 * np.maximum(scale, 1.0)):
        raise ValueError("startup Hamiltonian components do not sum to total")
    return {
        field: (np.asarray(series[field], dtype=float) - float(series[field][0])).tolist()
        for field in (*_COMPONENTS, "combined_energy")
    }


def _read_run(root: Path, label: str, factor: float, steps_per_save: int) -> dict:
    run = root / label
    metadata_path = run / "fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    config = json.loads((run / "config.uldm").read_text())
    summary = json.loads((run / "torch_run_summary.json").read_text())
    conservation = json.loads((run / "conservation_summary.json").read_text())
    provenance = json.loads((run / "torch_solver_provenance/manifest.json").read_text())
    expected = {
        "case_id": "qe_q030_e030_a020",
        "resolution": 256,
        "backend": "pytorch_cuda",
        "time_step_factor": factor,
        "save_number": _SAVED_INTERVALS,
        "actual_wave_steps": _SAVED_INTERVALS * steps_per_save,
        "diagnostic_stop_after_save": _DIAGNOSTIC_STOPS,
        "analytic_fdm_drag": False,
        "nbody_rk4_substeps_per_wave_step": 9,
    }
    if any(metadata.get(key) != value for key, value in expected.items()) or (
        type(metadata.get("analytic_fdm_drag")) is not bool
        or type(metadata.get("resolution")) is not int
        or type(metadata.get("save_number")) is not int
        or type(metadata.get("actual_wave_steps")) is not int
        or type(metadata.get("diagnostic_stop_after_save")) is not int
        or type(metadata.get("time_step_factor")) is not float
    ):
        raise ValueError(f"{label}: Torch metadata differs from diagnostic design")
    if (
        summary.get("status") != "diagnostic_partial"
        or summary.get("saved_intervals") != _DIAGNOSTIC_STOPS
        or summary.get("actual_wave_steps") != _DIAGNOSTIC_STOPS * steps_per_save
        or summary.get("planned_wave_steps") != _SAVED_INTERVALS * steps_per_save
        or conservation.get("status") != "diagnostic_partial"
        or conservation.get("samples") != _DIAGNOSTIC_STOPS + 1
    ):
        raise ValueError(f"{label}: diagnostic evolution or analysis is incomplete")
    launch_hashes = (
        metadata.get("reference_initial_wave_sha256"),
        metadata.get("reference_initial_particle_sha256"),
    )
    if any(
        not isinstance(value, str) or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in launch_hashes
    ):
        raise ValueError(f"{label}: launch-time initial-content hashes are missing")
    inputs = provenance.get("input_records")
    if (
        provenance.get("status") != "source_snapshot"
        or provenance.get("run") != str(run)
        or not isinstance(inputs, dict)
        or inputs.get("fdm_adapter_metadata_sha256")
        != _sha256(metadata_path)
        or inputs.get("config_sha256") != _sha256(run / "config.uldm")
    ):
        raise ValueError(f"{label}: solver source snapshot is unverified")
    source_files = provenance.get("source_files")
    if not isinstance(source_files, list) or not source_files or any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("sha256"), str)
        for item in source_files
    ):
        raise ValueError(f"{label}: solver source snapshot is invalid")
    sources = tuple(sorted((item["path"], item["sha256"]) for item in source_files))
    for relative, expected_hash in sources:
        relative_path = Path(relative)
        source_root = run / "torch_solver_provenance/source"
        if (
            not relative_path.parts or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(f"{label}: solver source path escapes snapshot")
        source = source_root / relative_path
        if not source.resolve().is_relative_to(source_root.resolve()):
            raise ValueError(f"{label}: solver source path escapes snapshot")
        if _sha256(source) != expected_hash:
            raise ValueError(f"{label}: frozen solver source hash differs")
    required_sources = {
        "scripts/run_torch_wave_case.py",
        "src/fdm_smbh_delay/torch_wave.py",
        "src/fdm_smbh_delay/pyul.py",
    }
    if not required_sources.issubset({name for name, _ in sources}):
        raise ValueError(f"{label}: solver source snapshot lacks numerical source")
    series_path = run / "conservation_timeseries.csv"
    series = np.genfromtxt(series_path, delimiter=",", names=True)
    changes = _validated_component_changes(series)
    try:
        duration = float(metadata["duration_myr"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: diagnostic duration is invalid") from error
    time = np.asarray(series["time_myr"], dtype=float)
    expected_time = duration * np.arange(_DIAGNOSTIC_STOPS + 1) / _SAVED_INTERVALS
    if (
        not np.isclose(duration, 0.1, rtol=0.0, atol=1.0e-12)
        or not np.allclose(time, expected_time, rtol=1.0e-12, atol=0.0)
    ):
        raise ValueError(f"{label}: diagnostic saved times are not aligned")
    if (
        not np.isclose(float(summary.get("duration_myr", np.nan)),
                       float(time[-1]), rtol=1.0e-12, atol=0.0)
        or conservation.get("case_id") != metadata["case_id"]
        or conservation.get("resolution") != metadata["resolution"]
        or not np.isclose(float(conservation.get("duration_myr", np.nan)),
                          duration, rtol=1.0e-12, atol=0.0)
    ):
        raise ValueError(f"{label}: diagnostic summary identity differs")
    if any(key not in metadata for key in _PHYSICAL_METADATA):
        raise ValueError(f"{label}: physical metadata is incomplete")
    try:
        matter = config["Matter Particles"]
        wave_step = float(metadata["wave_time_step_code"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: physical configuration is incomplete") from error
    if not np.isfinite(wave_step) or wave_step <= 0.0:
        raise ValueError(f"{label}: wave time step is invalid")
    tolerance = conservation.get("maximum_energy_error_over_transfer_tolerance")
    peak = conservation.get("maximum_initial_resolved_energy_error_over_transfer")
    resolved_samples = conservation.get("initial_spatially_resolved_samples")
    if (
        not isinstance(tolerance, (int, float))
        or not np.isclose(float(tolerance), 0.01, rtol=0.0, atol=1.0e-12)
        or not isinstance(peak, (int, float)) or not np.isfinite(peak)
        or peak < 0.0
        or type(resolved_samples) is not int
        or not 2 <= resolved_samples <= len(time)
    ):
        raise ValueError(f"{label}: registered conservation tolerance or peak is invalid")
    sampled_error = np.asarray(series["energy_error_over_transfer"], dtype=float)
    if (
        not np.all(np.isfinite(sampled_error))
        or np.any(sampled_error < 0.0)
        or not np.isclose(
            float(peak), float(np.max(sampled_error[:resolved_samples])),
            rtol=1.0e-10, atol=1.0e-12,
        )
        or conservation.get("initial_resolved_energy_conservation_passed")
        is not bool(float(peak) <= float(tolerance))
    ):
        raise ValueError(f"{label}: conservation summary does not match timeseries")
    return {
        "label": label,
        "time_step_factor": factor,
        "wave_steps_per_saved_interval": steps_per_save,
        "launch_hashes": launch_hashes,
        "physical_metadata": {key: metadata[key] for key in _PHYSICAL_METADATA},
        "matter_particles": matter,
        "wave_time_step_code": wave_step,
        "source_fingerprints": sources,
        "conservation_summary_sha256": _sha256(run / "conservation_summary.json"),
        "conservation_timeseries_sha256": _sha256(series_path),
        "torch_run_summary_sha256": _sha256(run / "torch_run_summary.json"),
        "fdm_adapter_metadata_sha256": _sha256(metadata_path),
        "time_myr": time.tolist(),
        "change_from_initial": changes,
        "maximum_sampled_prefix_error_over_transfer": float(peak),
        "registered_energy_error_tolerance": float(tolerance),
    }


def summarize(root: Path) -> dict:
    runs = [_read_run(root, *arguments) for arguments in _RUNS]
    if len({item["launch_hashes"] for item in runs}) != 1:
        raise ValueError("startup runs do not share identical launch-time states")
    if len({item["source_fingerprints"] for item in runs}) != 1:
        raise ValueError("startup runs did not use identical numerical source")
    if any(
        item["physical_metadata"] != runs[0]["physical_metadata"]
        or item["matter_particles"] != runs[0]["matter_particles"]
        for item in runs[1:]
    ):
        raise ValueError("startup runs do not share identical physical setup")
    if any(
        not np.isclose(item["wave_time_step_code"],
                       runs[0]["wave_time_step_code"] * item["time_step_factor"],
                       rtol=1.0e-12, atol=0.0)
        for item in runs[1:]
    ):
        raise ValueError("startup wave time steps do not follow requested factors")
    if any(
        not np.allclose(item["time_myr"], runs[0]["time_myr"], rtol=1.0e-12, atol=0.0)
        for item in runs[1:]
    ):
        raise ValueError("startup runs do not share the same physical save times")
    return {
        "status": "startup_component_diagnostic_not_a_calibration_release",
        "root": str(root),
        "case_id": "qe_q030_e030_a020",
        "resolution": 256,
        "registered_tolerance_unchanged": runs[0]["registered_energy_error_tolerance"],
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    payload = summarize(root)
    if output.exists():
        raise FileExistsError(f"refusing to replace startup analysis: {output}")
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if output.exists():
            raise FileExistsError(f"refusing to replace startup analysis: {output}")
        os.link(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
