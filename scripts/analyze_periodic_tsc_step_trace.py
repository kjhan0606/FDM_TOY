#!/usr/bin/env python3
"""Source-bound per-wave-step conservation audit of reciprocal TSC runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .analyze_periodic_tsc_dt_followup import _launch_owner
    from .analyze_qe_startup_component_dt import _sha256
else:
    from analyze_periodic_tsc_dt_followup import _launch_owner
    from analyze_qe_startup_component_dt import _sha256


_RUNS = (
    ("f100", 1.0, 58500, 10),
    ("f050", 0.5, 117000, 20),
    ("f025", 0.25, 234000, 40),
    ("f0125", 0.125, 468000, 80),
)
_NUMERICAL_SOURCES = {
    "scripts/run_torch_wave_case.py",
    "src/fdm_smbh_delay/torch_wave.py",
    "src/fdm_smbh_delay/pyul.py",
    "src/fdm_smbh_delay/periodic_mesh_coupling.py",
}
_ENERGY_FIELDS = (
    "wave_kinetic_energy", "wave_self_gravity_energy",
    "wave_bh_interaction_grid", "bh_total_kinetic_energy",
    "bh_mutual_gravity_energy",
)
_TRANSFER_FIELDS = (
    "binary_orbital_energy", "bh_com_kinetic_energy",
    "wave_intrinsic_energy", "wave_bh_interaction_grid",
)


def _read_trace(run: Path, factor: float, saves: int, stop: int) -> tuple[dict, np.ndarray]:
    project = Path(__file__).resolve().parents[1]
    analyser = project / "scripts/analyze_pyul_wave_run.py"
    if analyser.stat().st_mtime_ns > (
        run / "conservation_summary.json"
    ).stat().st_mtime_ns:
        raise ValueError(f"{run}: conservation analyser changed after output")
    metadata_path = run / "fdm_adapter_metadata.json"
    manifest_path = run / "torch_solver_provenance/manifest.json"
    metadata = json.loads(metadata_path.read_text())
    config = json.loads((run / "config.uldm").read_text())
    manifest = json.loads(manifest_path.read_text())
    solver = json.loads((run / "torch_run_summary.json").read_text())
    conservation = json.loads((run / "conservation_summary.json").read_text())
    expected = {
        "case_id": "qe_q030_e030_a020",
        "resolution": 256,
        "backend": "pytorch_cuda",
        "wave_smbh_coupling": "periodic_tsc_reciprocal",
        "experimental_coupling_not_a_calibration_release": True,
        "time_step_factor": factor,
        "save_number": saves,
        "actual_wave_steps": saves,
        "diagnostic_stop_after_save": stop,
        "nbody_rk4_substeps_per_wave_step": 9,
        "analytic_fdm_drag": False,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{run}: step-trace metadata differs from design")
    if (
        solver.get("status") != "diagnostic_partial"
        or solver.get("actual_wave_steps") != stop
        or solver.get("saved_intervals") != stop
        or conservation.get("status") != "diagnostic_partial"
        or conservation.get("samples") != stop + 1
        or conservation.get("initial_spatially_resolved_samples") != stop + 1
        or manifest.get("status") != "source_snapshot"
        or manifest.get("run") != str(run)
        or manifest.get("input_records", {}).get("fdm_adapter_metadata_sha256")
        != _sha256(metadata_path)
        or manifest.get("input_records", {}).get("config_sha256")
        != _sha256(run / "config.uldm")
    ):
        raise ValueError(f"{run}: step trace, spatial resolution or provenance is incomplete")
    records = manifest.get("source_files")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"{run}: source snapshot is malformed")
    sources = tuple(sorted((row.get("path"), row.get("sha256")) for row in records))
    if not _NUMERICAL_SOURCES.issubset({name for name, _ in sources}):
        raise ValueError(f"{run}: source snapshot omits numerical operators")
    for relative, digest in sources:
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError(f"{run}: source record is invalid")
        relative_path = Path(relative)
        if not relative_path.parts or relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{run}: source record escapes the snapshot")
        frozen = run / "torch_solver_provenance/source" / relative_path
        if _sha256(frozen) != digest:
            raise ValueError(f"{run}: frozen source differs from manifest")
    owner = _launch_owner(run, project, sources)
    series_path = run / "conservation_timeseries.csv"
    series = np.genfromtxt(series_path, delimiter=",", names=True)
    required_fields = (
        "time_myr", "combined_energy", "energy_error_over_transfer",
        "wave_mass_msun", "separation_pc", "bh_wave_point_estimator",
        *_ENERGY_FIELDS, *_TRANSFER_FIELDS,
    )
    if (
        series.ndim != 1 or series.size != stop + 1
        or series.dtype.names is None
        or any(field not in series.dtype.names for field in required_fields)
        or any(not np.all(np.isfinite(series[field])) for field in required_fields)
    ):
        raise ValueError(f"{run}: per-step energy series is invalid")
    duration = float(metadata["duration_myr"])
    expected_time = duration * np.arange(stop + 1) / saves
    if (
        not np.isclose(duration, 0.1, rtol=0.0, atol=1e-12)
        or not np.allclose(series["time_myr"], expected_time, rtol=1e-12, atol=0.0)
        or config.get("Matter Particles") is None
    ):
        raise ValueError(f"{run}: physical time or particle configuration differs")
    components = [np.asarray(series[field], dtype=float) for field in _ENERGY_FIELDS]
    total = np.asarray(series["combined_energy"], dtype=float)
    scale = np.maximum.reduce([np.abs(component) for component in components])
    if np.any(np.abs(sum(components) - total) > 1e-12 * np.maximum(scale, 1.0)):
        raise ValueError(f"{run}: Hamiltonian components do not close")
    transfers = [np.asarray(series[field], dtype=float) for field in _TRANSFER_FIELDS]
    exchange = np.maximum.reduce([
        np.maximum.accumulate(np.abs(component - component[0]))
        for component in transfers
    ])
    drift = np.maximum.accumulate(np.abs(total - total[0]))
    ratio = np.divide(
        drift, exchange, out=np.zeros_like(drift),
        where=exchange > np.finfo(float).tiny,
    )
    if (
        not np.allclose(
            ratio, series["energy_error_over_transfer"], rtol=1e-10, atol=1e-12
        )
        or not np.isclose(
            float(conservation["maximum_initial_resolved_energy_error_over_transfer"]),
            float(np.max(ratio)), rtol=1e-10, atol=1e-12,
        )
        or not np.isclose(
            float(conservation["maximum_energy_error_over_transfer_tolerance"]),
            0.01, rtol=0.0, atol=1e-12,
        )
    ):
        raise ValueError(f"{run}: conservation peak differs from the raw energy series")
    interaction = np.asarray(series["wave_bh_interaction_grid"], dtype=float)
    point = np.asarray(series["bh_wave_point_estimator"], dtype=float)
    reciprocity = float(np.max(np.abs(interaction - point)
                               / np.maximum(np.abs(interaction), 1.0)))
    if reciprocity > 1e-11:
        raise ValueError(f"{run}: TSC cross-energy reciprocity failed")
    first_common_index = int(round(1.0 / factor))
    if stop != 10 * first_common_index:
        raise ValueError(f"{run}: trace does not cover ten common save intervals")
    return {
        "run": str(run),
        "time_step_factor": factor,
        "wave_steps_per_common_save": first_common_index,
        "source_fingerprints": sources,
        "launch_hashes": (
            metadata.get("reference_initial_wave_sha256"),
            metadata.get("reference_initial_particle_sha256"),
        ),
        "physical_configuration": config["Matter Particles"],
        "physical_metadata": {
            key: metadata.get(key) for key in (
                "box_size_pc", "particle_mass_ev", "pyul_length_unit_m",
                "pyul_time_unit_s", "pyul_mass_unit_kg", "pyul_energy_unit_j",
            )
        },
        "wave_time_step_code": float(metadata["wave_time_step_code"]),
        "conservation_summary_sha256": _sha256(run / "conservation_summary.json"),
        "conservation_timeseries_sha256": _sha256(series_path),
        "torch_run_summary_sha256": _sha256(run / "torch_run_summary.json"),
        "provenance_manifest_sha256": _sha256(manifest_path),
        "max_reciprocal_interaction_relative_error": reciprocity,
        "max_wave_mass_relative_error": float(
            np.max(np.abs(series["wave_mass_msun"] / series["wave_mass_msun"][0] - 1.0))
        ),
        "maximum_per_wave_step_prefix_error_over_transfer": float(np.max(ratio)),
        "first_step_prefix_error_over_transfer": float(ratio[1]),
        "first_common_save_prefix_error_over_transfer": float(ratio[first_common_index]),
        "first_common_save_hamiltonian_change_msun_pc2_myr2": float(
            total[first_common_index] - total[0]
        ),
        "last_common_save_hamiltonian_change_msun_pc2_myr2": float(
            total[-1] - total[0]
        ),
        "last_common_save_separation_pc": float(series["separation_pc"][-1]),
        "registered_tolerance_passed_on_per_step_trace": bool(np.max(ratio) <= 0.01),
        "calibration_eligible": False,
        **owner,
    }, series


def _observed_orders(values: list[float]) -> list[float | None]:
    orders: list[float | None] = []
    for first, second in zip(values, values[1:], strict=False):
        if first * second <= 0.0:
            orders.append(None)
        else:
            orders.append(float(math.log2(abs(first / second))))
    return orders


def summarize(reference_run: Path, trace_root: Path, common_summary: Path) -> dict:
    project = Path(__file__).resolve().parents[1]
    source_paths = {
        "scripts/analyze_periodic_tsc_step_trace.py": Path(__file__).resolve(),
        "scripts/analyze_periodic_tsc_dt_followup.py": (
            project / "scripts/analyze_periodic_tsc_dt_followup.py"
        ),
        "scripts/analyze_qe_startup_component_dt.py": (
            project / "scripts/analyze_qe_startup_component_dt.py"
        ),
        "scripts/analyze_pyul_wave_run.py": project / "scripts/analyze_pyul_wave_run.py",
    }
    source_hashes = {key: _sha256(path) for key, path in source_paths.items()}
    comparison = json.loads(common_summary.read_text())
    recorded_comparison_sources = comparison.get("analysis_source_sha256")
    if (
        comparison.get("status")
        != "periodic_tsc_temporal_diagnostic_not_a_calibration_release"
        or comparison.get("reference_run") != str(reference_run)
        or not isinstance(recorded_comparison_sources, dict)
        or any(
            source_hashes.get(key) != digest
            for key, digest in recorded_comparison_sources.items()
        )
        or not {
            "scripts/analyze_periodic_tsc_dt_followup.py",
            "scripts/analyze_qe_startup_component_dt.py",
            "scripts/analyze_pyul_wave_run.py",
        }.issubset(recorded_comparison_sources)
    ):
        raise ValueError("common-time TSC comparison is unverified")
    locations = (
        reference_run,
        trace_root / "f050",
        trace_root / "f025",
        trace_root / "f0125",
    )
    read = [
        _read_trace(path, factor, saves, stop)
        for path, (_label, factor, saves, stop) in zip(locations, _RUNS, strict=True)
    ]
    runs = [item for item, _series in read]
    if (
        len({item["source_fingerprints"] for item in runs}) != 1
        or len({item["launch_hashes"] for item in runs}) != 1
        or any(
            item["physical_configuration"] != runs[0]["physical_configuration"]
            or item["physical_metadata"] != runs[0]["physical_metadata"]
            or not np.isclose(
                item["wave_time_step_code"],
                runs[0]["wave_time_step_code"] * item["time_step_factor"],
                rtol=1e-12, atol=0.0,
            )
            for item in runs[1:]
        )
    ):
        raise ValueError("step traces differ in source, launch state, or physics")
    common_runs = comparison.get("runs")
    if not isinstance(common_runs, list) or len(common_runs) != 3:
        raise ValueError("common-time TSC comparison has incomplete runs")
    for index, common in enumerate(common_runs):
        item = runs[index]
        stride = item["wave_steps_per_common_save"]
        if (
            common.get("source_fingerprints") != [list(row) for row in item["source_fingerprints"]]
            or common.get("launch_hashes") != list(item["launch_hashes"])
            or common.get("time_step_factor") != item["time_step_factor"]
            or abs(
                item["first_common_save_hamiltonian_change_msun_pc2_myr2"]
                - common["first_save_hamiltonian_change_msun_pc2_myr2"]
            ) > 1.0
            or abs(
                item["last_common_save_hamiltonian_change_msun_pc2_myr2"]
                - common["final_save_hamiltonian_change_msun_pc2_myr2"]
            ) > 1.0
            or np.shape(read[index][1]["time_myr"][::stride])
            != np.shape(common["time_myr"])
            or not np.allclose(
                read[index][1]["time_myr"][::stride],
                common["time_myr"], rtol=1e-12, atol=0.0,
            )
        ):
            raise ValueError("per-step trace differs from matched common-time run")
    first_changes = [
        item["first_common_save_hamiltonian_change_msun_pc2_myr2"] for item in runs
    ]
    last_changes = [
        item["last_common_save_hamiltonian_change_msun_pc2_myr2"] for item in runs
    ]
    separations = [item["last_common_save_separation_pc"] for item in runs]
    separation_differences = [
        separations[index] - separations[index + 1]
        for index in range(len(separations) - 1)
    ]
    if source_hashes != {key: _sha256(path) for key, path in source_paths.items()}:
        raise ValueError("step-trace analysis source changed during comparison")
    return {
        "status": "periodic_tsc_per_wave_step_diagnostic_not_a_calibration_release",
        "scope": "n256 q030/e030 first ten common physical saves only",
        "reference_run": str(reference_run),
        "trace_root": str(trace_root),
        "common_summary_sha256": _sha256(common_summary),
        "analysis_source_sha256": source_hashes,
        "registered_tolerance_unchanged": 0.01,
        "first_common_save_drift_observed_pair_orders": _observed_orders(first_changes),
        "last_common_save_drift_observed_pair_orders": _observed_orders(last_changes),
        "observed_drift_order_assumption": (
            "pairwise log2 of same-sign drift magnitudes assumes zero "
            "time-step-limit drift; four short-startup levels cannot prove the limit"
        ),
        "last_common_save_separation_successive_differences_pc": separation_differences,
        "last_common_save_separation_observed_pair_orders": _observed_orders(
            separation_differences
        ),
        "finest_level_common_time_cross_run_match": False,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_run", type=Path)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--common-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = args.reference_run.expanduser().resolve()
    trace_root = args.trace_root.expanduser().resolve()
    common_summary = args.common_summary.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace TSC per-step audit: {output}")
    payload = summarize(reference, trace_root, common_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
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
        os.link(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
