#!/usr/bin/env python3
"""Compare two source-bound short-prefix Strang force diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .analyze_qe_startup_component_dt import _sha256
    from .analyze_strang_momentum_refinement import summarize as old_summarize
    from .audit_periodic_tsc_strang_momentum_candidate import _LEVELS
else:
    from analyze_qe_startup_component_dt import _sha256
    from analyze_strang_momentum_refinement import summarize as old_summarize
    from audit_periodic_tsc_strang_momentum_candidate import _LEVELS


def _vector(record: dict, key: str) -> np.ndarray:
    value = np.asarray(record.get(key), dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"candidate has invalid {key}")
    return value


def _pair_orders(values: list[np.ndarray]) -> list[float]:
    differences = [
        float(np.linalg.norm(values[index] - values[index + 1]))
        for index in range(3)
    ]
    if any(not np.isfinite(value) or value <= 0.0 for value in differences):
        raise ValueError("candidate time-step differences are unresolved")
    return [math.log2(differences[index] / differences[index + 1])
            for index in (0, 1)]


def summarize(candidate_root: Path, reference_root: Path,
              reference_trace_summary: Path) -> dict:
    new_root = candidate_root.expanduser().resolve()
    old_root = reference_root.expanduser().resolve()
    old_trace_path = reference_trace_summary.expanduser().resolve()
    old_limit = old_summarize(old_root, old_trace_path)
    old_stored_path = old_root / "momentum_refinement_summary_v1.json"
    if json.loads(old_stored_path.read_text()) != old_limit:
        raise ValueError("stored reference momentum refinement differs")
    old_trace = json.loads(old_trace_path.read_text())
    if (
        old_limit.get("status")
        != "strang_momentum_limit_fails_registered_diagnostic_tolerance"
        or old_limit.get("calibration_eligible") is not False
    ):
        raise ValueError("reference Strang momentum diagnosis differs")
    project = Path(__file__).resolve().parents[1]
    audit_hash = _sha256(
        project / "scripts/audit_periodic_tsc_strang_momentum_candidate.py"
    )
    records = []
    hashes = []
    totals = []
    waves = []
    bodies = []
    energies = []
    separations = []
    for index, (label, (factor, _saves, stop)) in enumerate(_LEVELS.items()):
        audit_path = new_root / f"momentum_{label}_v1.json"
        run = new_root / label
        record = json.loads(audit_path.read_text())
        metadata = json.loads((run / "fdm_adapter_metadata.json").read_text())
        config = json.loads((run / "config.uldm").read_text())
        reference = old_trace["runs"][index]
        conservation_path = run / "conservation_summary.json"
        conservation = json.loads(conservation_path.read_text())
        if (
            record.get("status")
            != "experimental_spectral_momentum_short_prefix_not_a_calibration_release"
            or record.get("calibration_eligible") is not False
            or record.get("run") != str(run)
            or record.get("factor") != factor
            or record.get("last_wave_step") != stop
            or record.get("slurm_job_id") != "412175"
            or record.get("source_commit")
            != "d6709d2368ddd56ade596c6cbe79e28b6a749d15"
            or record.get("launch_owner_sha256") != _sha256(new_root / "OWNER.txt")
            or record.get("initial_wave_sha256") != reference["launch_hashes"][0]
            or record.get("initial_body_sha256") != reference["launch_hashes"][1]
            or metadata.get("case_id") != "qe_q030_e030_a020"
            or metadata.get("resolution") != 256
            or not np.isclose(metadata.get("duration_myr", np.nan), 0.1,
                              rtol=0, atol=1e-12)
            or config.get("Matter Particles") != reference["physical_configuration"]
            or any(
                metadata.get(key) != value
                for key, value in reference["physical_metadata"].items()
            )
            or record.get("analysis_source_sha256") != audit_hash
            or record.get("conservation_summary_sha256")
            != _sha256(conservation_path)
            or record.get("conservation_timeseries_sha256")
            != _sha256(run / "conservation_timeseries.csv")
            or record.get("provenance_manifest_sha256")
            != _sha256(run / "torch_solver_provenance/manifest.json")
            or record.get("solver_summary_sha256")
            != _sha256(run / "torch_run_summary.json")
            or conservation.get("status") != "diagnostic_partial"
            or conservation.get("smbh_force_is_interaction_energy_gradient") is not False
        ):
            raise ValueError(f"{audit_path}: candidate or provenance differs")
        wave = _vector(record, "wave_momentum_change_code")
        body = _vector(record, "body_momentum_change_code")
        total = _vector(record, "total_momentum_change_code")
        if not np.allclose(total, wave + body, rtol=0, atol=1e-10):
            raise ValueError(f"{audit_path}: momentum ledger does not close")
        exchange = max(np.linalg.norm(wave), np.linalg.norm(body))
        ratio = float(np.linalg.norm(total) / exchange)
        series = np.genfromtxt(
            run / "conservation_timeseries.csv", delimiter=",", names=True
        )
        transfer_fields = (
            "binary_orbital_energy", "bh_com_kinetic_energy",
            "wave_intrinsic_energy", "wave_bh_interaction_grid",
        )
        if (
            series.ndim != 1 or series.size != stop + 1
            or series.dtype.names is None
            or any(field not in series.dtype.names for field in (
                "combined_energy", "energy_error_over_transfer",
                *transfer_fields,
            ))
        ):
            raise ValueError(f"{audit_path}: candidate energy series is incomplete")
        energy = np.asarray(series["combined_energy"], dtype=float)
        transfer = [np.asarray(series[field], dtype=float)
                    for field in transfer_fields]
        exchange_energy = np.maximum.reduce([
            np.maximum.accumulate(np.abs(item - item[0]))
            for item in transfer
        ])
        drift_energy = np.maximum.accumulate(np.abs(energy - energy[0]))
        energy_ratio = np.divide(
            drift_energy, exchange_energy,
            out=np.zeros_like(drift_energy),
            where=exchange_energy > np.finfo(float).tiny,
        )
        if (
            not np.isfinite(ratio)
            or not np.isclose(
                ratio, record.get("momentum_change_over_exchange"),
                rtol=1e-12, atol=0,
            )
            or not np.isclose(
                record.get("max_energy_error_over_transfer"),
                conservation.get("max_total_energy_drift_over_energy_transfer"),
                rtol=1e-12, atol=0,
            )
            or not np.allclose(
                energy_ratio, series["energy_error_over_transfer"],
                rtol=1e-10, atol=1e-12,
            )
            or not np.isclose(
                np.max(energy_ratio), record["max_energy_error_over_transfer"],
                rtol=1e-10, atol=1e-12,
            )
        ):
            raise ValueError(f"{audit_path}: diagnostic ratios differ")
        records.append(record)
        hashes.append(_sha256(audit_path))
        totals.append(total)
        waves.append(wave)
        bodies.append(body)
        energies.append(float(record["max_energy_error_over_transfer"]))
        separations.append(float(conservation["final_separation_pc"]))
    limit = (4.0 * totals[-1] - totals[-2]) / 3.0
    wave_limit = (4.0 * waves[-1] - waves[-2]) / 3.0
    body_limit = (4.0 * bodies[-1] - bodies[-2]) / 3.0
    exchange_limit = max(np.linalg.norm(wave_limit), np.linalg.norm(body_limit))
    if exchange_limit <= 0.0:
        raise ValueError("candidate limit exchange is unresolved")
    new_ratio = float(np.linalg.norm(limit) / exchange_limit)
    old_ratio = float(old_limit["momentum_limit_over_exchange"])
    if not all(np.isfinite(value) for value in (*energies, *separations, new_ratio)):
        raise ValueError("candidate diagnostic is non-finite")
    return {
        "status": "short_prefix_spectral_momentum_improvement_not_a_calibration_release",
        "scope": "n256 q030/e030 first ten common physical saves only",
        "calibration_eligible": False,
        "candidate_root": str(new_root),
        "reference_root": str(old_root),
        "reference_trace_summary_sha256": _sha256(old_trace_path),
        "reference_momentum_summary_sha256": _sha256(old_stored_path),
        "candidate_audit_sha256": hashes,
        "analysis_source_sha256": _sha256(Path(__file__).resolve()),
        "time_step_factors": [item[0] for item in _LEVELS.values()],
        "candidate_endpoint_momentum_over_exchange": [
            item["momentum_change_over_exchange"] for item in records
        ],
        "candidate_momentum_limit_code": limit.tolist(),
        "candidate_momentum_limit_over_exchange": new_ratio,
        "reference_momentum_limit_over_exchange": old_ratio,
        "momentum_limit_improvement_factor": old_ratio / new_ratio,
        "candidate_total_momentum_pair_orders": _pair_orders(totals),
        "candidate_max_energy_error_over_transfer": energies,
        "reference_max_energy_error_over_transfer": [
            item["maximum_per_wave_step_prefix_error_over_transfer"]
            for item in old_trace["runs"]
        ],
        "candidate_last_common_save_separation_pc": separations,
        "reference_last_common_save_separation_pc": [
            item["last_common_save_separation_pc"]
            for item in old_trace["runs"]
        ],
        "candidate_peak_device_memory_bytes": max(
            item["peak_device_memory_bytes"] for item in records
        ),
        "registered_momentum_tolerance": 1e-3,
        "registered_energy_tolerance": 1e-2,
        "candidate_momentum_diagnostic_passed": bool(new_ratio <= 1e-3),
        "candidate_energy_diagnostic_passed_each_factor": [
            bool(value <= 1e-2) for value in energies
        ],
        "limit_assumption": (
            "quadratic extrapolation from the two finest short-prefix steps; "
            "not a spatial or long-time conservation result"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--reference-trace-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace comparison: {output}")
    payload = summarize(
        args.candidate_root, args.reference_root, args.reference_trace_summary
    )
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
