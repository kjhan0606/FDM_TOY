#!/usr/bin/env python3
"""Fail-closed time-step audit of TSC Strang momentum nonconservation."""

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
else:
    from analyze_qe_startup_component_dt import _sha256


_LEVELS = (("f100", 1.0, 10), ("f050", 0.5, 20),
           ("f025", 0.25, 40), ("f0125", 0.125, 80))
_MOMENTUM_TOLERANCE = 1e-3


def _vector(record: dict, name: str) -> np.ndarray:
    value = np.asarray(record.get(name), dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"momentum audit has an invalid {name}")
    return value


def _pair_orders(differences: list[np.ndarray]) -> list[float]:
    norms = [float(np.linalg.norm(value)) for value in differences]
    if any(not np.isfinite(value) or value <= 0.0 for value in norms):
        raise ValueError("momentum refinement differences are unresolved")
    return [math.log2(norms[index] / norms[index + 1]) for index in (0, 1)]


def summarize(root: Path, trace_summary: Path) -> dict:
    directory = root.expanduser().resolve()
    summary_path = trace_summary.expanduser().resolve()
    trace = json.loads(summary_path.read_text())
    if (
        trace.get("status")
        != "periodic_tsc_strang_short_prefix_diagnostic_not_a_calibration_release"
        or trace.get("trace_root") != str(directory)
        or trace.get("calibration_eligible") is not False
        or not isinstance(trace.get("runs"), list)
        or len(trace["runs"]) != len(_LEVELS)
    ):
        raise ValueError("Strang trace summary is not a complete diagnostic")
    analysis_hash = _sha256(
        Path(__file__).resolve().with_name("audit_periodic_tsc_momentum.py")
    )
    records = []
    file_hashes = []
    for index, (label, factor, stop) in enumerate(_LEVELS):
        run = trace["runs"][index]
        path = directory / f"momentum_{label}_v2.json"
        record = json.loads(path.read_text())
        if (
            run.get("run") != str(directory / label)
            or run.get("time_step_factor") != factor
            or record.get("status")
            != "periodic_tsc_strang_momentum_diagnostic_not_a_calibration_release"
            or record.get("run") != run["run"]
            or record.get("factor") != factor
            or record.get("last_wave_step") != stop
            or record.get("slurm_job_id") != run.get("slurm_job_id")
            or record.get("source_commit") != run.get("source_commit")
            or record.get("source_fingerprints") != run.get("source_fingerprints")
            or record.get("initial_wave_sha256") != run["launch_hashes"][0]
            or record.get("initial_body_sha256") != run["launch_hashes"][1]
            or record.get("conservation_summary_sha256")
            != run.get("conservation_summary_sha256")
            or record.get("conservation_timeseries_sha256")
            != run.get("conservation_timeseries_sha256")
            or record.get("provenance_manifest_sha256")
            != run.get("provenance_manifest_sha256")
            or record.get("analysis_source_sha256") != analysis_hash
            or record.get("calibration_eligible") is not False
            or record.get("momentum_change_resolved_above_rounding_floor") is not True
        ):
            raise ValueError(f"{path}: momentum record or provenance differs")
        total = _vector(record, "total_momentum_change_code")
        wave = (
            _vector(record, "wave_momentum_final_code")
            - _vector(record, "wave_momentum_initial_code")
        )
        body = (
            _vector(record, "body_momentum_final_code")
            - _vector(record, "body_momentum_initial_code")
        )
        if not np.allclose(total, wave + body, rtol=1e-12, atol=1e-7):
            raise ValueError(f"{path}: wave and binary momentum ledger does not close")
        records.append(record)
        file_hashes.append(_sha256(path))
    totals = [_vector(record, "total_momentum_change_code") for record in records]
    waves = [
        _vector(record, "wave_momentum_final_code")
        - _vector(record, "wave_momentum_initial_code")
        for record in records
    ]
    bodies = [
        _vector(record, "body_momentum_final_code")
        - _vector(record, "body_momentum_initial_code")
        for record in records
    ]
    differences = [totals[index] - totals[index + 1] for index in range(3)]
    observed_orders = _pair_orders(differences)
    limit = (4.0 * totals[-1] - totals[-2]) / 3.0
    previous_limit = (4.0 * totals[-2] - totals[-3]) / 3.0
    wave_limit = (4.0 * waves[-1] - waves[-2]) / 3.0
    body_limit = (4.0 * bodies[-1] - bodies[-2]) / 3.0
    exchange = max(np.linalg.norm(wave_limit), np.linalg.norm(body_limit))
    if exchange <= 0.0:
        raise ValueError("momentum exchange is unresolved")
    ratio = float(np.linalg.norm(limit) / exchange)
    return {
        "status": "strang_momentum_limit_fails_registered_diagnostic_tolerance"
        if ratio > _MOMENTUM_TOLERANCE
        else "strang_momentum_limit_within_registered_diagnostic_tolerance",
        "scope": "n256 q030/e030 first ten common physical saves only",
        "calibration_eligible": False,
        "trace_root": str(directory),
        "trace_summary_sha256": _sha256(summary_path),
        "momentum_audit_sha256": file_hashes,
        "analysis_source_sha256": _sha256(Path(__file__).resolve()),
        "registered_diagnostic_momentum_tolerance": _MOMENTUM_TOLERANCE,
        "momentum_limit_over_exchange": ratio,
        "momentum_diagnostic_passed": bool(ratio <= _MOMENTUM_TOLERANCE),
        "total_momentum_limit_code": limit.tolist(),
        "wave_momentum_limit_code": wave_limit.tolist(),
        "body_momentum_limit_code": body_limit.tolist(),
        "richardson_limit_difference_code": (limit - previous_limit).tolist(),
        "total_momentum_successive_differences_code": [
            item.tolist() for item in differences
        ],
        "total_momentum_observed_pair_orders": observed_orders,
        "limit_assumption": (
            "quadratic extrapolation from the two finest short-prefix steps; "
            "not a proof of a long-time or spatial limit"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--trace-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace momentum refinement: {output}")
    payload = summarize(args.trace_root, args.trace_summary)
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
