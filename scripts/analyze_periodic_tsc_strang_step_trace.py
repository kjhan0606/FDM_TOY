#!/usr/bin/env python3
"""Source-bound short-prefix audit of joint wave--SMBH Strang TSC runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .analyze_periodic_tsc_step_trace import _RUNS, _observed_orders, _read_trace
    from .analyze_qe_startup_component_dt import _sha256
else:
    from analyze_periodic_tsc_step_trace import _RUNS, _observed_orders, _read_trace
    from analyze_qe_startup_component_dt import _sha256


def summarize(trace_root: Path) -> dict:
    root = trace_root.expanduser().resolve()
    project = Path(__file__).resolve().parents[1]
    source_paths = {
        "scripts/analyze_periodic_tsc_strang_step_trace.py": Path(__file__).resolve(),
        "scripts/analyze_periodic_tsc_step_trace.py": (
            project / "scripts/analyze_periodic_tsc_step_trace.py"
        ),
        "scripts/analyze_periodic_tsc_dt_followup.py": (
            project / "scripts/analyze_periodic_tsc_dt_followup.py"
        ),
        "scripts/analyze_qe_startup_component_dt.py": (
            project / "scripts/analyze_qe_startup_component_dt.py"
        ),
        "scripts/analyze_pyul_wave_run.py": (
            project / "scripts/analyze_pyul_wave_run.py"
        ),
    }
    read = [
        _read_trace(root / label, factor, saves, stop, coupling="periodic_tsc_strang")
        for label, factor, saves, stop in _RUNS
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
        raise ValueError("Strang traces differ in source, launch state, or physics")
    phase_jumps = []
    phase_hashes = []
    for label, _factor, _saves, _stop in _RUNS:
        path = root / label / "initial_compact_phase_jump.json"
        record = json.loads(path.read_text())
        value = record.get("max_neighbour_phase_difference_over_pi")
        if not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{path}: initial compact phase jump is invalid")
        phase_jumps.append(float(value))
        phase_hashes.append(_sha256(path))
    if any(
        not np.isclose(value, phase_jumps[0] * factor, rtol=1e-10, atol=1e-14)
        for value, (_label, factor, _saves, _stop) in zip(
            phase_jumps, _RUNS, strict=True
        )
    ):
        raise ValueError("initial compact phase jump does not scale with time step")
    first_drift = [
        item["first_common_save_hamiltonian_change_msun_pc2_myr2"]
        for item in runs
    ]
    last_drift = [
        item["last_common_save_hamiltonian_change_msun_pc2_myr2"]
        for item in runs
    ]
    separation = [item["last_common_save_separation_pc"] for item in runs]
    separation_differences = [
        separation[index] - separation[index + 1]
        for index in range(len(separation) - 1)
    ]
    source_hashes = {key: _sha256(path) for key, path in source_paths.items()}
    return {
        "status": "periodic_tsc_strang_short_prefix_diagnostic_not_a_calibration_release",
        "scope": "n256 q030/e030 first ten common physical saves only",
        "trace_root": str(root),
        "analysis_source_sha256": source_hashes,
        "registered_tolerance_unchanged": 0.01,
        "initial_compact_phase_jump_over_pi": phase_jumps,
        "initial_compact_phase_jump_sha256": phase_hashes,
        "first_common_save_drift_observed_pair_orders": _observed_orders(first_drift),
        "last_common_save_drift_observed_pair_orders": _observed_orders(last_drift),
        "last_common_save_separation_successive_differences_pc": (
            separation_differences
        ),
        "last_common_save_separation_observed_pair_orders": _observed_orders(
            separation_differences
        ),
        "observed_order_assumption": (
            "pairwise log2 of same-sign differences is a short-prefix diagnostic; "
            "neither an asymptotic limit nor a calibrated drag rate is established"
        ),
        "calibration_eligible": False,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace Strang trace audit: {output}")
    payload = summarize(args.trace_root)
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
