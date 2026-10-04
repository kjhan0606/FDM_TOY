#!/usr/bin/env python3
"""CPU re-audit of one immutable n256 wave-only GPU null-control endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

if __package__:
    from .analyze_qe_startup_component_dt import _sha256
    from .audit_periodic_tsc_momentum import spectral_wave_diagnostics
else:
    from analyze_qe_startup_component_dt import _sha256
    from audit_periodic_tsc_momentum import spectral_wave_diagnostics

from fdm_smbh_delay.pyul import pyul_unit_system


_LEVELS = {"f100": 10, "f050": 20, "f025": 40, "f0125": 80}
_SOURCE_PATHS = {
    "scripts/probe_wave_only_momentum_control.py",
    "src/fdm_smbh_delay/torch_wave.py",
    "src/fdm_smbh_delay/pyul.py",
}


def _vector(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"wave-only null has an invalid {label}")
    return result


def summarize(control_run: Path, candidate_run: Path) -> dict:
    path = control_run.expanduser().resolve()
    candidate = candidate_run.expanduser().resolve()
    label = path.name
    if label not in _LEVELS or candidate.name != label:
        raise ValueError("wave-only null and coupled levels differ")
    project = Path(__file__).resolve().parents[1]
    control_path = path / "control.json"
    final_path = path / "final_wave.npy"
    metadata_path = candidate / "fdm_adapter_metadata.json"
    coupled_path = candidate.parent / f"momentum_{label}_v1.json"
    record = json.loads(control_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    coupled = json.loads(coupled_path.read_text())
    commit = record.get("source_commit", "")
    job_id = record.get("slurm_job_id", "")
    source_hashes = record.get("source_sha256")
    if (
        record.get("status") != "wave_only_short_prefix_null_not_a_calibration_release"
        or record.get("calibration_eligible") is not False
        or record.get("candidate_run") != str(candidate)
        or record.get("candidate_metadata_sha256") != _sha256(metadata_path)
        or record.get("wave_steps") != _LEVELS[label]
        or coupled.get("run") != str(candidate)
        or coupled.get("last_wave_step") != _LEVELS[label]
        or record.get("initial_wave_sha256") != coupled.get("initial_wave_sha256")
        or record.get("initial_body_sha256") != coupled.get("initial_body_sha256")
        or record.get("final_wave_sha256") != _sha256(final_path)
        or not re.fullmatch(r"[0-9a-f]{40}", commit)
        or not isinstance(job_id, str) or not re.fullmatch(r"[0-9]+", job_id)
        or not isinstance(source_hashes, dict)
        or set(source_hashes) != _SOURCE_PATHS
    ):
        raise ValueError("wave-only null scope, seed or provenance differs")
    guard_path = (candidate.parent.parent / "logs"
                  / f"wave-only-null-guard-{job_id}.json")
    guard = json.loads(guard_path.read_text())
    if guard.get("status") != "solver_exited":
        raise ValueError("wave-only null GPU guard did not exit normally")
    for relative, digest in source_hashes.items():
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=project, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if hashlib.sha256(completed.stdout).hexdigest() != digest:
            raise ValueError("wave-only source differs from committed launch code")
    seed = Path(metadata["reference_initial_state"]).resolve()
    initial_path = seed / "Outputs/3Wfn/P3D_#000.npy"
    if _sha256(initial_path) != record["initial_wave_sha256"]:
        raise ValueError("wave-only initial field differs from coupled seed")
    units = pyul_unit_system(metadata)
    box_code = float(metadata["box_size_pc"]) / units.length_pc
    initial_wave = np.load(initial_path, mmap_mode="r")
    initial = spectral_wave_diagnostics(initial_wave, box_code)
    del initial_wave
    final_wave = np.load(final_path, mmap_mode="r")
    final = spectral_wave_diagnostics(final_wave, box_code)
    del final_wave
    history = np.asarray(record.get("wave_momentum_code"), dtype=float)
    stop = _LEVELS[label]
    if history.shape != (stop + 1, 3) or not np.all(np.isfinite(history)):
        raise ValueError("wave-only momentum history is incomplete")
    gpu_floor = float(record["estimated_endpoint_momentum_rounding_floor_code"])
    cpu_floor = (
        initial["summation_rounding_floor_code"]
        + final["summation_rounding_floor_code"]
    )
    cross_gap = max(
        np.linalg.norm(initial["momentum_code"] - history[0]),
        np.linalg.norm(final["momentum_code"] - history[-1]),
    )
    cross_tolerance = 10.0 * (gpu_floor + cpu_floor)
    cpu_change = final["momentum_code"] - initial["momentum_code"]
    recorded_change = _vector(record.get("final_momentum_change_code"),
                              "recorded final momentum change")
    if (
        not np.isfinite(gpu_floor) or gpu_floor < 0.0
        or not np.isfinite(cpu_floor) or cpu_floor < 0.0
        or cross_gap > cross_tolerance
        or not np.allclose(recorded_change, history[-1] - history[0],
                           rtol=0, atol=cross_tolerance)
        or not np.isclose(initial["mass_code"],
                          record["initial_wave_mass_code"],
                          rtol=1e-11, atol=1e-9)
        or not np.isclose(final["mass_code"],
                          record["final_wave_mass_code"],
                          rtol=1e-11, atol=1e-9)
        or abs(final["mass_code"] / initial["mass_code"] - 1.0) > 1e-10
    ):
        raise ValueError("wave-only CPU endpoint differs from GPU control")
    coupled_total = _vector(coupled.get("total_momentum_change_code"),
                            "coupled total momentum change")
    return {
        "status": "wave_only_null_endpoint_cpu_verified_not_a_coupled_bound",
        "calibration_eligible": False,
        "control_run": str(path),
        "candidate_run": str(candidate),
        "slurm_job_id": job_id,
        "source_commit": commit,
        "control_json_sha256": _sha256(control_path),
        "guard_sha256": _sha256(guard_path),
        "coupled_momentum_audit_sha256": _sha256(coupled_path),
        "analysis_source_sha256": _sha256(Path(__file__).resolve()),
        "initial_wave_sha256": _sha256(initial_path),
        "final_wave_sha256": _sha256(final_path),
        "wave_steps": stop,
        "elapsed_myr": record["elapsed_myr"],
        "cpu_wave_only_momentum_change_code": cpu_change.tolist(),
        "cpu_wave_only_momentum_change_norm_code": float(np.linalg.norm(cpu_change)),
        "gpu_maximum_wave_only_momentum_change_norm_code": float(
            np.max(np.linalg.norm(history - history[0], axis=1))
        ),
        "coupled_total_momentum_change_code": coupled_total.tolist(),
        "coupled_total_momentum_change_norm_code": float(
            np.linalg.norm(coupled_total)
        ),
        "cpu_summation_rounding_floor_code": cpu_floor,
        "gpu_summation_rounding_floor_code": gpu_floor,
        "cpu_gpu_endpoint_disagreement_code": float(cross_gap),
        "cpu_gpu_endpoint_tolerance_code": cross_tolerance,
        "wave_only_change_resolved_above_cpu_floor": bool(
            np.linalg.norm(cpu_change) > cpu_floor
        ),
        "initial_high_frequency_power_fraction": initial[
            "high_frequency_power_fraction"
        ],
        "final_high_frequency_power_fraction": final[
            "high_frequency_power_fraction"
        ],
        "interpretation": (
            "Wave-only evolution omits the compact potential and samples a "
            "different state; its drift is neither an upper nor a lower bound "
            "on the coupled residual"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_run", type=Path)
    parser.add_argument("candidate_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace null audit: {output}")
    payload = summarize(args.control_run, args.candidate_run)
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
