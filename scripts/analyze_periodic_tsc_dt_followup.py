#!/usr/bin/env python3
"""Compare source-bound reciprocal TSC runs at common physical save times."""

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
    from .analyze_qe_startup_component_dt import _read_run, _sha256
else:
    from analyze_qe_startup_component_dt import _read_run, _sha256


_TSC_SOURCE = "src/fdm_smbh_delay/periodic_mesh_coupling.py"
_REQUIRED_SOURCES = {
    "scripts/run_torch_wave_case.py",
    "src/fdm_smbh_delay/torch_wave.py",
    "src/fdm_smbh_delay/pyul.py",
    _TSC_SOURCE,
}


def _launch_owner(run: Path, project: Path, sources: tuple) -> dict:
    owner_path = run.parent / "OWNER.txt"
    pairs = [line.split("=", 1) for line in owner_path.read_text().splitlines()]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"{run}: launch ownership record is malformed")
    owner = dict(pairs)
    commit = owner.get("source_commit", "")
    if (
        owner.get("project") != "FDM_TOY"
        or not re.fullmatch(r"[0-9a-f]{40}", commit)
        or not re.fullmatch(r"[0-9]+", owner.get("job_id", ""))
        or not _REQUIRED_SOURCES.issubset({name for name, _ in sources})
    ):
        raise ValueError(f"{run}: launch owner or numerical source is unverified")
    for relative, frozen_hash in sources:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=project,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if hashlib.sha256(completed.stdout).hexdigest() != frozen_hash:
            raise ValueError(f"{run}: source snapshot differs from launch commit")
    return {
        "slurm_job_id": owner["job_id"],
        "source_commit": commit,
        "launch_owner_sha256": _sha256(owner_path),
    }


def summarize(reference_run: Path, followup_root: Path) -> dict:
    project = Path(__file__).resolve().parents[1]
    analysis_paths = {
        "scripts/analyze_periodic_tsc_dt_followup.py": Path(__file__).resolve(),
        "scripts/analyze_qe_startup_component_dt.py": (
            project / "scripts/analyze_qe_startup_component_dt.py"
        ),
        "scripts/analyze_pyul_wave_run.py": (
            project / "scripts/analyze_pyul_wave_run.py"
        ),
    }
    analysis_hashes = {name: _sha256(path) for name, path in analysis_paths.items()}
    locations = (
        (reference_run, 1.0, 1),
        (followup_root / "f050", 0.5, 2),
        (followup_root / "f025", 0.25, 4),
    )
    runs = []
    for path, factor, wave_steps in locations:
        item = _read_run(path.parent, path.name, factor, wave_steps)
        metadata = json.loads((path / "fdm_adapter_metadata.json").read_text())
        if (
            metadata.get("wave_smbh_coupling") != "periodic_tsc_reciprocal"
            or metadata.get("experimental_coupling_not_a_calibration_release")
            is not True
            or _TSC_SOURCE not in {name for name, _ in item["source_fingerprints"]}
        ):
            raise ValueError(f"{path}: reciprocal TSC source or scope is unverified")
        if (
            analysis_paths["scripts/analyze_pyul_wave_run.py"].stat().st_mtime_ns
            > (path / "conservation_summary.json").stat().st_mtime_ns
        ):
            raise ValueError(f"{path}: conservation analyser changed after output")
        item["run"] = str(path)
        item.update(_launch_owner(path, project, item["source_fingerprints"]))
        runs.append(item)
    if (
        len({item["launch_hashes"] for item in runs}) != 1
        or len({item["source_fingerprints"] for item in runs}) != 1
        or any(
            item["physical_metadata"] != runs[0]["physical_metadata"]
            or item["matter_particles"] != runs[0]["matter_particles"]
            for item in runs[1:]
        )
    ):
        raise ValueError("reciprocal TSC runs lack common source and physical inputs")
    if any(
        not np.isclose(
            item["wave_time_step_code"],
            runs[0]["wave_time_step_code"] * item["time_step_factor"],
            rtol=1e-12, atol=0.0,
        )
        or not np.allclose(
            item["time_myr"], runs[0]["time_myr"], rtol=1e-12, atol=0.0
        )
        for item in runs[1:]
    ):
        raise ValueError("reciprocal TSC runs lack matched physical save times")
    for item in runs:
        item["first_save_hamiltonian_change_msun_pc2_myr2"] = item[
            "change_from_initial"
        ]["combined_energy"][1]
        item["final_save_hamiltonian_change_msun_pc2_myr2"] = item[
            "change_from_initial"
        ]["combined_energy"][-1]
        item["calibration_eligible"] = False
    if analysis_hashes != {
        name: _sha256(path) for name, path in analysis_paths.items()
    }:
        raise ValueError("time-step analysis source changed during comparison")
    return {
        "status": "periodic_tsc_temporal_diagnostic_not_a_calibration_release",
        "reference_run": str(reference_run),
        "followup_root": str(followup_root),
        "reference_conservation_summary_sha256": _sha256(
            reference_run / "conservation_summary.json"
        ),
        "registered_tolerance_unchanged": 0.01,
        "analysis_source_sha256": analysis_hashes,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_run", type=Path)
    parser.add_argument("followup_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = args.reference_run.expanduser().resolve()
    followup = args.followup_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace TSC time-step analysis: {output}")
    payload = summarize(reference, followup)
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
