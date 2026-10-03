#!/usr/bin/env python3
"""Read one registered q/e pilot against its preregistered fixed bin."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from fdm_smbh_delay.convergence import (
    _initial_resolved_orbit_indices,
    load_convergence_run,
)
from fdm_smbh_delay.qe_followup_design import (
    read_verified_qe_followup_design,
    verify_qe_design_run_request,
)
from fdm_smbh_delay.run_metadata import validate_torch_calibration_completion


PILOT_INPUTS = (
    "fdm_adapter_metadata.json",
    "config.uldm",
    "torch_run_summary.json",
    "conservation_summary.json",
    "conservation_timeseries.csv",
    "orbit_averaged_exchange_summary.json",
    "orbit_averaged_exchange.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_pilot_complete(
    run: Path, *, case_id: str, resolution: int, duration_myr: float
) -> dict[str, str]:
    """Validate the completed Torch, conservation, and orbit pilot stages."""

    missing = [name for name in PILOT_INPUTS if not (run / name).is_file()]
    if missing:
        raise ValueError(f"registered q/e pilot lacks required inputs {missing}: {run}")
    metadata = json.loads((run / "fdm_adapter_metadata.json").read_text(encoding="utf-8"))
    validate_torch_calibration_completion(
        run,
        expected_case_id=case_id,
        expected_resolution=resolution,
        expected_duration_myr=duration_myr,
        expected_saved_intervals=int(metadata.get("save_number", 0)),
        expected_saved_3d_states=int(metadata.get("saved_3d_states", 0)),
        expected_rk4_substeps=int(metadata.get("nbody_rk4_substeps_per_wave_step", 0)),
        expected_checkpoint_interval=int(
            metadata.get("checkpoint_every_saved_intervals", 0)
        ),
        expected_time_step_factor=float(metadata.get("time_step_factor", math.nan)),
        expected_run_id=metadata.get("run_id"),
    )
    conservation = json.loads(
        (run / "conservation_summary.json").read_text(encoding="utf-8")
    )
    if (
        conservation.get("status") != "diagnosed"
        or conservation.get("case_id") != case_id
        or conservation.get("resolution") != resolution
        or not math.isclose(
            float(conservation.get("duration_myr", math.nan)),
            duration_myr,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("registered q/e pilot conservation identity is invalid")
    orbit = json.loads(
        (run / "orbit_averaged_exchange_summary.json").read_text(encoding="utf-8")
    )
    if orbit.get("status") != "orbit_averaged" or int(orbit.get("complete_orbits", 0)) < 1:
        raise ValueError("registered q/e pilot orbit average is incomplete")
    return {name: _sha256(run / name) for name in PILOT_INPUTS}


def _manifest_run_id(
    run_manifest: Path, *, case_id: str, resolution: int
) -> str:
    with run_manifest.open(newline="", encoding="utf-8") as stream:
        matches = [
            row.get("run_id", "")
            for row in csv.DictReader(stream)
            if row.get("case_id") == case_id
            and row.get("effective_grid_cells") == str(resolution)
        ]
    if len(matches) != 1 or not matches[0]:
        raise ValueError("registered q/e pilot lacks a unique bound manifest run ID")
    return matches[0]


def audit_registered_single_pilot(
    design_path: Path,
    physical_cases: Path,
    run_manifest: Path,
    run: Path,
) -> dict:
    """Measure fixed-bin occupancy for one complete, design-bound run."""

    design_path = design_path.expanduser().resolve()
    physical_cases = physical_cases.expanduser().resolve()
    run_manifest = run_manifest.expanduser().resolve()
    run = run.expanduser().resolve()
    design, design_file_sha256 = read_verified_qe_followup_design(
        design_path, physical_cases=physical_cases, run_manifest=run_manifest
    )
    metadata_path = run / "fdm_adapter_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"registered q/e pilot lacks adapter metadata: {run}")
    metadata_bytes = metadata_path.read_bytes()
    metadata_sha256 = _sha256_bytes(metadata_bytes)
    metadata = json.loads(metadata_bytes)
    binding = metadata.get("qe_design_binding")
    if (
        not isinstance(binding, dict)
        or binding.get("status")
        != "qe_prospective_design_bound_not_a_calibration_release"
        or binding.get("design_sha256") != design["design_sha256"]
        or binding.get("file_sha256") != design_file_sha256
    ):
        raise ValueError("q/e pilot lacks the registered design identity")
    role = binding.get("role")
    roles = {
        "coarse": design["resolution_pair"]["coarse"],
        "fine": design["resolution_pair"]["fine"],
        "doubled_box_control": design["doubled_box_control"],
    }
    if role not in roles:
        raise ValueError("q/e pilot registered design role is invalid")
    expected = roles[role]
    manifest_sha256_before_lookup = _sha256(run_manifest)
    if manifest_sha256_before_lookup != design["run_manifest_sha256"]:
        raise ValueError("registered q/e pilot manifest differs from the bound design")
    manifest_run_id = _manifest_run_id(
        run_manifest,
        case_id=design["case_id"],
        resolution=expected["resolution"],
    )
    if _sha256(run_manifest) != manifest_sha256_before_lookup:
        raise ValueError("registered q/e pilot manifest changed during identity validation")
    if (
        metadata.get("run_id") != manifest_run_id
        or run.name != metadata.get("run_id")
        or (
            expected.get("run_id") is not None
            and manifest_run_id != expected["run_id"]
        )
    ):
        raise ValueError("q/e pilot run ID disagrees with the registered design")
    verify_qe_design_run_request(
        design,
        case_id=metadata.get("case_id"),
        role=role,
        resolution=metadata.get("resolution"),
        box_size_pc=metadata.get("box_size_pc"),
        duration_myr=metadata.get("duration_myr"),
    )
    input_sha256 = _require_pilot_complete(
        run, case_id=design["case_id"], resolution=expected["resolution"],
        duration_myr=float(metadata["duration_myr"]),
    )
    if input_sha256["fdm_adapter_metadata.json"] != metadata_sha256:
        raise ValueError("registered q/e pilot metadata changed during identity validation")
    loaded = load_convergence_run(metadata.get("run_id", run.name), run)
    if _require_pilot_complete(
        run, case_id=design["case_id"], resolution=expected["resolution"],
        duration_myr=float(metadata["duration_myr"]),
    ) != input_sha256:
        raise ValueError("registered q/e pilot inputs changed during the audit")
    orbit = loaded["orbit_series"]
    if orbit is None:
        raise ValueError("registered q/e pilot lacks orbit-averaged output")
    orbit_summary = loaded["orbit"]
    if int(orbit_summary["complete_orbits"]) != int(orbit.size):
        raise ValueError("registered q/e pilot orbit summary disagrees with its table")
    valid = _initial_resolved_orbit_indices(
        orbit,
        float(loaded["conservation"]["initial_spatially_resolved_duration_myr"]),
    )
    edges = design["separation_bin_edges_pc"]
    if len(edges) != 2:
        raise ValueError("single-run pilot diagnostic requires one registered fixed bin")
    lo, hi = map(float, edges)
    separation = np.asarray(orbit["mean_separation_pc"][valid], dtype=float)
    selection = valid[(separation >= lo) & (separation <= hi)]
    eccentricity = np.asarray(
        orbit["mean_eccentricity_osculating"][selection], dtype=float
    )
    durations = np.asarray(orbit["orbital_period_myr"][selection], dtype=float)
    count = int(selection.size)
    threshold = int(design["minimum_orbits_per_bin"])
    passed = count >= threshold
    if (
        _sha256(design_path) != design_file_sha256
        or _sha256(run_manifest) != design["run_manifest_sha256"]
        or _sha256(physical_cases) != design["physical_cases_sha256"]
    ):
        raise ValueError("registered q/e bound design inputs changed during the audit")
    return {
        "status": "qe_registered_single_run_fixed_bin_design_diagnostic_only",
        "release_status": "no_calibration_release",
        "production_calibration_row_admitted": False,
        "design": str(design_path),
        "design_sha256": design["design_sha256"],
        "design_file_sha256": design_file_sha256,
        "run": str(run),
        "run_id": metadata.get("run_id"),
        "case_id": design["case_id"],
        "role": role,
        "resolution": expected["resolution"],
        "run_input_sha256": input_sha256,
        "fixed_bin_lower_separation_pc": lo,
        "fixed_bin_upper_separation_pc": hi,
        "initially_resolved_complete_orbits_in_fixed_bin": count,
        "minimum_required_complete_orbits": threshold,
        "orbit_count_threshold_passed": passed,
        "measured_eccentricity_minimum": (
            None if count == 0 else float(np.min(eccentricity))
        ),
        "measured_eccentricity_maximum": (
            None if count == 0 else float(np.max(eccentricity))
        ),
        "measured_eccentricity_duration_weighted_mean": (
            None if count == 0 else float(np.average(eccentricity, weights=durations))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--physical-cases", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_registered_single_pilot(
        args.design, args.physical_cases, args.run_manifest, args.run
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
