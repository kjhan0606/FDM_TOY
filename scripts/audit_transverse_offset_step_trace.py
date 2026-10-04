#!/usr/bin/env python3
"""Compare x/y/z half-cell offsets on one source-bound n256 prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from fdm_smbh_delay.pyul import pyul_unit_system
from scripts.audit_periodic_offset_step_trace import (
    initial_energy_components, read_run, sha256,
)


SOURCE = "scripts/audit_transverse_offset_step_trace.py"
DEPENDENCY = "scripts/audit_periodic_offset_step_trace.py"
SEED_STATUS = "source_bound_periodic_offset_seed_not_a_calibration_release"


def separation_series(run: Path, *, stop: int, length_pc: float) -> np.ndarray:
    separations = []
    for index in range(stop + 1):
        state = np.load(
            run / f"Outputs/NBody/NTM_#{index:03d}.npy"
        ).reshape(2, 6)
        if not np.all(np.isfinite(state)):
            raise ValueError("nonfinite directional-offset SMBH state")
        separations.append(float(
            np.linalg.norm(state[1, :3] - state[0, :3]) * length_pc
        ))
    return np.asarray(separations)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("x_run", type=Path)
    parser.add_argument("transverse_root", type=Path)
    parser.add_argument("transverse_seed_root", type=Path)
    parser.add_argument("--x-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("source commit must be a full Git revision")
    project = Path(__file__).resolve().parents[1]
    source_hashes = {
        relative: sha256(project / relative)
        for relative in (SOURCE, DEPENDENCY)
    }
    for relative, digest in source_hashes.items():
        committed = subprocess.run(
            ["git", "show", f"{args.source_commit}:{relative}"],
            cwd=project, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if hashlib.sha256(committed.stdout).hexdigest() != digest:
            raise ValueError(f"directional audit source differs: {relative}")
    baseline_run = args.baseline_run.expanduser().resolve()
    x_run = args.x_run.expanduser().resolve()
    transverse_root = args.transverse_root.expanduser().resolve()
    transverse_seed_root = args.transverse_seed_root.expanduser().resolve()
    x_audit_path = args.x_audit.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace directional audit: {output}")
    x_audit = json.loads(x_audit_path.read_text())
    if (
        x_audit.get("status")
        != "n256_short_prefix_offset_diagnostic_not_a_calibration_release"
        or x_audit.get("calibration_eligible") is not False
        or len(x_audit.get("levels", [])) != 2
        or x_audit["levels"][1].get("label") != "f0125"
    ):
        raise ValueError("registered x-offset audit is invalid")
    factor, stop, saves = 0.125, 80, 468000
    runs = {
        "base": baseline_run,
        "x": x_run,
        "y": transverse_root / "half_y_f0125",
        "z": transverse_root / "half_z_f0125",
    }
    records = {
        axis: read_run(path, factor=factor, stop=stop, saves=saves)
        for axis, path in runs.items()
    }
    baseline = records["base"]["metadata"]
    baseline_seed = Path(baseline["reference_initial_state"]).resolve()
    if not np.isclose(
        x_audit["levels"][1]["endpoint_base_separation_pc"],
        separation_series(
            baseline_run, stop=stop,
            length_pc=pyul_unit_system(baseline).length_pc,
        )[-1], rtol=0, atol=1e-14,
    ):
        raise ValueError("x audit baseline endpoint disagrees")
    seed_hashes = {}
    for axis in ("y", "z"):
        seed = transverse_seed_root / f"half_{axis}"
        manifest_path = seed / "offset_seed_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        metadata = records[axis]["metadata"]
        config = records[axis]["config"]
        base_config = records["base"]["config"]
        if (
            manifest.get("status") != SEED_STATUS
            or manifest.get("calibration_eligible") is not False
            or manifest.get("parent_reference") != str(baseline_seed)
            or manifest.get("axis") != axis
            or manifest.get("shift_cells") != 0.5
            or any(
                sha256(baseline_seed / relative) != digest
                for relative, digest in manifest["parent_sha256"].items()
            )
            or any(
                sha256(seed / relative) != digest
                for relative, digest in manifest["derived_sha256"].items()
            )
            or metadata.get("reference_initial_state") != str(seed)
            or metadata.get("reference_initial_wave_sha256")
            != manifest["derived_sha256"]["Outputs/3Wfn/P3D_#000.npy"]
            or metadata.get("reference_initial_particle_sha256")
            != manifest["derived_sha256"]["Outputs/NBody/NTM_#000.npy"]
            or metadata.get("reference_config_sha256")
            != manifest["derived_sha256"]["config.uldm"]
            or metadata.get("reference_metadata_sha256")
            != manifest["derived_sha256"]["fdm_adapter_metadata.json"]
            or metadata.get("solver_source_sha256")
            != baseline.get("solver_source_sha256")
            or any(metadata.get(key) != baseline.get(key) for key in (
                "box_size_pc", "particle_mass_ev", "plummer_radius_pc",
                "pyul_length_unit_m", "pyul_mass_unit_kg", "pyul_time_unit_s",
                "mass_ratio_q", "initial_eccentricity",
                "initial_separation_pc", "semi_major_axis_pc",
                "qe_design_binding",
            ))
            or [row[0] for row in config["Matter Particles"]["Condition"]]
            != [row[0] for row in base_config["Matter Particles"]["Condition"]]
            or [row[2] for row in config["Matter Particles"]["Condition"]]
            != [row[2] for row in base_config["Matter Particles"]["Condition"]]
        ):
            raise ValueError(f"transverse {axis} seed or run differs from baseline")
        seed_hashes[axis] = sha256(manifest_path)
    if records["x"]["hashes"] != x_audit["levels"][1][
        "run_input_sha256"
    ]["half"]:
        raise ValueError("x-offset run changed since registered audit")
    length_pc = pyul_unit_system(baseline).length_pc
    separations = {
        axis: separation_series(path, stop=stop, length_pc=length_pc)
        for axis, path in runs.items()
    }
    if not np.isclose(
        separations["x"][-1] - separations["base"][-1],
        x_audit["levels"][1]["endpoint_half_minus_base_separation_pc"],
        rtol=0, atol=1e-14,
    ):
        raise ValueError("x-offset trajectory changed since registered audit")
    directions = {}
    for axis in ("x", "y", "z"):
        conservation = records[axis]["conservation"]
        energy = float(conservation[
            "max_total_energy_drift_over_energy_transfer"
        ])
        mass = float(conservation["max_wave_mass_relative_error"])
        if (
            not np.isfinite(energy) or energy > 0.01
            or not np.isfinite(mass) or mass > 1e-10
        ):
            raise ValueError(f"directional {axis} short-prefix conservation fails")
        differences = separations[axis] - separations["base"]
        directions[axis] = {
            "endpoint_separation_difference_pc": float(differences[-1]),
            "maximum_absolute_prefix_separation_difference_pc": float(
                np.max(np.abs(differences))
            ),
            "initial_energy_components_code": initial_energy_components(
                runs[axis]
            ),
            "max_energy_error_over_transfer": energy,
            "max_wave_mass_relative_error": mass,
            "run_input_sha256": records[axis]["hashes"],
        }
    x_absolute = abs(directions["x"]["endpoint_separation_difference_pc"])
    transverse_maximum = max(abs(directions[axis][
        "endpoint_separation_difference_pc"]) for axis in ("y", "z"))
    payload = {
        "status": "n256_three_axis_half_cell_diagnostic_not_a_calibration_release",
        "calibration_eligible": False,
        "source_commit": args.source_commit,
        "source_sha256": source_hashes,
        "x_audit_sha256": sha256(x_audit_path),
        "transverse_seed_manifest_sha256": seed_hashes,
        "baseline_run_input_sha256": records["base"]["hashes"],
        "directions": directions,
        "x_to_largest_transverse_endpoint_offset_ratio": (
            x_absolute / max(transverse_maximum, np.finfo(float).tiny)
        ),
        "interpretation": (
            "Three coordinate-axis half-cell shifts on one n256 short "
            "prefix expose anisotropic grid anchoring. They are not a "
            "spatial-resolution limit or coalescence-time uncertainty."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(temporary, output)
    temporary.unlink()
    print(json.dumps({"status": payload["status"],
                      "x_over_transverse": payload[
                          "x_to_largest_transverse_endpoint_offset_ratio"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
