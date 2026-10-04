#!/usr/bin/env python3
"""Bind and compare same-grid 17-year periodic-TSC prefixes at dt and dt/2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from fdm_smbh_delay.pyul import pyul_unit_system
from scripts.analyze_direct_softening_coupled_excess import isolated_plummer_kdk
from scripts.audit_direct_softening_step_trace import separation_series
from scripts.audit_periodic_offset_step_trace import sha256
from scripts.audit_spatial_resolution_step_trace import (
    validate_fixed_physics, verified_run,
)


SOURCE = "scripts/audit_spatial_temporal_refinement.py"
DEPENDENCIES = (
    "scripts/audit_spatial_resolution_step_trace.py",
    "scripts/analyze_direct_softening_coupled_excess.py",
    "scripts/audit_direct_softening_step_trace.py",
    "scripts/audit_periodic_offset_step_trace.py",
)
LEVELS = (("n192", 192, 0.0625), ("n256", 256, 0.0625), ("n384", 384, 0.15))
STEPS = 160
SAVES = 936000


def checked_source(project: Path, commit: str) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source commit must be a full Git revision")
    result = {}
    for relative in (SOURCE, *DEPENDENCIES):
        digest = sha256(project / relative)
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=project,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if hashlib.sha256(committed.stdout).hexdigest() != digest:
            raise ValueError(f"temporal audit source differs: {relative}")
        result[relative] = digest
    return result


def read_bound_coarse(row: dict, *, steps: int = 80) -> tuple[Path, dict, dict, np.ndarray]:
    run = Path(row["run_directory"]).resolve()
    hashes = row["run_input_sha256"]
    if any(sha256(run / relative) != digest for relative, digest in hashes.items()):
        raise ValueError(f"prior coarse run inputs changed: {run}")
    if (
        set(row["body_state_sha256"])
        != {str(index) for index in range(steps + 1)}
        or any(
            sha256(run / f"Outputs/NBody/NTM_#{index:03d}.npy")
            != row["body_state_sha256"].get(str(index))
            for index in range(steps + 1)
        )
    ):
        raise ValueError(f"prior coarse SMBH states changed: {run}")
    metadata = json.loads((run / "fdm_adapter_metadata.json").read_text())
    config = json.loads((run / "config.uldm").read_text())
    units = pyul_unit_system(metadata)
    series = separation_series(run, stop=steps, length_pc=units.length_pc)
    if not np.isclose(
        series[-1], row["endpoint_coupled_separation_pc"],
        rtol=0, atol=1e-14,
    ):
        raise ValueError("prior coarse separation differs from audited endpoint")
    return run, metadata, config, series


def compare_matched_series(old: np.ndarray, half: np.ndarray) -> dict[str, float]:
    coarse = np.asarray(old, dtype=np.float64)
    refined = np.asarray(half, dtype=np.float64)
    if (
        coarse.shape != (81,) or refined.shape != (161,)
        or not np.all(np.isfinite(coarse))
        or not np.all(np.isfinite(refined))
        or not np.isclose(refined[0], coarse[0], rtol=0, atol=1e-14)
    ):
        raise ValueError("temporal series do not share finite initial/common saves")
    difference = refined[::2] - coarse
    return {
        "endpoint_pc": float(difference[-1]),
        "maximum_absolute_prefix_pc": float(np.max(np.abs(difference))),
    }


def config_matches_except_save_count(old: dict, half: dict) -> bool:
    comparison = json.loads(json.dumps(half))
    try:
        old_count = old["Save Options"]["Number"]
        half_count = comparison["Save Options"]["Number"]
        if (
            type(old_count) is not int or old_count <= 0
            or type(half_count) is not int or half_count != 2 * old_count
        ):
            return False
        comparison["Save Options"]["Number"] = old_count
    except (KeyError, TypeError):
        return False
    return comparison == old


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coarse_audit", type=Path)
    parser.add_argument("half_run_root", type=Path)
    parser.add_argument("parent_seed", type=Path)
    parser.add_argument("derived_seed_root", type=Path)
    parser.add_argument("--coarse-audit-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source_hashes = checked_source(project, args.source_commit)
    old_path = args.coarse_audit.expanduser().resolve()
    if sha256(old_path) != args.coarse_audit_sha256:
        raise ValueError("coarse spatial audit digest changed")
    old = json.loads(old_path.read_text())
    if (
        old.get("status")
        != "n192_n256_n384_fixed_physics_tsc_prefix_diagnostic_not_calibration"
        or old.get("calibration_eligible") is not False
        or [row["label"] for row in old.get("levels", [])]
        != [label for label, _, _ in LEVELS]
    ):
        raise ValueError("coarse audit is not the registered three-grid diagnostic")
    half_root = args.half_run_root.expanduser().resolve()
    parent_seed = args.parent_seed.expanduser().resolve()
    seed_root = args.derived_seed_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace spatial temporal audit: {output}")
    rows = []
    new_metadata_by_label = {}
    new_configs_by_label = {}
    for (label, resolution, factor), previous in zip(
        LEVELS, old["levels"], strict=True,
    ):
        old_run, old_metadata, old_config, old_sep = read_bound_coarse(previous)
        seed = parent_seed if label == "n256" else seed_root / label
        new_run = half_root / f"{label}_dt2"
        new_metadata, new_config, conservation = verified_run(
            new_run, seed, resolution=resolution, factor=factor,
            steps=STEPS, saves=SAVES,
        )
        physical_keys = (
            "box_size_pc", "cell_size_pc", "plummer_radius_pc",
            "particle_mass_ev", "pyul_length_unit_m", "pyul_mass_unit_kg",
            "pyul_time_unit_s", "mass_ratio_q", "initial_eccentricity",
            "initial_separation_pc", "semi_major_axis_pc",
        )
        if (
            not config_matches_except_save_count(old_config, new_config)
            or any(new_metadata.get(key) != old_metadata.get(key)
                   for key in physical_keys)
            or old_metadata.get("solver_source_sha256")
            != new_metadata.get("solver_source_sha256")
            or old_metadata.get("reference_initial_wave_sha256")
            != new_metadata.get("reference_initial_wave_sha256")
            or old_metadata.get("reference_initial_particle_sha256")
            != new_metadata.get("reference_initial_particle_sha256")
            or old_metadata.get("reference_config_sha256")
            != new_metadata.get("reference_config_sha256")
            or old_metadata.get("reference_metadata_sha256")
            != new_metadata.get("reference_metadata_sha256")
            or not np.isclose(
                old_metadata.get("wave_time_step_code", np.nan),
                2.0 * new_metadata.get("wave_time_step_code", np.nan),
                rtol=0, atol=1e-25,
            )
        ):
            raise ValueError(f"same-grid temporal controls differ beyond time step: {label}")
        new_metadata_by_label[label] = new_metadata
        new_configs_by_label[label] = new_config
        units = pyul_unit_system(new_metadata)
        new_sep = separation_series(new_run, stop=STEPS, length_pc=units.length_pc)
        initial = np.load(new_run / "Outputs/NBody/NTM_#000.npy").reshape(2, 6)
        if sha256(new_run / "Outputs/NBody/NTM_#000.npy") != previous[
            "body_state_sha256"
        ]["0"]:
            raise ValueError(f"same-grid initial SMBH state changed: {label}")
        masses = np.asarray([
            row[0] for row in new_config["Matter Particles"]["Condition"]
        ], dtype=float) / units.mass_msun
        isolated = isolated_plummer_kdk(
            initial, masses,
            new_config["Matter Particles"]["Plummer Radius"] / units.length_pc,
            new_metadata["wave_time_step_code"], STEPS,
        )
        isolated_sep = np.linalg.norm(
            isolated[:, 1, :3] - isolated[:, 0, :3], axis=1,
        ) * units.length_pc
        temporal = compare_matched_series(old_sep, new_sep)
        energy = float(conservation["max_total_energy_drift_over_energy_transfer"])
        mass_error = float(conservation["max_wave_mass_relative_error"])
        if (
            not np.isfinite(energy) or energy < 0.0
            or not np.isfinite(mass_error) or mass_error > 1e-10
        ):
            raise ValueError(f"half-step conservation summary invalid: {label}")
        rows.append({
            "label": label,
            "resolution": resolution,
            "old_run_directory": str(old_run),
            "half_run_directory": str(new_run),
            "old_time_step_code": old_metadata["wave_time_step_code"],
            "half_time_step_code": new_metadata["wave_time_step_code"],
            "old_endpoint_coupled_separation_pc": float(old_sep[-1]),
            "half_endpoint_coupled_separation_pc": float(new_sep[-1]),
            "half_minus_old_endpoint_separation_pc": temporal["endpoint_pc"],
            "maximum_absolute_matched_prefix_difference_pc": temporal[
                "maximum_absolute_prefix_pc"
            ],
            "half_endpoint_coupled_minus_isolated_pc": float(
                new_sep[-1] - isolated_sep[-1]
            ),
            "old_max_energy_error_over_transfer": previous[
                "conservation"
            ]["max_energy_error_over_transfer"],
            "half_max_energy_error_over_transfer": energy,
            "half_short_prefix_energy_limit_passed": energy <= 0.01,
            "half_max_wave_mass_relative_error": mass_error,
            "half_body_state_sha256": {
                str(index): sha256(
                    new_run / f"Outputs/NBody/NTM_#{index:03d}.npy"
                ) for index in range(STEPS + 1)
            },
            "half_run_input_sha256": {
                relative: sha256(new_run / relative)
                for relative in (
                    "fdm_adapter_metadata.json", "config.uldm",
                    "torch_run_summary.json", "conservation_summary.json",
                    "torch_solver_provenance/manifest.json",
                )
            },
        })
    base_new = new_metadata_by_label["n256"]
    for label, resolution, _ in LEVELS:
        validate_fixed_physics(
            base_new, new_configs_by_label["n256"],
            new_metadata_by_label[label], new_configs_by_label[label],
            resolution=resolution,
        )
    old_values = {row["label"]: row["endpoint_coupled_separation_pc"]
                  for row in old["levels"]}
    half_values = {row["label"]: row["half_endpoint_coupled_separation_pc"]
                   for row in rows}
    old_mesh = {
        label: float(old_values[label] - old_values["n256"])
        for label in ("n192", "n384")
    }
    half_mesh = {
        label: float(half_values[label] - half_values["n256"])
        for label in ("n192", "n384")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "n192_n256_n384_temporal_refinement_prefix_diagnostic_not_calibration",
        "calibration_eligible": False,
        "coarse_audit": str(old_path),
        "coarse_audit_sha256": args.coarse_audit_sha256,
        "source_commit": args.source_commit,
        "source_sha256": source_hashes,
        "levels": rows,
        "old_mesh_differences_pc": old_mesh,
        "half_step_mesh_differences_pc": half_mesh,
        "mesh_difference_change_pc": {
            label: half_mesh[label] - old_mesh[label]
            for label in old_mesh
        },
        "interpretation": (
            "Each grid uses its original wave and SMBH seed for dt/2 on the "
            "same 17-year interval. This quantifies temporal contamination "
            "of the three-grid TSC comparison; it does not isolate physical "
            "wake drag or establish a full-orbit spatial limit."
        ),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "mesh_difference_change_pc": payload["mesh_difference_change_pc"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
