#!/usr/bin/env python3
"""Audit matched direct-SMBH Plummer controls on short coupled prefixes."""

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


SOURCE = "scripts/audit_direct_softening_step_trace.py"
DEPENDENCY = "scripts/audit_periodic_offset_step_trace.py"
LEVELS = (("f100", 1.0, 10, 58500), ("f0125", 0.125, 80, 468000))
VARIANTS = (("quarter", 0.25), ("full", 1.0))
SEED_STATUS = "source_bound_direct_binary_softening_seed_not_calibration"


def matching_except_direct_plummer(candidate: dict, baseline: dict) -> bool:
    restored = json.loads(json.dumps(candidate))
    restored["Matter Particles"]["Plummer Radius"] = baseline[
        "Matter Particles"
    ]["Plummer Radius"]
    return restored == baseline


def separation_series(run: Path, *, stop: int, length_pc: float) -> np.ndarray:
    values = []
    for index in range(stop + 1):
        state = np.load(
            run / f"Outputs/NBody/NTM_#{index:03d}.npy"
        ).reshape(2, 6)
        if not np.all(np.isfinite(state)):
            raise ValueError("direct-softening state is nonfinite")
        values.append(float(np.linalg.norm(
            state[1, :3] - state[0, :3]
        ) * length_pc))
    return np.asarray(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("control_root", type=Path)
    parser.add_argument("seed_root", type=Path)
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
            raise ValueError(f"direct-softening audit source differs: {relative}")
    baseline_root = args.baseline_root.expanduser().resolve()
    control_root = args.control_root.expanduser().resolve()
    seed_root = args.seed_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace softening audit: {output}")
    seed_records = {}
    for label, fraction in VARIANTS:
        seed = seed_root / label
        manifest_path = seed / "direct_softening_seed_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") != SEED_STATUS
            or manifest.get("calibration_eligible") is not False
            or manifest.get("radius_over_cell") != fraction
            or manifest.get("changed_operator")
            != "direct_smbh_smbh_plummer_only"
            or manifest.get("wave_compact_assignment")
            != "periodic_tsc_unchanged"
            or any(
                sha256(seed / relative) != digest
                for relative, digest in manifest["derived_sha256"].items()
            )
        ):
            raise ValueError(f"direct-softening seed differs: {label}")
        seed_records[label] = {
            "path": str(seed),
            "manifest_sha256": sha256(manifest_path),
            "manifest": manifest,
        }
    results = []
    common_source = None
    for level, factor, stop, saves in LEVELS:
        paths = {
            "baseline": baseline_root / level,
            **{
                label: control_root / f"{label}_{level}"
                for label, _fraction in VARIANTS
            },
        }
        records = {
            label: read_run(path, factor=factor, stop=stop, saves=saves)
            for label, path in paths.items()
        }
        baseline = records["baseline"]["metadata"]
        baseline_config = records["baseline"]["config"]
        baseline_seed = Path(baseline["reference_initial_state"]).resolve()
        source = baseline["solver_source_sha256"]
        if common_source is None:
            common_source = source
        if source != common_source:
            raise ValueError("softening levels use different numerical sources")
        for label, fraction in VARIANTS:
            seed = seed_records[label]
            manifest = seed["manifest"]
            metadata = records[label]["metadata"]
            config = records[label]["config"]
            binding = metadata.get("direct_softening_binding", {})
            physical_keys = (
                "box_size_pc", "particle_mass_ev", "pyul_length_unit_m",
                "pyul_mass_unit_kg", "pyul_time_unit_s", "mass_ratio_q",
                "initial_eccentricity", "initial_separation_pc",
                "semi_major_axis_pc", "qe_design_binding",
            )
            if (
                manifest.get("parent_reference") != str(baseline_seed)
                or any(
                    sha256(baseline_seed / relative) != digest
                    for relative, digest in manifest["parent_sha256"].items()
                )
                or metadata.get("reference_initial_state") != seed["path"]
                or metadata.get("solver_source_sha256") != source
                or metadata.get("reference_initial_wave_sha256")
                != manifest["derived_sha256"]["Outputs/3Wfn/P3D_#000.npy"]
                or metadata.get("reference_initial_particle_sha256")
                != manifest["derived_sha256"]["Outputs/NBody/NTM_#000.npy"]
                or metadata.get("reference_config_sha256")
                != manifest["derived_sha256"]["config.uldm"]
                or metadata.get("reference_metadata_sha256")
                != manifest["derived_sha256"]["fdm_adapter_metadata.json"]
                or metadata.get("reference_initial_wave_sha256")
                != baseline.get("reference_initial_wave_sha256")
                or metadata.get("reference_initial_particle_sha256")
                != baseline.get("reference_initial_particle_sha256")
                or any(metadata.get(key) != baseline.get(key)
                       for key in physical_keys)
                or not matching_except_direct_plummer(config, baseline_config)
                or metadata.get("plummer_radius_pc")
                != manifest.get("radius_pc")
                or binding.get("radius_over_cell") != fraction
                or binding.get("changed_operator")
                != "direct_smbh_smbh_plummer_only"
                or metadata.get("compact_potential_layout")
                != baseline.get("compact_potential_layout")
            ):
                raise ValueError(f"only direct Plummer may differ: {label}")
        length_pc = pyul_unit_system(baseline).length_pc
        separations = {
            label: separation_series(path, stop=stop, length_pc=length_pc)
            for label, path in paths.items()
        }
        baseline_initial = separations["baseline"][0]
        if any(
            not np.isclose(series[0], baseline_initial, rtol=0, atol=1e-14)
            for series in separations.values()
        ):
            raise ValueError("softening variants start at different separations")
        energy = {}
        mass = {}
        initial_energy = {}
        variants = {}
        for label, path in paths.items():
            conservation = records[label]["conservation"]
            energy[label] = float(conservation[
                "max_total_energy_drift_over_energy_transfer"
            ])
            mass[label] = float(conservation["max_wave_mass_relative_error"])
            if (
                not np.isfinite(energy[label]) or energy[label] < 0.0
                or not np.isfinite(mass[label]) or mass[label] > 1e-10
            ):
                raise ValueError(f"invalid softening conservation: {label}")
            initial_energy[label] = initial_energy_components(path)
            difference = separations[label] - separations["baseline"]
            variants[label] = {
                "endpoint_separation_pc": float(separations[label][-1]),
                "endpoint_minus_baseline_separation_pc": float(difference[-1]),
                "maximum_absolute_prefix_separation_difference_pc": float(
                    np.max(np.abs(difference))
                ),
                "max_energy_error_over_transfer": energy[label],
                "short_prefix_energy_limit_passed": energy[label] <= 0.01,
                "max_wave_mass_relative_error": mass[label],
                "initial_energy_components_code": initial_energy[label],
                "run_input_sha256": records[label]["hashes"],
            }
        results.append({
            "label": level, "time_step_factor": factor,
            "wave_steps": stop,
            "baseline_initial_separation_pc": baseline_initial,
            "baseline_prefix_separation_change_pc": float(
                separations["baseline"][-1] - baseline_initial
            ),
            "variants": variants,
        })
    for label, _fraction in VARIANTS:
        temporal_difference = abs(
            results[0]["variants"][label]["endpoint_minus_baseline_separation_pc"]
            - results[1]["variants"][label]["endpoint_minus_baseline_separation_pc"]
        )
        seed_records[label]["offset_temporal_difference_pc"] = temporal_difference
    payload = {
        "status": "n256_direct_binary_softening_prefix_diagnostic_not_calibration",
        "calibration_eligible": False,
        "changed_operator": "direct_smbh_smbh_plummer_only",
        "wave_compact_assignment": "periodic_tsc_unchanged",
        "source_commit": args.source_commit,
        "source_sha256": source_hashes,
        "seed_records": seed_records,
        "levels": results,
        "interpretation": (
            "The direct SMBH-SMBH Plummer length changes the short-prefix "
            "trajectory; the wave-compact TSC assignment and initial fields "
            "are fixed. This is not a joint mesh-softening or full-orbit "
            "convergence test and cannot release a calibration row."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(temporary, output)
    temporary.unlink()
    print(json.dumps({"status": payload["status"],
                      "quarter_temporal_difference_pc": seed_records[
                          "quarter"]["offset_temporal_difference_pc"],
                      "full_temporal_difference_pc": seed_records[
                          "full"]["offset_temporal_difference_pc"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
