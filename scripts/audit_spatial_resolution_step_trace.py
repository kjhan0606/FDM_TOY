#!/usr/bin/env python3
"""Audit a fixed-input, variable-TSC-kernel n192/n256/n384 prefix ladder."""

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
from scripts.audit_periodic_offset_step_trace import initial_energy_components, sha256


SOURCE = "scripts/audit_spatial_resolution_step_trace.py"
DEPENDENCIES = (
    "scripts/derive_spatial_resolution_seed.py",
    "scripts/analyze_direct_softening_coupled_excess.py",
    "scripts/audit_direct_softening_step_trace.py",
    "scripts/audit_periodic_offset_step_trace.py",
)
GRID = (("n192", 192, 0.125), ("n256", 256, 0.125), ("n384", 384, 0.30))
STEPS = 80
SAVES = 468000


def verified_source(project: Path, commit: str) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source commit must be a full Git revision")
    hashes = {}
    for relative in (SOURCE, *DEPENDENCIES):
        digest = sha256(project / relative)
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=project,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if hashlib.sha256(committed.stdout).hexdigest() != digest:
            raise ValueError(f"spatial audit source differs from commit: {relative}")
        hashes[relative] = digest
    return hashes


def verified_run(
    run: Path, seed: Path, *, resolution: int, factor: float,
    steps: int = STEPS, saves: int = SAVES,
) -> tuple[dict, dict, dict]:
    metadata_path = run / "fdm_adapter_metadata.json"
    config_path = run / "config.uldm"
    provenance_path = run / "torch_solver_provenance/manifest.json"
    metadata = json.loads(metadata_path.read_text())
    config = json.loads(config_path.read_text())
    summary = json.loads((run / "torch_run_summary.json").read_text())
    conservation = json.loads((run / "conservation_summary.json").read_text())
    provenance = json.loads(provenance_path.read_text())
    frozen = {
        row["path"]: row["sha256"]
        for row in provenance["source_files"]
    }
    if (
        metadata.get("reference_initial_state") != str(seed)
        or metadata.get("resolution") != resolution
        or metadata.get("save_number") != saves
        or metadata.get("actual_wave_steps") != saves
        or metadata.get("time_step_factor") != factor
        or metadata.get("diagnostic_stop_after_save") != steps
        or metadata.get("wave_smbh_coupling") != "periodic_tsc_strang_momentum"
        or metadata.get("binary_integrator")
        != "joint_kick_drift_kick_spectral_momentum_v1"
        or metadata.get("compact_potential_layout") != "periodic_tsc_poisson_v1"
        or metadata.get("analytic_fdm_drag") is not False
        or config.get("Spatial Resolution") != resolution
        or config["Matter Particles"].get("Position Units") != "pc"
        or summary.get("status") != "diagnostic_partial"
        or summary.get("actual_wave_steps") != steps
        or conservation.get("status") != "diagnostic_partial"
        or conservation.get("samples") != steps + 1
        or provenance.get("status") != "source_snapshot"
        or provenance.get("run") != str(run)
        or provenance.get("input_records", {}).get("config_sha256")
        != sha256(config_path)
        or provenance.get("input_records", {}).get("fdm_adapter_metadata_sha256")
        != sha256(metadata_path)
        or metadata.get("reference_config_sha256")
        != sha256(seed / "config.uldm")
        or metadata.get("reference_metadata_sha256")
        != sha256(seed / "fdm_adapter_metadata.json")
        or metadata.get("reference_initial_wave_sha256")
        != sha256(seed / "Outputs/3Wfn/P3D_#000.npy")
        or metadata.get("reference_initial_particle_sha256")
        != sha256(seed / "Outputs/NBody/NTM_#000.npy")
        or metadata.get("solver_source_sha256") != frozen
    ):
        raise ValueError(f"spatial run is not the frozen short prefix: {run}")
    return metadata, config, conservation


def validate_fixed_physics(
    baseline_metadata: dict, baseline_config: dict,
    metadata: dict, config: dict, *, resolution: int,
) -> None:
    comparison = json.loads(json.dumps(config))
    comparison["Spatial Resolution"] = baseline_config["Spatial Resolution"]
    physical_keys = (
        "case_id", "box_size_pc", "particle_mass_ev", "pyul_length_unit_m",
        "pyul_mass_unit_kg", "pyul_time_unit_s", "mass_ratio_q",
        "initial_eccentricity", "initial_separation_pc", "semi_major_axis_pc",
        "plummer_radius_pc", "core_radius_reference_pc",
    )
    if (
        comparison != baseline_config
        or any(metadata.get(key) != baseline_metadata.get(key)
               for key in physical_keys)
        or not np.isclose(
            metadata.get("cell_size_pc", np.nan),
            baseline_metadata["box_size_pc"] / resolution, rtol=0, atol=1e-12,
        )
        or metadata.get("wave_time_step_code")
        != baseline_metadata.get("wave_time_step_code")
        or metadata.get("solver_source_sha256")
        != baseline_metadata.get("solver_source_sha256")
    ):
        raise ValueError("spatial comparison changes physics, source or time step")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("derived_run_root", type=Path)
    parser.add_argument("parent_seed", type=Path)
    parser.add_argument("derived_seed_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source_hashes = verified_source(project, args.source_commit)
    baseline_run = args.baseline_run.expanduser().resolve()
    derived_root = args.derived_run_root.expanduser().resolve()
    parent_seed = args.parent_seed.expanduser().resolve()
    seed_root = args.derived_seed_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace spatial audit: {output}")
    paths = {
        "n192": (derived_root / "n192_f0125", seed_root / "n192"),
        "n256": (baseline_run, parent_seed),
        "n384": (derived_root / "n384_f0125", seed_root / "n384"),
    }
    parent_hashes = None
    for label in ("n192", "n384"):
        seed = paths[label][1]
        manifest_path = seed / "spatial_resolution_seed_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status")
            != "spectrally_resampled_fixed_physics_seed_not_calibration"
            or manifest.get("calibration_eligible") is not False
            or manifest.get("resolution") != int(label[1:])
            or manifest.get("parent_reference") != str(parent_seed)
            or manifest.get("source_sha256")
            != source_hashes["scripts/derive_spatial_resolution_seed.py"]
            or abs(manifest.get("relative_wave_mass_change", np.inf)) > 1e-8
            or any(sha256(seed / key) != digest
                   for key, digest in manifest["derived_sha256"].items())
            or any(sha256(parent_seed / key) != digest
                   for key, digest in manifest["parent_sha256"].items())
        ):
            raise ValueError(f"spatial seed differs from source-bound manifest: {label}")
        if parent_hashes is None:
            parent_hashes = manifest["parent_sha256"]
        elif manifest["parent_sha256"] != parent_hashes:
            raise ValueError("spatial seeds do not share the same parent")
    records = {}
    for label, resolution, factor in GRID:
        run, seed = paths[label]
        metadata, config, conservation = verified_run(
            run, seed, resolution=resolution, factor=factor,
        )
        units = pyul_unit_system(metadata)
        initial = np.load(run / "Outputs/NBody/NTM_#000.npy").reshape(2, 6)
        body_hashes = {
            str(index): sha256(run / f"Outputs/NBody/NTM_#{index:03d}.npy")
            for index in range(STEPS + 1)
        }
        if body_hashes["0"] != metadata["reference_initial_particle_sha256"]:
            raise ValueError(f"spatial run initial SMBH state changed: {label}")
        coupled_sep = separation_series(run, stop=STEPS, length_pc=units.length_pc)
        masses = np.asarray([
            row[0] for row in config["Matter Particles"]["Condition"]
        ], dtype=float) / units.mass_msun
        isolated = isolated_plummer_kdk(
            initial, masses,
            config["Matter Particles"]["Plummer Radius"] / units.length_pc,
            metadata["wave_time_step_code"], STEPS,
        )
        isolated_sep = np.linalg.norm(
            isolated[:, 1, :3] - isolated[:, 0, :3], axis=1,
        ) * units.length_pc
        energy = float(conservation["max_total_energy_drift_over_energy_transfer"])
        mass_error = float(conservation["max_wave_mass_relative_error"])
        phase_path = run / "initial_compact_phase_jump.json"
        phase = json.loads(phase_path.read_text())
        phase_jump = float(phase["max_neighbour_phase_difference_over_pi"])
        if (
            not np.isfinite(energy) or energy < 0.0
            or not np.isfinite(mass_error) or mass_error > 1e-10
            or phase.get("wave_time_step_code") != metadata["wave_time_step_code"]
            or not np.isfinite(phase_jump) or not 0.0 <= phase_jump < 0.1
        ):
            raise ValueError(f"spatial conservation/phase summary invalid: {label}")
        records[label] = {
            "metadata": metadata, "config": config, "initial": initial,
            "coupled_separation_pc": coupled_sep,
            "isolated_separation_pc": isolated_sep,
            "initial_energy_components_code": initial_energy_components(run),
            "initial_compact_phase_jump_over_pi": phase_jump,
            "initial_compact_phase_jump_sha256": sha256(phase_path),
            "body_state_sha256": body_hashes,
            "conservation": {
                "max_energy_error_over_transfer": energy,
                "short_prefix_energy_limit_passed": energy <= 0.01,
                "max_wave_mass_relative_error": mass_error,
            },
            "run_input_sha256": {
                name: sha256(run / name)
                for name in (
                    "fdm_adapter_metadata.json", "config.uldm",
                    "torch_run_summary.json", "conservation_summary.json",
                    "torch_solver_provenance/manifest.json",
                )
            },
        }
    base = records["n256"]
    rows = []
    for label, resolution, factor in GRID:
        row = records[label]
        validate_fixed_physics(
            base["metadata"], base["config"], row["metadata"], row["config"],
            resolution=resolution,
        )
        if (
            not np.array_equal(row["initial"], base["initial"])
            or not np.allclose(row["isolated_separation_pc"],
                               base["isolated_separation_pc"], rtol=0, atol=1e-14)
        ):
            raise ValueError(f"spatial ladder changed direct SMBH trajectory: {label}")
        coupled = row["coupled_separation_pc"]
        isolated = row["isolated_separation_pc"]
        wave_associated = coupled - isolated
        rows.append({
            "label": label,
            "run_directory": str(paths[label][0]),
            "resolution": resolution,
            "cell_size_pc": row["metadata"]["cell_size_pc"],
            "time_step_factor": factor,
            "wave_time_step_code": row["metadata"]["wave_time_step_code"],
            "initial_separation_pc": float(coupled[0]),
            "endpoint_coupled_separation_pc": float(coupled[-1]),
            "endpoint_isolated_separation_pc": float(isolated[-1]),
            "endpoint_coupled_minus_isolated_pc": float(wave_associated[-1]),
            "endpoint_coupled_minus_n256_pc": float(
                coupled[-1] - base["coupled_separation_pc"][-1]
            ),
            "maximum_prefix_coupled_minus_n256_pc": float(np.max(np.abs(
                coupled - base["coupled_separation_pc"]
            ))),
            "initial_energy_components_code": row["initial_energy_components_code"],
            "initial_wave_kinetic_minus_n256_code": float(
                row["initial_energy_components_code"]["wave_kinetic"]
                - base["initial_energy_components_code"]["wave_kinetic"]
            ),
            "initial_compact_phase_jump_over_pi": row[
                "initial_compact_phase_jump_over_pi"
            ],
            "initial_compact_phase_jump_sha256": row[
                "initial_compact_phase_jump_sha256"
            ],
            "conservation": row["conservation"],
            "body_state_sha256": row["body_state_sha256"],
            "run_input_sha256": row["run_input_sha256"],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "n192_n256_n384_fixed_physics_tsc_prefix_diagnostic_not_calibration",
        "calibration_eligible": False,
        "source_commit": args.source_commit,
        "source_sha256": source_hashes,
        "parent_seed": str(parent_seed),
        "derived_seed_root": str(seed_root),
        "baseline_run": str(baseline_run),
        "derived_run_root": str(derived_root),
        "levels": rows,
        "interpretation": (
            "Same n256 wave seed spectrally resampled onto three periodic grids, "
            "with a fixed physical direct Plummer radius and identical SMBH "
            "initial state and actual time step. TSC kernel smoothing and "
            "kinetic-phase sampling both change with grid spacing; the "
            "time_step_factor arguments only control minimum-step padding. "
            "The n384 archive retains an older _f0125 directory suffix "
            "although its time_step_factor is 0.30. This is a 17-year "
            "mesh/start-up diagnostic, not a pure spatial truncation error; "
            "coupled-minus-isolated displacement includes smooth FDM field, "
            "live response and numerical coupling, not isolated wake drag. "
            "No full-orbit or continuum calibration is established."
        ),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output), "status": payload["status"],
        "endpoint_coupled_minus_n256_pc": {
            row["label"]: row["endpoint_coupled_minus_n256_pc"] for row in rows
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
