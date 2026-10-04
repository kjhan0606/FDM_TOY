#!/usr/bin/env python3
"""Audit a completed n256 live-wave orbit prefix without calibration release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from fdm_smbh_delay.pyul import pyul_unit_system
from scripts.audit_periodic_offset_step_trace import sha256


SOURCE = "scripts/audit_one_orbit_prefix.py"
SEED_MANIFEST_SHA256 = (
    "7a51b8df7bb6f360244e8f83ab6684b0bf41ed431c3c24fd336c116b344416f8"
)
EXPECTED_SOLVER_HASHES = {
    "scripts/run_torch_wave_case.py":
        "b5de6d1f9e964e3b8e0eca130b8e88bc4e54d6c7bdfc32e5f651527fd24c1156",
    "src/fdm_smbh_delay/torch_wave.py":
        "3b4952cea5b552265af854068ef08f9fb1c3d19b2863a925bb9af37606272d06",
    "src/fdm_smbh_delay/periodic_mesh_coupling.py":
        "fbf928849179bb701d7819d6cfd60d988bb2a82f81b2fa451e0e5e7a656f09a9",
    "src/fdm_smbh_delay/pyul.py":
        "e531203ac43809df3c13c0b05c87cf7534197207fab53660c683b4e2afe786d1",
}
SAVES = 18
STEPS = 16686
DURATION_MYR = 0.0036


def orbital_coverage(states: np.ndarray, length_pc: float) -> dict[str, float | int | bool]:
    body = np.asarray(states, dtype=np.float64)
    if (
        body.ndim != 3 or body.shape[1:] != (2, 6)
        or body.shape[0] < 3 or not np.all(np.isfinite(body))
        or not np.isfinite(length_pc) or length_pc <= 0.0
    ):
        raise ValueError("orbital coverage needs finite two-SMBH states")
    relative = body[:, 1, :3] - body[:, 0, :3]
    projected = np.linalg.norm(relative[:, :2], axis=1)
    separation = np.linalg.norm(relative, axis=1) * length_pc
    if np.any(projected <= 0.0) or np.any(separation <= 0.0):
        raise ValueError("relative SMBH separation is zero")
    angle = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
    increments = np.diff(angle)
    maximum_increment = float(np.max(np.abs(increments)))
    if maximum_increment >= np.pi / 2:
        raise ValueError("saved orbital phase is undersampled")
    turn = float(abs(angle[-1] - angle[0]) / (2.0 * np.pi))
    radial_change = np.diff(separation)
    turning_indices = np.flatnonzero(
        np.sign(radial_change[:-1]) != np.sign(radial_change[1:])
    ) + 1
    return {
        "projected_azimuthal_turns": turn,
        "one_projected_turn_reached": turn >= 1.0,
        "maximum_saved_phase_advance_radians": maximum_increment,
        "initial_separation_pc": float(separation[0]),
        "final_separation_pc": float(separation[-1]),
        "minimum_saved_separation_pc": float(np.min(separation)),
        "maximum_saved_separation_pc": float(np.max(separation)),
        "sampled_radial_turning_points": int(turning_indices.size),
    }


def validate_completed_checkpoint(
    wave_path: Path, state_path: Path, final_state: np.ndarray,
    *, resolution: int, step: int, save_index: int,
) -> None:
    wave = np.load(wave_path, mmap_mode="r")
    if wave.shape != (resolution,) * 3 or wave.dtype != np.complex128:
        raise ValueError("completed checkpoint wave has wrong shape or precision")
    for start in range(0, resolution, 16):
        if not np.all(np.isfinite(wave[start:start + 16])):
            raise ValueError("completed checkpoint wave contains nonfinite values")
    with np.load(state_path) as saved:
        body = np.asarray(saved["state"], dtype=np.float64)
        saved_step = int(saved["step"])
        saved_index = int(saved["save_index"])
    if (
        body.shape != (12,) or not np.array_equal(body, final_state.reshape(12))
        or saved_step != step or saved_index != save_index
    ):
        raise ValueError("completed checkpoint state differs from final save")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("seed", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("source commit must be a full Git revision")
    project = Path(__file__).resolve().parents[1]
    source_hash = sha256(project / SOURCE)
    committed = subprocess.run(
        ["git", "show", f"{args.source_commit}:{SOURCE}"], cwd=project,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if hashlib.sha256(committed.stdout).hexdigest() != source_hash:
        raise ValueError("one-orbit audit source differs from committed revision")
    run = args.run.expanduser().resolve()
    seed = args.seed.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace orbit audit: {output}")
    seed_manifest_path = seed / "orbit_prefix_seed_manifest.json"
    if sha256(seed_manifest_path) != SEED_MANIFEST_SHA256:
        raise ValueError("orbit seed manifest differs from registered input")
    seed_manifest = json.loads(seed_manifest_path.read_text())
    if (
        seed_manifest.get("calibration_eligible") is not False
        or any(sha256(seed / relative) != digest
               for relative, digest in seed_manifest["derived_sha256"].items())
        or any(sha256(Path(seed_manifest["parent_reference"]) / relative) != digest
               for relative, digest in seed_manifest["parent_sha256"].items())
    ):
        raise ValueError("orbit seed or parent differs from source-bound manifest")
    metadata_path = run / "fdm_adapter_metadata.json"
    config_path = run / "config.uldm"
    summary_path = run / "torch_run_summary.json"
    conservation_path = run / "conservation_summary.json"
    provenance_path = run / "torch_solver_provenance/manifest.json"
    checkpoint_path = run / "Checkpoints/latest.json"
    metadata = json.loads(metadata_path.read_text())
    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())
    conservation = json.loads(conservation_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    source_files = {
        row["path"]: row["sha256"] for row in provenance["source_files"]
    }
    seed_metadata = json.loads((seed / "fdm_adapter_metadata.json").read_text())
    if (
        metadata.get("reference_initial_state") != str(seed)
        or metadata.get("case_id") != "qe_q030_e030_a020"
        or metadata.get("resolution") != 256
        or metadata.get("wave_smbh_coupling") != "periodic_tsc_strang_momentum"
        or metadata.get("binary_integrator")
        != "joint_kick_drift_kick_spectral_momentum_v1"
        or metadata.get("compact_potential_layout") != "periodic_tsc_poisson_v1"
        or metadata.get("experimental_coupling_not_a_calibration_release") is not True
        or metadata.get("analytic_fdm_drag") is not False
        or metadata.get("orbit_prefix_binding")
        != seed_metadata.get("orbit_prefix_binding")
        or metadata.get("qe_design_binding") is not None
        or metadata.get("save_number") != SAVES
        or metadata.get("actual_wave_steps") != STEPS
        or not np.isclose(metadata.get("duration_myr", np.nan), DURATION_MYR,
                          rtol=0, atol=1e-15)
        or summary.get("status") != "diagnostic_complete"
        or summary.get("actual_wave_steps") != STEPS
        or summary.get("saved_intervals") != SAVES
        or conservation.get("status") != "diagnostic_complete"
        or conservation.get("samples") != SAVES + 1
        or provenance.get("status") != "source_snapshot"
        or provenance.get("run") != str(run)
        or provenance.get("input_records", {}).get("config_sha256")
        != sha256(config_path)
        or provenance.get("input_records", {}).get("fdm_adapter_metadata_sha256")
        != sha256(metadata_path)
        or metadata.get("reference_config_sha256") != sha256(seed / "config.uldm")
        or metadata.get("reference_metadata_sha256")
        != sha256(seed / "fdm_adapter_metadata.json")
        or metadata.get("reference_initial_wave_sha256")
        != sha256(seed / "Outputs/3Wfn/P3D_#000.npy")
        or metadata.get("reference_initial_particle_sha256")
        != sha256(seed / "Outputs/NBody/NTM_#000.npy")
        or metadata.get("solver_source_sha256") != EXPECTED_SOLVER_HASHES
        or source_files != EXPECTED_SOLVER_HASHES
        or config.get("Spatial Resolution") != 256
        or config["Save Options"].get("Number") != SAVES
        or config["Matter Particles"].get("Position Units") != "pc"
        or checkpoint.get("step") != STEPS
        or checkpoint.get("save_index") != SAVES
    ):
        raise ValueError("one-orbit run differs from registered source-bound design")
    checkpoint_wave = run / "Checkpoints" / checkpoint["wave"]
    checkpoint_state = run / "Checkpoints" / checkpoint["state"]
    if not checkpoint_wave.is_file() or not checkpoint_state.is_file():
        raise ValueError("completed orbit checkpoint files are missing")
    units = pyul_unit_system(metadata)
    states = np.asarray([
        np.load(run / f"Outputs/NBody/NTM_#{index:03d}.npy").reshape(2, 6)
        for index in range(SAVES + 1)
    ])
    validate_completed_checkpoint(
        checkpoint_wave, checkpoint_state, states[-1],
        resolution=256, step=STEPS, save_index=SAVES,
    )
    coverage = orbital_coverage(states, units.length_pc)
    energy = float(conservation["max_total_energy_drift_over_energy_transfer"])
    mass_error = float(conservation["max_wave_mass_relative_error"])
    if (
        not np.isfinite(energy) or energy < 0.0
        or not np.isfinite(mass_error) or mass_error > 1e-10
    ):
        raise ValueError("one-orbit conservation summary is invalid")
    payload = {
        "status": "source_bound_one_orbit_diagnostic_not_calibration",
        "calibration_eligible": False,
        "source_commit": args.source_commit,
        "source_sha256": source_hash,
        "run": str(run),
        "seed": str(seed),
        "seed_manifest_sha256": SEED_MANIFEST_SHA256,
        "coverage": coverage,
        "one_orbit_coverage_gate_passed": coverage["one_projected_turn_reached"],
        "energy_error_over_transfer": energy,
        "short_orbit_energy_gate_passed": energy <= 0.01,
        "wave_mass_relative_error": mass_error,
        "body_state_sha256": {
            str(index): sha256(run / f"Outputs/NBody/NTM_#{index:03d}.npy")
            for index in range(SAVES + 1)
        },
        "checkpoint_sha256": {
            "wave": sha256(checkpoint_wave),
            "state": sha256(checkpoint_state),
            "latest": sha256(checkpoint_path),
        },
        "run_input_sha256": {
            relative: sha256(run / relative)
            for relative in (
                "fdm_adapter_metadata.json", "config.uldm",
                "torch_run_summary.json", "conservation_summary.json",
                "torch_solver_provenance/manifest.json",
            )
        },
        "interpretation": (
            "Projected azimuthal coverage and short-orbit energy are diagnostics "
            "only. Momentum, spatial convergence, repeat-GPU floor, and "
            "full q/e calibration gates remain unproven. No FDM delay row "
            "is released."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "projected_azimuthal_turns": coverage["projected_azimuthal_turns"],
        "energy_error_over_transfer": energy,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
