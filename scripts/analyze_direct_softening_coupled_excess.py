#!/usr/bin/env python3
"""Compare live-wave prefixes with matched isolated Plummer KDK references.

The coupled-minus-isolated difference includes the smooth FDM field and its
response to the moving SMBHs. It is not an isolated wake/drag measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from fdm_smbh_delay.pyul import pyul_unit_system
from scripts.audit_direct_softening_step_trace import separation_series
from scripts.audit_periodic_offset_step_trace import sha256


SOURCE = "scripts/analyze_direct_softening_coupled_excess.py"
LEVELS = (("f100", 10), ("f0125", 80))
VARIANTS = ("baseline", "quarter", "full")


def plummer_acceleration(
    state: np.ndarray, masses: np.ndarray, radius: float,
) -> np.ndarray:
    """Return mutual accelerations in PyUL code units, with no wave force."""

    body = np.asarray(state, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    if (
        body.shape != (2, 6) or mass.shape != (2,)
        or not np.all(np.isfinite(body)) or not np.all(np.isfinite(mass))
        or np.any(mass <= 0.0) or not np.isfinite(radius) or radius <= 0.0
    ):
        raise ValueError("isolated Plummer acceleration needs finite positive inputs")
    displacement = body[1, :3] - body[0, :3]
    inverse_cube = (float(np.dot(displacement, displacement)) + radius**2) ** -1.5
    acceleration = np.empty((2, 3), dtype=np.float64)
    acceleration[0] = mass[1] * displacement * inverse_cube
    acceleration[1] = -mass[0] * displacement * inverse_cube
    return acceleration


def isolated_plummer_kdk(
    initial: np.ndarray, masses: np.ndarray, radius: float,
    time_step: float, steps: int,
) -> np.ndarray:
    """Save the isolated binary at every coupled wave-step endpoint."""

    if (
        not np.isfinite(time_step) or time_step <= 0.0
        or type(steps) is not int or steps < 1
    ):
        raise ValueError("isolated Plummer steps must be positive")
    state = np.asarray(initial, dtype=np.float64).copy()
    plummer_acceleration(state, masses, radius)
    history = np.empty((steps + 1, 2, 6), dtype=np.float64)
    history[0] = state
    for index in range(1, steps + 1):
        state[:, 3:] += 0.5 * time_step * plummer_acceleration(
            state, masses, radius,
        )
        state[:, :3] += time_step * state[:, 3:]
        state[:, 3:] += 0.5 * time_step * plummer_acceleration(
            state, masses, radius,
        )
        if not np.all(np.isfinite(state)):
            raise ValueError("isolated Plummer trajectory became nonfinite")
        history[index] = state
    return history


def read_bound_run(
    run: Path, *, expected_steps: int, input_hashes: dict,
    body_hashes: dict,
) -> tuple[dict, dict]:
    metadata_path = run / "fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    config = json.loads((run / "config.uldm").read_text())
    paths = {
        "metadata": metadata_path,
        "summary": run / "torch_run_summary.json",
        "conservation": run / "conservation_summary.json",
        "provenance": run / "torch_solver_provenance/manifest.json",
    }
    manifest = json.loads(paths["provenance"].read_text())
    if (
        any(sha256(path) != input_hashes.get(key) for key, path in paths.items())
        or sha256(run / "config.uldm")
        != manifest.get("input_records", {}).get("config_sha256")
        or set(body_hashes) != {str(index) for index in range(expected_steps + 1)}
        or any(
            sha256(run / f"Outputs/NBody/NTM_#{index:03d}.npy")
            != body_hashes.get(str(index))
            for index in range(expected_steps + 1)
        )
        or metadata.get("wave_smbh_coupling") != "periodic_tsc_strang_momentum"
        or metadata.get("binary_integrator")
        != "joint_kick_drift_kick_spectral_momentum_v1"
        or metadata.get("analytic_fdm_drag") is not False
        or metadata.get("resolution") != 256
        or metadata.get("actual_wave_steps") != metadata.get("save_number")
        or metadata.get("diagnostic_stop_after_save") != expected_steps
        or config["Matter Particles"].get("Position Units") != "pc"
        or sha256(run / "Outputs/NBody/NTM_#000.npy")
        != metadata.get("reference_initial_particle_sha256")
        or not np.isfinite(metadata.get("wave_time_step_code", np.nan))
        or metadata["wave_time_step_code"] <= 0.0
    ):
        raise ValueError(f"coupled run or audited inputs changed: {run}")
    return metadata, config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("control_root", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("source commit must be a full Git revision")
    if not re.fullmatch(r"[0-9a-f]{64}", args.audit_sha256):
        raise ValueError("parent audit must have a SHA-256 digest")
    project = Path(__file__).resolve().parents[1]
    source_hash = sha256(project / SOURCE)
    committed = subprocess.run(
        ["git", "show", f"{args.source_commit}:{SOURCE}"], cwd=project,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if hashlib.sha256(committed.stdout).hexdigest() != source_hash:
        raise ValueError("isolated-reference source differs from committed revision")
    audit_path = args.audit.expanduser().resolve()
    if sha256(audit_path) != args.audit_sha256:
        raise ValueError("parent softening audit differs")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status")
        != "n256_direct_binary_softening_prefix_diagnostic_not_calibration"
        or audit.get("calibration_eligible") is not False
        or [row["label"] for row in audit.get("levels", [])]
        != [label for label, _ in LEVELS]
    ):
        raise ValueError("parent is not the registered direct-softening audit")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace coupled-excess audit: {output}")
    result_levels = []
    for (level, steps), parent in zip(LEVELS, audit["levels"], strict=True):
        paths = {
            "baseline": args.baseline_root.expanduser().resolve() / level,
            "quarter": args.control_root.expanduser().resolve() / f"quarter_{level}",
            "full": args.control_root.expanduser().resolve() / f"full_{level}",
        }
        records = {}
        for label in VARIANTS:
            metadata, config = read_bound_run(
                paths[label], expected_steps=steps,
                input_hashes=parent["variants"][label]["run_input_sha256"],
                body_hashes=parent["variants"][label]["body_state_sha256"],
            )
            units = pyul_unit_system(metadata)
            initial = np.load(
                paths[label] / "Outputs/NBody/NTM_#000.npy",
            ).reshape(2, 6)
            masses = np.asarray([
                row[0] for row in config["Matter Particles"]["Condition"]
            ], dtype=float) / units.mass_msun
            radius_code = (
                float(config["Matter Particles"]["Plummer Radius"])
                / units.length_pc
            )
            isolated = isolated_plummer_kdk(
                initial, masses, radius_code,
                float(metadata["wave_time_step_code"]), steps,
            )
            isolated_sep = np.linalg.norm(
                isolated[:, 1, :3] - isolated[:, 0, :3], axis=1,
            ) * units.length_pc
            coupled_sep = separation_series(
                paths[label], stop=steps, length_pc=units.length_pc,
            )
            if (
                not np.isclose(coupled_sep[-1], parent["variants"][label][
                    "endpoint_separation_pc"], rtol=0, atol=1e-14)
                or not np.isclose(isolated_sep[0], coupled_sep[0], rtol=0, atol=1e-14)
            ):
                raise ValueError(f"audited endpoint or initial state differs: {label}")
            records[label] = {
                "coupled": coupled_sep,
                "isolated": isolated_sep,
                "time_step_code": metadata["wave_time_step_code"],
                "masses_code": masses.tolist(),
                "plummer_radius_code": radius_code,
            }
        reference = records["baseline"]
        if any(
            records[label]["time_step_code"] != reference["time_step_code"]
            or records[label]["masses_code"] != reference["masses_code"]
            or not np.isclose(records[label]["coupled"][0],
                              reference["coupled"][0], rtol=0, atol=1e-14)
            for label in VARIANTS
        ):
            raise ValueError(f"softening controls are not time/mass matched: {level}")
        variants = {}
        for label in VARIANTS:
            coupled = records[label]["coupled"]
            isolated = records[label]["isolated"]
            coupled_effect = coupled - reference["coupled"]
            isolated_effect = isolated - reference["isolated"]
            excess = coupled_effect - isolated_effect
            variants[label] = {
                "coupled_endpoint_separation_pc": float(coupled[-1]),
                "isolated_endpoint_separation_pc": float(isolated[-1]),
                "coupled_minus_isolated_endpoint_pc": float(coupled[-1] - isolated[-1]),
                "softening_coupled_minus_baseline_endpoint_pc": float(coupled_effect[-1]),
                "softening_isolated_minus_baseline_endpoint_pc": float(isolated_effect[-1]),
                "softening_coupled_excess_endpoint_pc": float(excess[-1]),
                "maximum_absolute_prefix_coupled_excess_pc": float(np.max(np.abs(excess))),
                "body_state_sha256": parent["variants"][label]["body_state_sha256"],
            }
        result_levels.append({
            "label": level,
            "wave_steps": steps,
            "time_step_code": reference["time_step_code"],
            "variants": variants,
        })
    payload = {
        "status": "n256_direct_softening_isolated_reference_diagnostic_not_calibration",
        "calibration_eligible": False,
        "parent_audit": str(audit_path),
        "parent_audit_sha256": args.audit_sha256,
        "source_commit": args.source_commit,
        "source_sha256": source_hash,
        "levels": result_levels,
        "interpretation": (
            "This difference of differences removes the common smooth-field "
            "contribution and retains trajectory-dependent live-wave response "
            "and numerical coupling cross terms. Without a quantified error "
            "floor it is not a resolved physical effect or isolated FDM drag, a "
            "spatial-convergence bound, or a calibration release."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
