#!/usr/bin/env python3
"""Audit fixed-grid translation controls for an n256 short coupled prefix."""

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


SOURCE = "scripts/audit_periodic_offset_step_trace.py"
LEVELS = (("f100", 1.0, 10, 58500), ("f0125", 0.125, 80, 468000))
SEED_STATUS = "source_bound_periodic_offset_seed_not_a_calibration_release"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_rolled_wave_difference(base_path: Path, shifted_path: Path) -> tuple[float, float]:
    """Compare whole-cell translation slabwise without materializing an n256 cube."""

    base = np.load(base_path, mmap_mode="r")
    shifted = np.load(shifted_path, mmap_mode="r")
    if (
        base.shape != shifted.shape or base.ndim != 3
        or len(set(base.shape)) != 1
        or base.dtype != np.complex128 or shifted.dtype != np.complex128
    ):
        raise ValueError("translated checkpoint wave shape or dtype differs")
    maximum_difference = 0.0
    maximum_amplitude = 0.0
    for start in range(0, base.shape[0], 8):
        stop = min(start + 8, base.shape[0])
        indices = (np.arange(start, stop) - 1) % base.shape[0]
        expected = base[indices]
        actual = shifted[start:stop]
        maximum_difference = max(maximum_difference,
                                 float(np.max(np.abs(actual - expected))))
        maximum_amplitude = max(maximum_amplitude,
                                float(np.max(np.abs(expected))))
    if maximum_amplitude <= 0.0:
        raise ValueError("translated checkpoint wave has zero amplitude")
    return maximum_difference, maximum_difference / maximum_amplitude


def compare_bodies(
    base_run: Path, whole_run: Path, half_run: Path, *, stop: int,
    whole_shift_code: float, length_pc: float,
) -> dict:
    maximum_body_difference = 0.0
    maximum_whole_separation_difference = 0.0
    maximum_half_separation_difference = 0.0
    endpoint = None
    for index in range(stop + 1):
        relative = f"Outputs/NBody/NTM_#{index:03d}.npy"
        states = [np.load(run / relative).reshape(2, 6)
                  for run in (base_run, whole_run, half_run)]
        if any(not np.all(np.isfinite(state)) for state in states):
            raise ValueError("offset body state is nonfinite")
        base, whole, half = states
        translated_whole = whole.copy()
        translated_whole[:, 0] -= whole_shift_code
        maximum_body_difference = max(
            maximum_body_difference,
            float(np.max(np.abs(base - translated_whole))),
        )
        separations = [
            float(np.linalg.norm(state[1, :3] - state[0, :3]) * length_pc)
            for state in states
        ]
        maximum_whole_separation_difference = max(
            maximum_whole_separation_difference,
            abs(separations[1] - separations[0]),
        )
        maximum_half_separation_difference = max(
            maximum_half_separation_difference,
            abs(separations[2] - separations[0]),
        )
        if index == stop:
            endpoint = separations
    if maximum_body_difference > 1e-10:
        raise ValueError("whole-cell SMBH trajectory violates translation gate")
    assert endpoint is not None
    return {
        "maximum_whole_cell_body_difference_code": maximum_body_difference,
        "maximum_whole_cell_separation_difference_pc": (
            maximum_whole_separation_difference
        ),
        "maximum_half_cell_separation_difference_pc": (
            maximum_half_separation_difference
        ),
        "endpoint_base_separation_pc": endpoint[0],
        "endpoint_whole_cell_separation_pc": endpoint[1],
        "endpoint_half_cell_separation_pc": endpoint[2],
        "endpoint_half_minus_base_separation_pc": endpoint[2] - endpoint[0],
    }


def read_run(run: Path, *, factor: float, stop: int, saves: int) -> dict:
    metadata_path = run / "fdm_adapter_metadata.json"
    summary_path = run / "torch_run_summary.json"
    conservation_path = run / "conservation_summary.json"
    manifest_path = run / "torch_solver_provenance/manifest.json"
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads(summary_path.read_text())
    conservation = json.loads(conservation_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (
        metadata.get("case_id") != "qe_q030_e030_a020"
        or metadata.get("resolution") != 256
        or metadata.get("wave_smbh_coupling") != "periodic_tsc_strang_momentum"
        or metadata.get("time_step_factor") != factor
        or metadata.get("save_number") != saves
        or metadata.get("diagnostic_stop_after_save") != stop
        or metadata.get("analytic_fdm_drag") is not False
        or summary.get("status") != "diagnostic_partial"
        or summary.get("actual_wave_steps") != stop
        or conservation.get("status") != "diagnostic_partial"
        or conservation.get("samples") != stop + 1
        or manifest.get("status") != "source_snapshot"
        or manifest.get("run") != str(run)
        or manifest.get("input_records", {}).get("fdm_adapter_metadata_sha256")
        != sha256(metadata_path)
        or manifest.get("input_records", {}).get("config_sha256")
        != sha256(run / "config.uldm")
    ):
        raise ValueError(f"offset run is not a registered short prefix: {run}")
    return {
        "metadata": metadata,
        "summary": summary,
        "conservation": conservation,
        "hashes": {
            "metadata": sha256(metadata_path),
            "summary": sha256(summary_path),
            "conservation": sha256(conservation_path),
            "provenance": sha256(manifest_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("offset_root", type=Path)
    parser.add_argument("seed_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
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
        raise ValueError("offset audit source differs from registered commit")
    baseline_root = args.baseline_root.expanduser().resolve()
    offset_root = args.offset_root.expanduser().resolve()
    seed_root = args.seed_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace offset audit: {output}")
    seed_records = {}
    for label, fraction in (("whole_x", 1.0), ("half_x", 0.5)):
        seed = seed_root / label
        manifest_path = seed / "offset_seed_manifest.json"
        record = json.loads(manifest_path.read_text())
        if (
            record.get("status") != SEED_STATUS
            or record.get("calibration_eligible") is not False
            or record.get("shift_cells") != fraction
            or record.get("axis") != "x"
            or record.get("wave_mass_relative_change", 1.0) > 1e-12
            or any(
                sha256(seed / relative) != digest
                for relative, digest in record["derived_sha256"].items()
            )
        ):
            raise ValueError(f"offset seed manifest differs: {label}")
        seed_records[label] = {
            "path": str(seed), "manifest_sha256": sha256(manifest_path),
            "shift_code": float(record["shift_code"]),
        }
    if not np.isclose(seed_records["half_x"]["shift_code"] * 2.0,
                      seed_records["whole_x"]["shift_code"], rtol=0, atol=1e-20):
        raise ValueError("whole- and half-cell seed shifts disagree")
    results = []
    common_source = None
    for label, factor, stop, saves in LEVELS:
        baseline = baseline_root / label
        whole = offset_root / f"whole_{label}"
        half = offset_root / f"half_{label}"
        run_records = {
            name: read_run(path, factor=factor, stop=stop, saves=saves)
            for name, path in (("base", baseline), ("whole", whole),
                               ("half", half))
        }
        reference = run_records["base"]["metadata"]
        numerical_source = reference["solver_source_sha256"]
        if common_source is None:
            common_source = numerical_source
        for name, seed_label in (("whole", "whole_x"), ("half", "half_x")):
            metadata = run_records[name]["metadata"]
            binding = metadata.get("periodic_offset_binding", {})
            if (
                metadata.get("solver_source_sha256") != numerical_source
                or numerical_source != common_source
                or metadata.get("reference_initial_state")
                != seed_records[seed_label]["path"]
                or binding.get("shift_cells")
                != (1.0 if name == "whole" else 0.5)
                or binding.get("shift_code")
                != seed_records[seed_label]["shift_code"]
            ):
                raise ValueError(f"offset solver or seed identity differs: {name}")
        length_pc = pyul_unit_system(reference).length_pc
        body = compare_bodies(
            baseline, whole, half, stop=stop,
            whole_shift_code=seed_records["whole_x"]["shift_code"],
            length_pc=length_pc,
        )
        wave_max, wave_relative = max_rolled_wave_difference(
            baseline / "Checkpoints" / f"wave_{stop:06d}.npy",
            whole / "Checkpoints" / f"wave_{stop:06d}.npy",
        )
        if wave_relative > 1e-10:
            raise ValueError("whole-cell wave checkpoint violates translation gate")
        energy = {
            name: float(record["conservation"][
                "max_total_energy_drift_over_energy_transfer"
            ])
            for name, record in run_records.items()
        }
        mass = {
            name: float(record["conservation"]["max_wave_mass_relative_error"])
            for name, record in run_records.items()
        }
        if (
            any(not np.isfinite(value) for value in (*energy.values(), *mass.values()))
            or abs(energy["whole"] - energy["base"]) > 1e-7
            or any(value > 1e-10 for value in mass.values())
        ):
            raise ValueError("offset energy or mass conservation gate differs")
        results.append({
            "label": label, "time_step_factor": factor,
            "wave_steps": stop, **body,
            "whole_cell_wave_checkpoint_max_difference": wave_max,
            "whole_cell_wave_checkpoint_relative_difference": wave_relative,
            "max_energy_error_over_transfer": energy,
            "max_wave_mass_relative_error": mass,
            "run_input_sha256": {
                name: record["hashes"] for name, record in run_records.items()
            },
        })
    offset_change = abs(
        results[0]["endpoint_half_minus_base_separation_pc"]
        - results[1]["endpoint_half_minus_base_separation_pc"]
    )
    payload = {
        "status": "n256_short_prefix_offset_diagnostic_not_a_calibration_release",
        "calibration_eligible": False,
        "source_commit": args.source_commit,
        "source_sha256": source_hash,
        "seed_records": seed_records,
        "levels": results,
        "half_cell_offset_temporal_difference_pc": offset_change,
        "interpretation": (
            "Whole-cell translation symmetry passes on this fixed n256 "
            "prefix. Half-cell offset sensitivity is measured, not an "
            "orbital-decay or spatial-convergence error bar."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(temporary, output)
    temporary.unlink()
    print(json.dumps({"status": payload["status"],
                      "half_cell_temporal_difference_pc": offset_change}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
