#!/usr/bin/env python3
"""Derive a source-bound translated PyUL seed for a periodic mesh offset gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import torch

from fdm_smbh_delay.pyul import pyul_unit_system


SOURCE = "scripts/derive_periodic_offset_seed.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shifted_wave(
    wave: torch.Tensor, *, axis: int, shift_cells: float,
) -> torch.Tensor:
    """Translate a periodic complex field by integer roll or Fourier phase."""

    if (
        wave.ndim != 3 or len(set(wave.shape)) != 1
        or wave.dtype != torch.complex128 or axis not in (0, 1, 2)
        or shift_cells not in (0.5, 1.0)
    ):
        raise ValueError("offset seed needs a cubic complex128 wave and approved shift")
    if shift_cells == 1.0:
        return torch.roll(wave, shifts=1, dims=axis)
    resolution = wave.shape[axis]
    cycles = torch.fft.fftfreq(
        resolution, d=1.0, dtype=torch.float64, device=wave.device,
    )
    phase = torch.polar(
        torch.ones_like(cycles), -2.0 * torch.pi * cycles * shift_cells,
    )
    shape = [1, 1, 1]
    shape[axis] = resolution
    return torch.fft.ifftn(torch.fft.fftn(wave) * phase.reshape(shape))


def shifted_particle_state(
    state: np.ndarray, *, axis: int, shift_code: float,
    box_code: float,
) -> np.ndarray:
    result = np.asarray(state, dtype=np.float64).copy().reshape(2, 6)
    if (
        axis not in (0, 1, 2) or not np.isfinite(shift_code)
        or not np.isfinite(box_code) or box_code <= 0.0
        or not np.all(np.isfinite(result))
    ):
        raise ValueError("offset particle state is invalid")
    result[:, axis] += shift_code
    if np.any(np.abs(result[:, :3]) >= 0.5 * box_code):
        raise ValueError("offset particles leave nonperiodic binary domain")
    return result.reshape(12)


def shifted_config(
    config: dict, *, axis: int, shift_pc: float,
    original_state: np.ndarray, units,
) -> dict:
    result = json.loads(json.dumps(config))
    conditions = result["Matter Particles"]["Condition"]
    if len(conditions) != 2:
        raise ValueError("offset seed requires two compact masses")
    positions = np.asarray([item[1] for item in conditions], dtype=float)
    if not np.allclose(
        positions, original_state.reshape(2, 6)[:, :3] * units.length_pc,
        rtol=0, atol=1e-9,
    ):
        raise ValueError("parent config and NBody positions disagree")
    for item in conditions:
        item[1][axis] += shift_pc
    solitons = result["ULDM Solitons"]
    if len(solitons["Condition"]) != 1 or solitons["Embedded"]:
        raise ValueError("offset seed requires one non-embedded soliton")
    soliton_position = solitons["Condition"][0][1]
    if len(soliton_position) != 3 or not np.all(np.isfinite(soliton_position)):
        raise ValueError("parent soliton position is invalid")
    soliton_position[axis] += shift_pc
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--shift-cells", type=float, choices=(0.5, 1.0),
                        required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
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
        raise ValueError("offset source differs from registered commit")
    reference = args.reference.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace offset seed: {output}")
    wave_path = reference / "Outputs/3Wfn/P3D_#000.npy"
    body_path = reference / "Outputs/NBody/NTM_#000.npy"
    config_path = reference / "config.uldm"
    metadata_path = reference / "fdm_adapter_metadata.json"
    reproducibility_path = reference / "reproducibility.uldm"
    paths = (wave_path, body_path, config_path, metadata_path,
             reproducibility_path)
    parent_hashes = {path.relative_to(reference).as_posix(): sha256(path)
                     for path in paths}
    metadata = json.loads(metadata_path.read_text())
    config = json.loads(config_path.read_text())
    resolution = int(metadata["resolution"])
    box_pc = float(metadata["box_size_pc"])
    if (
        resolution < 4 or not np.isfinite(box_pc) or box_pc <= 0.0
        or metadata.get("case_id") != "qe_q030_e030_a020"
        or metadata.get("analytic_fdm_drag") is not False
    ):
        raise ValueError("offset parent is not the approved q/e/a seed")
    units = pyul_unit_system(metadata)
    cell_pc = box_pc / resolution
    if not np.isclose(cell_pc, metadata["cell_size_pc"], rtol=0, atol=1e-12):
        raise ValueError("parent cell length is inconsistent")
    axis = "xyz".index(args.axis)
    shift_pc = cell_pc * args.shift_cells
    shift_code = shift_pc / units.length_pc
    body = np.load(body_path)
    if body.shape != (12,) or body.dtype != np.float64:
        raise ValueError("parent NBody array shape or dtype differs")
    translated_body = shifted_particle_state(
        body, axis=axis, shift_code=shift_code,
        box_code=box_pc / units.length_pc,
    )
    translated_config = shifted_config(
        config, axis=axis, shift_pc=shift_pc,
        original_state=body, units=units,
    )
    device = torch.device(args.device)
    if resolution == 256 and (
        device.type != "cuda" or not torch.cuda.is_available()
    ):
        raise ValueError("n256 offset seed requires an allocated CUDA GPU")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    original_wave = np.load(wave_path)
    if original_wave.shape != (resolution,) * 3 or original_wave.dtype != np.complex128:
        raise ValueError("parent wave shape or dtype differs")
    wave = torch.as_tensor(original_wave, dtype=torch.complex128, device=device)
    del original_wave
    translated_wave = shifted_wave(
        wave, axis=axis, shift_cells=args.shift_cells,
    )
    mass_before = float(torch.sum(wave.real.square() + wave.imag.square()))
    mass_after = float(torch.sum(
        translated_wave.real.square() + translated_wave.imag.square()
    ))
    mass_change = abs(mass_after - mass_before) / mass_before
    if not np.isfinite(mass_change) or mass_change > 1e-12:
        raise ValueError("offset wave translation changes spectral mass")
    if any(sha256(path) != parent_hashes[path.relative_to(reference).as_posix()]
           for path in paths):
        raise ValueError("parent seed changed during offset construction")
    translated_metadata = dict(metadata)
    translated_metadata["run_id"] = output.name
    translated_metadata["periodic_offset_binding"] = {
        "status": "source_bound_periodic_offset_seed_not_a_calibration_release",
        "parent_reference": str(reference),
        "parent_sha256": parent_hashes,
        "source_commit": args.source_commit,
        "source_sha256": source_hash,
        "axis": args.axis,
        "shift_cells": args.shift_cells,
        "shift_pc": shift_pc,
        "shift_code": shift_code,
        "wave_translation": (
            "torch_roll_exact_integer_cell"
            if args.shift_cells == 1.0 else "periodic_fourier_phase"
        ),
    }
    output.mkdir(parents=True)
    (output / "Outputs/3Wfn").mkdir(parents=True)
    (output / "Outputs/NBody").mkdir(parents=True)
    np.save(output / "Outputs/3Wfn/P3D_#000.npy",
            translated_wave.detach().cpu().numpy())
    np.save(output / "Outputs/NBody/NTM_#000.npy", translated_body)
    (output / "config.uldm").write_text(
        json.dumps(translated_config, indent=2) + "\n"
    )
    (output / "fdm_adapter_metadata.json").write_text(
        json.dumps(translated_metadata, indent=2, sort_keys=True) + "\n"
    )
    shutil.copyfile(reproducibility_path, output / "reproducibility.uldm")
    derived_paths = (
        output / "Outputs/3Wfn/P3D_#000.npy",
        output / "Outputs/NBody/NTM_#000.npy",
        output / "config.uldm",
        output / "fdm_adapter_metadata.json",
        output / "reproducibility.uldm",
    )
    manifest = {
        "status": "source_bound_periodic_offset_seed_not_a_calibration_release",
        "calibration_eligible": False,
        "parent_reference": str(reference),
        "parent_sha256": parent_hashes,
        "source_commit": args.source_commit,
        "source_sha256": source_hash,
        "axis": args.axis,
        "shift_cells": args.shift_cells,
        "shift_pc": shift_pc,
        "shift_code": shift_code,
        "wave_mass_relative_change": mass_change,
        "derived_sha256": {
            path.relative_to(output).as_posix(): sha256(path)
            for path in derived_paths
        },
        "peak_device_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (output / "offset_seed_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "output": str(output), "shift_cells": args.shift_cells,
        "wave_mass_relative_change": mass_change,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
