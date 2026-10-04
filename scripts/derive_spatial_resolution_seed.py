#!/usr/bin/env python3
"""Resample one registered wave seed for a fixed-physics TSC mesh diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from scipy.signal import resample

from fdm_smbh_delay.pyul import pyul_unit_system


SOURCE = "scripts/derive_spatial_resolution_seed.py"
INPUTS = (
    "Outputs/3Wfn/P3D_#000.npy",
    "Outputs/NBody/NTM_#000.npy",
    "config.uldm",
    "fdm_adapter_metadata.json",
    "reproducibility.uldm",
)
RESOLUTIONS = (192, 384)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spectral_resample_wave(wave: np.ndarray, resolution: int) -> np.ndarray:
    """Periodically interpolate/truncate each Cartesian spectral axis."""

    initial = np.asarray(wave)
    if (
        initial.ndim != 3 or len(set(initial.shape)) != 1
        or initial.shape[0] != 256 or initial.dtype != np.complex128
        or resolution not in RESOLUTIONS or not np.all(np.isfinite(initial))
    ):
        raise ValueError("registered spatial seed requires finite n256 complex128 wave")
    result = initial
    for axis in range(3):
        result = resample(result, resolution, axis=axis)
    if result.shape != (resolution,) * 3 or not np.all(np.isfinite(result)):
        raise ValueError("spectral resampling produced an invalid wave")
    return np.asarray(result, dtype=np.complex128)


def relative_wave_mass_change(wave: np.ndarray, changed: np.ndarray) -> float:
    original = np.sum(np.abs(wave) ** 2, dtype=np.float64) / wave.size
    resampled = np.sum(np.abs(changed) ** 2, dtype=np.float64) / changed.size
    if not np.isfinite(original) or original <= 0.0 or not np.isfinite(resampled):
        raise ValueError("spatial seed has invalid wave mass")
    return float((resampled - original) / original)


def derive_metadata_and_config(
    metadata: dict, config: dict, *, resolution: int,
    output_name: str, parent: Path, parent_sha256: dict,
    source_commit: str, source_sha256: str,
) -> tuple[dict, dict]:
    if (
        resolution not in RESOLUTIONS
        or metadata.get("case_id") != "qe_q030_e030_a020"
        or metadata.get("resolution") != 256
        or metadata.get("analytic_fdm_drag") is not False
        or config.get("Spatial Resolution") != 256
        or config["Simulation Box"].get("Length Units") != "pc"
        or config["Matter Particles"].get("Position Units") != "pc"
    ):
        raise ValueError("parent seed is not the registered physical setup")
    box_pc = float(metadata["box_size_pc"])
    radius_pc = float(config["Matter Particles"]["Plummer Radius"])
    if (
        not np.isfinite(box_pc) or box_pc <= 0.0
        or not np.isclose(metadata["cell_size_pc"], box_pc / 256, rtol=0, atol=1e-12)
        or not np.isclose(metadata["plummer_radius_pc"], radius_pc, rtol=0, atol=1e-12)
        or not np.isclose(radius_pc, box_pc / 512, rtol=0, atol=1e-12)
        or not np.isclose(config["Simulation Box"]["Box Length"], box_pc,
                          rtol=0, atol=1e-12)
    ):
        raise ValueError("parent cell, box and direct Plummer length are inconsistent")
    changed = json.loads(json.dumps(metadata))
    parent_design = changed.pop("qe_design_binding", None)
    changed["run_id"] = output_name
    changed["resolution"] = resolution
    changed["cell_size_pc"] = box_pc / resolution
    changed["spatial_resolution_binding"] = {
        "status": "spectrally_resampled_fixed_physics_seed_not_calibration",
        "parent_reference": str(parent),
        "parent_sha256": parent_sha256,
        "parent_qe_design_binding": parent_design,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "parent_resolution": 256,
        "resolution": resolution,
        "box_size_pc": box_pc,
        "direct_plummer_radius_pc": radius_pc,
        "resampling": "scipy_signal_periodic_spectral_each_axis_v1",
        "wave_compact_assignment": "periodic_tsc_poisson_v1",
        "calibration_eligible": False,
    }
    changed_config = json.loads(json.dumps(config))
    changed_config["Spatial Resolution"] = resolution
    return changed, changed_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=RESOLUTIONS, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("source commit must be a full Git revision")
    project = Path(__file__).resolve().parents[1]
    source_sha = sha256(project / SOURCE)
    committed = subprocess.run(
        ["git", "show", f"{args.source_commit}:{SOURCE}"], cwd=project,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if hashlib.sha256(committed.stdout).hexdigest() != source_sha:
        raise ValueError("spatial seed source differs from committed revision")
    reference = args.reference.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace spatial seed: {output}")
    hashes = {relative: sha256(reference / relative) for relative in INPUTS}
    metadata = json.loads((reference / "fdm_adapter_metadata.json").read_text())
    config = json.loads((reference / "config.uldm").read_text())
    changed_metadata, changed_config = derive_metadata_and_config(
        metadata, config, resolution=args.resolution, output_name=output.name,
        parent=reference, parent_sha256=hashes,
        source_commit=args.source_commit, source_sha256=source_sha,
    )
    units = pyul_unit_system(metadata)
    body = np.load(reference / INPUTS[1], mmap_mode="r")
    if body.shape != (12,) or body.dtype != np.float64:
        raise ValueError("parent SMBH state has wrong shape or precision")
    body_positions = body.reshape(2, 6)[:, :3] * units.length_pc
    config_positions = np.asarray([
        row[1] for row in config["Matter Particles"]["Condition"]
    ], dtype=float)
    if not np.allclose(body_positions, config_positions, rtol=0, atol=1e-9):
        raise ValueError("parent initial SMBH positions disagree with config")
    wave = np.load(reference / INPUTS[0])
    changed_wave = spectral_resample_wave(wave, args.resolution)
    mass_change = relative_wave_mass_change(wave, changed_wave)
    if abs(mass_change) > 1e-8:
        raise ValueError("spatial resampling changes wave mass by more than 1e-8")
    if any(sha256(reference / relative) != digest for relative, digest in hashes.items()):
        raise ValueError("parent seed changed during spatial derivation")
    output.mkdir(parents=True)
    (output / "Outputs/3Wfn").mkdir(parents=True)
    (output / "Outputs/NBody").mkdir(parents=True)
    np.save(output / INPUTS[0], changed_wave)
    shutil.copyfile(reference / INPUTS[1], output / INPUTS[1])
    shutil.copyfile(reference / INPUTS[4], output / INPUTS[4])
    (output / "config.uldm").write_text(json.dumps(changed_config, indent=2) + "\n")
    (output / "fdm_adapter_metadata.json").write_text(
        json.dumps(changed_metadata, indent=2, sort_keys=True) + "\n"
    )
    derived_hashes = {relative: sha256(output / relative) for relative in INPUTS}
    if derived_hashes[INPUTS[1]] != hashes[INPUTS[1]]:
        raise ValueError("derived SMBH state differs from parent")
    manifest = {
        "status": "spectrally_resampled_fixed_physics_seed_not_calibration",
        "calibration_eligible": False,
        "parent_reference": str(reference),
        "parent_sha256": hashes,
        "derived_sha256": derived_hashes,
        "source_commit": args.source_commit,
        "source_sha256": source_sha,
        "resolution": args.resolution,
        "cell_size_pc": metadata["box_size_pc"] / args.resolution,
        "direct_plummer_radius_pc": config["Matter Particles"]["Plummer Radius"],
        "relative_wave_mass_change": mass_change,
        "resampling": "scipy_signal_periodic_spectral_each_axis_v1",
    }
    (output / "spatial_resolution_seed_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "output": str(output), "resolution": args.resolution,
        "relative_wave_mass_change": mass_change,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
