#!/usr/bin/env python3
"""Derive immutable matched seeds differing only in direct-binary Plummer length."""

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


SOURCE = "scripts/derive_direct_softening_seed.py"
INPUTS = (
    "Outputs/3Wfn/P3D_#000.npy",
    "Outputs/NBody/NTM_#000.npy",
    "config.uldm",
    "fdm_adapter_metadata.json",
    "reproducibility.uldm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_config_and_metadata(
    config: dict, metadata: dict, *, cell_fraction: float,
    output_name: str, parent: Path, parent_hashes: dict[str, str],
    source_commit: str, source_hash: str,
) -> tuple[dict, dict, float]:
    if cell_fraction not in (0.25, 1.0):
        raise ValueError("only quarter- and one-cell direct softening are registered")
    resolution = metadata.get("resolution")
    box_pc = metadata.get("box_size_pc")
    if (
        metadata.get("case_id") != "qe_q030_e030_a020"
        or type(resolution) is not int or resolution != 256
        or not np.isfinite(box_pc) or box_pc <= 0.0
        or metadata.get("analytic_fdm_drag") is not False
    ):
        raise ValueError("direct-softening seed parent is incompatible")
    cell_pc = float(box_pc) / resolution
    original_radius = float(config["Matter Particles"]["Plummer Radius"])
    if (
        not np.isclose(original_radius, 0.5 * cell_pc, rtol=0, atol=1e-12)
        or not np.isclose(metadata.get("plummer_radius_pc"), original_radius,
                          rtol=0, atol=1e-12)
    ):
        raise ValueError("parent direct-binary softening is not half a cell")
    radius_pc = cell_fraction * cell_pc
    new_config = json.loads(json.dumps(config))
    new_config["Matter Particles"]["Plummer Radius"] = radius_pc
    new_metadata = dict(metadata)
    new_metadata["run_id"] = output_name
    new_metadata["plummer_radius_pc"] = radius_pc
    new_metadata["direct_softening_binding"] = {
        "status": "source_bound_direct_binary_softening_seed_not_calibration",
        "parent_reference": str(parent),
        "parent_sha256": parent_hashes,
        "source_commit": source_commit,
        "source_sha256": source_hash,
        "parent_radius_pc": original_radius,
        "radius_pc": radius_pc,
        "radius_over_cell": cell_fraction,
        "changed_operator": "direct_smbh_smbh_plummer_only",
        "wave_compact_assignment": "periodic_tsc_unchanged",
    }
    return new_config, new_metadata, radius_pc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-over-cell", type=float,
                        choices=(0.25, 1.0), required=True)
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
        raise ValueError("direct-softening seed source differs from commit")
    reference = args.reference.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace direct-softening seed: {output}")
    parent_hashes = {relative: sha256(reference / relative)
                     for relative in INPUTS}
    config = json.loads((reference / "config.uldm").read_text())
    metadata = json.loads((reference / "fdm_adapter_metadata.json").read_text())
    new_config, new_metadata, radius_pc = derive_config_and_metadata(
        config, metadata, cell_fraction=args.radius_over_cell,
        output_name=output.name, parent=reference, parent_hashes=parent_hashes,
        source_commit=args.source_commit, source_hash=source_hash,
    )
    wave = np.load(reference / INPUTS[0], mmap_mode="r")
    bodies = np.load(reference / INPUTS[1], mmap_mode="r")
    if (
        wave.shape != (256, 256, 256) or wave.dtype != np.complex128
        or bodies.shape != (12,) or bodies.dtype != np.float64
    ):
        raise ValueError("direct-softening parent arrays differ from registered shape")
    if any(sha256(reference / relative) != digest
           for relative, digest in parent_hashes.items()):
        raise ValueError("parent seed changed during softening derivation")
    output.mkdir(parents=True)
    (output / "Outputs/3Wfn").mkdir(parents=True)
    (output / "Outputs/NBody").mkdir(parents=True)
    for relative in (INPUTS[0], INPUTS[1], INPUTS[4]):
        shutil.copyfile(reference / relative, output / relative)
    (output / "config.uldm").write_text(
        json.dumps(new_config, indent=2) + "\n"
    )
    (output / "fdm_adapter_metadata.json").write_text(
        json.dumps(new_metadata, indent=2, sort_keys=True) + "\n"
    )
    derived_hashes = {relative: sha256(output / relative)
                      for relative in INPUTS}
    for relative in (INPUTS[0], INPUTS[1], INPUTS[4]):
        if derived_hashes[relative] != parent_hashes[relative]:
            raise ValueError(f"unchanged seed content differs: {relative}")
    manifest = {
        "status": "source_bound_direct_binary_softening_seed_not_calibration",
        "calibration_eligible": False,
        "parent_reference": str(reference),
        "parent_sha256": parent_hashes,
        "derived_sha256": derived_hashes,
        "source_commit": args.source_commit,
        "source_sha256": source_hash,
        "radius_pc": radius_pc,
        "radius_over_cell": args.radius_over_cell,
        "changed_operator": "direct_smbh_smbh_plummer_only",
        "wave_compact_assignment": "periodic_tsc_unchanged",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (output / "direct_softening_seed_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(output), "radius_pc": radius_pc}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
