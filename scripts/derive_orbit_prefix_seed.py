#!/usr/bin/env python3
"""Copy one q/e initial state into a source-bound, resumable orbit diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from fdm_smbh_delay.pyul import pyul_unit_system


SOURCE = "scripts/derive_orbit_prefix_seed.py"
INPUTS = (
    "Outputs/3Wfn/P3D_#000.npy",
    "Outputs/NBody/NTM_#000.npy",
    "config.uldm",
    "fdm_adapter_metadata.json",
    "reproducibility.uldm",
)
PLANNED_DURATION_MYR = 0.0036
PLANNED_SAVES = 18


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_metadata(
    metadata: dict, config: dict, *, parent: Path,
    parent_sha256: dict[str, str], output_name: str,
    source_commit: str, source_sha256: str,
) -> dict:
    if (
        metadata.get("case_id") != "qe_q030_e030_a020"
        or metadata.get("resolution") != 256
        or metadata.get("analytic_fdm_drag") is not False
        or not isinstance(metadata.get("qe_design_binding"), dict)
        or metadata["qe_design_binding"].get("status")
        != "qe_prospective_design_bound_not_a_calibration_release"
        or config.get("Spatial Resolution") != 256
        or config["Matter Particles"].get("Position Units") != "pc"
        or not np.isclose(
            config["Matter Particles"]["Plummer Radius"],
            metadata.get("plummer_radius_pc", np.nan), rtol=0, atol=1e-12,
        )
    ):
        raise ValueError("orbit diagnostic requires registered n256 q/e/a parent")
    units = pyul_unit_system(metadata)
    masses_code = np.asarray([
        row[0] for row in config["Matter Particles"]["Condition"]
    ], dtype=float) / units.mass_msun
    if masses_code.shape != (2,) or np.any(masses_code <= 0.0):
        raise ValueError("orbit diagnostic requires two positive SMBH masses")
    a_code = float(metadata["semi_major_axis_pc"]) / units.length_pc
    period_myr = float(
        2.0 * np.pi * np.sqrt(a_code**3 / masses_code.sum()) * units.time_myr
    )
    if not np.isfinite(period_myr) or not 1.0 < PLANNED_DURATION_MYR / period_myr < 1.5:
        raise ValueError("orbit prefix does not bracket one initial Kepler period")
    derived = json.loads(json.dumps(metadata))
    old_design = derived.pop("qe_design_binding")
    derived["run_id"] = output_name
    derived["orbit_prefix_binding"] = {
        "status": "source_bound_resumable_one_orbit_diagnostic_not_calibration",
        "calibration_eligible": False,
        "parent_reference": str(parent),
        "parent_sha256": parent_sha256,
        "parent_qe_design_binding": old_design,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "planned_duration_myr": PLANNED_DURATION_MYR,
        "planned_saves": PLANNED_SAVES,
        "initial_kepler_period_myr": period_myr,
        "initial_periods_planned": PLANNED_DURATION_MYR / period_myr,
        "wave_smbh_coupling": "periodic_tsc_strang_momentum",
        "analytic_fdm_drag": False,
        "reason": "registered 0.1 Myr q/e calibration design is distinct from this short diagnostic",
    }
    return derived


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
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
        raise ValueError("orbit seed source differs from committed revision")
    parent = args.reference.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace orbit seed: {output}")
    parent_hashes = {relative: sha256(parent / relative) for relative in INPUTS}
    metadata = json.loads((parent / "fdm_adapter_metadata.json").read_text())
    config = json.loads((parent / "config.uldm").read_text())
    derived = derive_metadata(
        metadata, config, parent=parent, parent_sha256=parent_hashes,
        output_name=output.name, source_commit=args.source_commit,
        source_sha256=source_hash,
    )
    wave = np.load(parent / INPUTS[0], mmap_mode="r")
    body = np.load(parent / INPUTS[1], mmap_mode="r")
    if (
        wave.shape != (256, 256, 256) or wave.dtype != np.complex128
        or body.shape != (12,) or body.dtype != np.float64
    ):
        raise ValueError("orbit seed parent arrays have wrong shape or precision")
    if any(sha256(parent / relative) != digest
           for relative, digest in parent_hashes.items()):
        raise ValueError("orbit parent changed during derivation")
    output.mkdir(parents=True)
    (output / "Outputs/3Wfn").mkdir(parents=True)
    (output / "Outputs/NBody").mkdir(parents=True)
    for relative in (INPUTS[0], INPUTS[1], INPUTS[2], INPUTS[4]):
        shutil.copyfile(parent / relative, output / relative)
    (output / "fdm_adapter_metadata.json").write_text(
        json.dumps(derived, indent=2, sort_keys=True) + "\n"
    )
    derived_hashes = {relative: sha256(output / relative) for relative in INPUTS}
    if any(derived_hashes[relative] != parent_hashes[relative]
           for relative in (INPUTS[0], INPUTS[1], INPUTS[2], INPUTS[4])):
        raise ValueError("orbit seed changes copied physical input")
    manifest = {
        "status": "source_bound_resumable_one_orbit_diagnostic_not_calibration",
        "calibration_eligible": False,
        "parent_reference": str(parent),
        "parent_sha256": parent_hashes,
        "derived_sha256": derived_hashes,
        "source_commit": args.source_commit,
        "source_sha256": source_hash,
        "planned_duration_myr": PLANNED_DURATION_MYR,
        "planned_saves": PLANNED_SAVES,
        "initial_kepler_period_myr": derived["orbit_prefix_binding"][
            "initial_kepler_period_myr"
        ],
    }
    (output / "orbit_prefix_seed_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "output": str(output),
        "initial_kepler_period_myr": manifest["initial_kepler_period_myr"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
