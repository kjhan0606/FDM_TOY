#!/usr/bin/env python3
"""Compare coupled-kick attribution under two spectral reduction orders."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


SOURCE = "scripts/compare_coupled_kick_precision.py"
LEVELS = ("f100", "f0125")
COMPONENTS = (
    "wave_compact_plus_body_attribution_code",
    "wave_compact_plus_body_alternate_order_attribution_code",
    "wave_self_attribution_code",
    "wave_self_alternate_order_attribution_code",
    "wave_drift_attribution_code",
)
STATUS = "coupled_factorized_kick_diagnostic_not_a_calibration_release"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(payload: dict, key: str) -> np.ndarray:
    value = np.asarray(payload[key], dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid momentum vector: {key}")
    return value


def compare_level(original: dict, reordered: dict, label: str) -> dict:
    if (
        original.get("status") != STATUS
        or reordered.get("status") != STATUS
        or original.get("calibration_eligible") is not False
        or reordered.get("calibration_eligible") is not False
        or original.get("momentum_estimator", "marginal") != "marginal"
        or reordered.get("momentum_estimator") != "reordered"
        or original.get("run") != reordered.get("run")
        or not str(original.get("run", "")).endswith("/" + label)
        or original.get("wave_steps") != reordered.get("wave_steps")
        or original.get("time_step_factor")
        != reordered.get("time_step_factor")
        or original.get("initial_wave_sha256")
        != reordered.get("initial_wave_sha256")
        or original.get("initial_body_sha256")
        != reordered.get("initial_body_sha256")
        or original.get("final_wave_sha256")
        != reordered.get("final_wave_sha256")
        or original.get("final_body_sha256")
        != reordered.get("final_body_sha256")
    ):
        raise ValueError(f"incompatible reduction-order runs: {label}")
    old_sources = original["source_sha256"]
    new_sources = reordered["source_sha256"]
    if (
        old_sources.keys() != new_sources.keys()
        or any(
            old_sources[key] != new_sources[key]
            for key in old_sources if key != "scripts/probe_coupled_kick_attribution.py"
        )
    ):
        raise ValueError(f"numerical operator sources differ: {label}")
    residual = vector(original, "final_total_momentum_residual_code")
    new_residual = vector(reordered, "final_total_momentum_residual_code")
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 0.0 or not np.array_equal(residual, new_residual):
        raise ValueError(f"endpoint residual differs between estimators: {label}")
    differences = {}
    for key in COMPONENTS:
        value = float(np.linalg.norm(vector(original, key)
                                     - vector(reordered, key)))
        differences[key] = {
            "vector_difference_code": value,
            "difference_over_endpoint_residual": value / residual_norm,
        }
    pair_keys = COMPONENTS[:2]
    if max(differences[key]["difference_over_endpoint_residual"]
           for key in pair_keys) >= 0.01:
        raise ValueError(f"interaction attribution is reduction-order sensitive: {label}")
    return {
        "label": label,
        "wave_steps": original["wave_steps"],
        "endpoint_residual_norm_code": residual_norm,
        "original_source_commit": original["source_commit"],
        "reordered_source_commit": reordered["source_commit"],
        "original_wave_relative_replay_difference": original[
            "wave_relative_replay_difference"
        ],
        "reordered_wave_relative_replay_difference": reordered[
            "wave_relative_replay_difference"
        ],
        "components": differences,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("original_root", type=Path)
    parser.add_argument("reordered_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("source commit must be a full Git revision")
    project = Path(__file__).resolve().parents[1]
    source_hash = sha256(project / SOURCE)
    committed = subprocess.run(
        ["git", "show", f"{args.source_commit}:{SOURCE}"],
        cwd=project, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if hashlib.sha256(committed.stdout).hexdigest() != source_hash:
        raise ValueError("comparison source differs from registered commit")
    original_root = args.original_root.expanduser().resolve()
    reordered_root = args.reordered_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace precision comparison: {output}")
    input_hashes = {}
    levels = []
    for label in LEVELS:
        original_path = original_root / f"{label}.json"
        reordered_path = reordered_root / f"{label}.json"
        levels.append(compare_level(
            json.loads(original_path.read_text()),
            json.loads(reordered_path.read_text()), label,
        ))
        input_hashes[label] = {
            "marginal": sha256(original_path),
            "reordered": sha256(reordered_path),
        }
    payload = {
        "status": "short_prefix_reduction_order_check_not_a_calibration_release",
        "calibration_eligible": False,
        "source_commit": args.source_commit,
        "source_sha256": source_hash,
        "input_sha256": input_hashes,
        "levels": levels,
        "interpretation": (
            "The interaction-pair attribution survives a changed spectral "
            "summation order on f100 and f0125. The tiny self and drift "
            "terms are not precision-resolved; this does not validate "
            "the FFT, spatial convergence, full orbit, or calibration."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(temporary, output)
    temporary.unlink()
    print(json.dumps({"status": payload["status"],
                      "levels": [row["label"] for row in levels]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
