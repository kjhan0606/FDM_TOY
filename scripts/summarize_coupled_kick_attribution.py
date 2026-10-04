#!/usr/bin/env python3
"""Check four short-prefix kick attributions without releasing calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np


LEVELS = (("f100", 1.0, 10), ("f050", 0.5, 20),
          ("f025", 0.25, 40), ("f0125", 0.125, 80))
SOURCE = "scripts/summarize_coupled_kick_attribution.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(record: dict, key: str) -> np.ndarray:
    value = np.asarray(record[key], dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid momentum vector: {key}")
    return value


def assess_level(payload: dict, *, label: str, factor: float,
                 steps: int) -> dict:
    if (
        payload.get("status")
        != "coupled_factorized_kick_diagnostic_not_a_calibration_release"
        or payload.get("calibration_eligible") is not False
        or payload.get("time_step_factor") != factor
        or payload.get("wave_steps") != steps
        or len(payload.get("per_step", [])) != steps
        or not str(payload.get("run", "")).endswith("/" + label)
    ):
        raise ValueError(f"invalid coupled attribution provenance: {label}")
    rows = payload["per_step"]
    pair = np.zeros(3)
    pair_alternate = np.zeros(3)
    self_kick = np.zeros(3)
    self_alternate = np.zeros(3)
    drift = np.zeros(3)
    interaction_half_norms = []
    joint_half_norms = []
    for step, row in enumerate(rows, start=1):
        if row.get("step") != step:
            raise ValueError(f"nonsequential attribution step: {label}")
        for suffix in ("first", "second"):
            interaction = (
                vector(row, f"wave_compact_{suffix}")
                + vector(row, f"body_{suffix}")
            )
            interaction_alternate = (
                vector(row, f"wave_compact_{suffix}_alternate")
                + vector(row, f"body_{suffix}")
            )
            self_delta = vector(row, f"wave_self_{suffix}")
            self_alt_delta = vector(row, f"wave_self_{suffix}_alternate")
            joint = vector(row, f"{suffix}_half_kick_defect")
            if not np.allclose(
                joint, interaction + self_delta, rtol=0, atol=1e-11
            ):
                raise ValueError(f"joint half-kick closure failed: {label}")
            pair += interaction
            pair_alternate += interaction_alternate
            self_kick += self_delta
            self_alternate += self_alt_delta
            interaction_half_norms.append(float(np.linalg.norm(interaction)))
            joint_half_norms.append(float(np.linalg.norm(joint)))
        drift += vector(row, "wave_drift")
    residual = vector(payload, "final_total_momentum_residual_code")
    alternate_order_closure_gap = float(np.linalg.norm(
        residual - (pair_alternate + self_alternate + drift)
    ))
    if not (
        np.allclose(residual, pair + self_kick + drift, rtol=0, atol=1e-11)
        and alternate_order_closure_gap <= 1e-7
        and np.allclose(
            pair, vector(payload, "wave_compact_plus_body_attribution_code"),
            rtol=0, atol=1e-11,
        )
        and np.allclose(
            pair_alternate,
            vector(payload, "wave_compact_plus_body_alternate_order_attribution_code"),
            rtol=0, atol=1e-11,
        )
    ):
        raise ValueError(f"full attribution closure failed: {label}")
    norm = float(np.linalg.norm(residual))
    if norm <= 0.0:
        raise ValueError(f"zero endpoint residual: {label}")
    pair_gap = float(np.linalg.norm(pair - residual)) / norm
    alternate_gap = float(np.linalg.norm(pair_alternate - residual)) / norm
    if max(pair_gap, alternate_gap) >= 0.01:
        raise ValueError(f"interaction attribution does not dominate: {label}")
    return {
        "label": label,
        "wave_steps": steps,
        "time_step_factor": factor,
        "endpoint_residual_norm_code": norm,
        "interaction_pair_norm_code": float(np.linalg.norm(pair)),
        "alternate_interaction_pair_norm_code": float(
            np.linalg.norm(pair_alternate)
        ),
        "interaction_pair_vector_gap_over_residual": pair_gap,
        "alternate_interaction_pair_vector_gap_over_residual": alternate_gap,
        "alternate_phase_order_closure_gap_code": (
            alternate_order_closure_gap
        ),
        "self_kick_norm_code": float(np.linalg.norm(self_kick)),
        "alternate_self_kick_norm_code": float(np.linalg.norm(self_alternate)),
        "kinetic_drift_norm_code": float(np.linalg.norm(drift)),
        "maximum_interaction_only_half_kick_defect_code": max(
            interaction_half_norms
        ),
        "maximum_joint_half_kick_change_code": max(joint_half_norms),
        "cpu_gpu_endpoint_gap_code": float(
            payload["cpu_gpu_endpoint_momentum_gap_code"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
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
        raise ValueError("summary source differs from the registered commit")
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite attribution summary: {output}")
    results = []
    input_hashes = {}
    initial_hashes = set()
    numerical_source_hashes = set()
    for label, factor, steps in LEVELS:
        path = root / f"{label}.json"
        payload = json.loads(path.read_text())
        results.append(assess_level(
            payload, label=label, factor=factor, steps=steps,
        ))
        input_hashes[label] = sha256(path)
        initial_hashes.add((payload["initial_wave_sha256"],
                            payload["initial_body_sha256"]))
        numerical_source_hashes.add(tuple(sorted(
            payload["source_sha256"].items()
        )))
    if len(initial_hashes) != 1 or len(numerical_source_hashes) != 1:
        raise ValueError("refinement levels do not share seed and numerical source")
    norms = [row["endpoint_residual_norm_code"] for row in results]
    changes = [norms[index + 1] - norms[index] for index in range(3)]
    if not all(change > 0.0 for change in changes):
        raise ValueError("endpoint refinement is not monotonic")
    payload = {
        "status": "short_prefix_interaction_attribution_diagnostic",
        "calibration_eligible": False,
        "attribution_precision_resolved": False,
        "source_commit": args.source_commit,
        "source_sha256": source_hash,
        "input_sha256": input_hashes,
        "initial_wave_sha256": next(iter(initial_hashes))[0],
        "initial_body_sha256": next(iter(initial_hashes))[1],
        "levels": results,
        "endpoint_residual_norm_refinement_ratios": [
            changes[0] / changes[1], changes[1] / changes[2],
        ],
        "interpretation": (
            "The interaction kick pair dominates the measured endpoint "
            "spectral-momentum residual in both phase orderings on this "
            "short fixed-grid prefix. Self and kinetic-drift terms are "
            "below the single-evaluation summation heuristic, so their "
            "individual values are unresolved. Neither physical cause "
            "nor full-orbit conservation nor calibration is established."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.link_to(output)
    temporary.unlink()
    print(json.dumps({"status": payload["status"],
                      "refinement_ratios": payload[
                          "endpoint_residual_norm_refinement_ratios"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
