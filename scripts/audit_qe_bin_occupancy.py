#!/usr/bin/env python3
"""Diagnose resolved-orbit occupancy without changing calibration gates.

This is a resource-design diagnostic, not a source of accepted table rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from fdm_smbh_delay.convergence import (
    _initial_resolved_orbit_indices,
    load_convergence_run,
)

# Support both ``python -m scripts.audit_qe_bin_occupancy`` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.reassess_qe_extension import RUN_INPUTS, _sha256


def pair_occupancy(
    fine: dict,
    coarse: dict,
    *,
    separation_bins: int = 8,
    minimum_orbits_per_bin: int = 8,
) -> dict:
    """Count complete, initially resolved orbits in each common bin."""

    if separation_bins < 1 or minimum_orbits_per_bin < 2:
        raise ValueError("separation-bin and minimum-orbit counts are invalid")
    support = []
    for loaded in (fine, coarse):
        orbit = loaded["orbit_series"]
        if orbit is None:
            raise ValueError("orbit-averaged exchange table is required")
        valid = _initial_resolved_orbit_indices(
            orbit,
            float(loaded["conservation"]["initial_spatially_resolved_duration_myr"]),
        )
        values = np.asarray(orbit["mean_separation_pc"][valid], dtype=float)
        support.append(values)
    result: dict = {
        "resolved_complete_orbits": [int(values.size) for values in support],
        "separation_bins": separation_bins,
        "minimum_complete_orbits_per_run_per_bin": minimum_orbits_per_bin,
        "minimum_total_orbits_for_full_bin_coverage": (
            separation_bins * minimum_orbits_per_bin
        ),
        "purpose": "resource_design_only_no_calibration_admission",
        "bins": [],
    }
    if any(values.size < 2 for values in support):
        result["status"] = "insufficient_initially_resolved_orbits"
        return result
    lower = max(float(np.min(values)) for values in support)
    upper = min(float(np.max(values)) for values in support)
    result["common_minimum_separation_pc"] = lower
    result["common_maximum_separation_pc"] = upper
    if upper <= lower:
        result["status"] = "no_common_resolved_separation"
        return result
    edges = np.linspace(lower, upper, separation_bins + 1)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        counts = []
        for values in support:
            in_bin = values >= left - 1.0e-14
            if index == separation_bins - 1:
                in_bin &= values <= right + 1.0e-14
            else:
                in_bin &= values < right
            counts.append(int(np.count_nonzero(in_bin)))
        result["bins"].append({
            "index": index,
            "lower_separation_pc": float(left),
            "upper_separation_pc": float(right),
            "complete_orbits": counts,
            "passes_orbit_count_only": all(
                count >= minimum_orbits_per_bin for count in counts
            ),
        })
    result["orbit_count_eligible_bins"] = sum(
        bin_row["passes_orbit_count_only"] for bin_row in result["bins"]
    )
    result["status"] = "common_resolved_separation_binned"
    return result


def audit_assessment(assessment_path: Path, *, separation_bins: int = 8) -> dict:
    """Recheck immutable source hashes before describing bin occupancy."""

    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    if (
        assessment.get("status") != "qe_finest_adjacent_pairs_reassessed"
        or assessment.get("case_count") != len(assessment.get("cases", []))
    ):
        raise ValueError("q-e assessment is incomplete")
    cases = []
    for case in assessment["cases"]:
        runs = []
        for role in ("fine", "coarse"):
            run = Path(case[f"{role}_run"])
            if not run.is_dir():
                raise ValueError(f"missing q-e run: {run}")
            expected = case[f"{role}_input_sha256"]
            if set(expected) != set(RUN_INPUTS) or any(
                _sha256(run / name) != expected[name] for name in RUN_INPUTS
            ):
                raise ValueError(f"q-e inputs changed since reassessment: {run}")
            runs.append(load_convergence_run(role, run))
        diagnostic = pair_occupancy(*runs, separation_bins=separation_bins)
        cases.append({
            "case_id": case["case_id"],
            "assessment_status": case["status"],
            "diagnostic": diagnostic,
        })
    return {
        "status": "qe_bin_occupancy_design_diagnostic_only",
        "assessment": str(assessment_path.resolve()),
        "assessment_sha256": _sha256(assessment_path),
        "case_count": len(cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--separation-bins", type=int, default=8)
    args = parser.parse_args()
    result = audit_assessment(
        args.assessment.expanduser().resolve(),
        separation_bins=args.separation_bins,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
