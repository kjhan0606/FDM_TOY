#!/usr/bin/env python3
"""Estimate prerequisites for a new q-e calibration design; launch nothing."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

from fdm_smbh_delay.calibration import estimated_uniform_grid_memory_gib

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.reassess_qe_extension import finest_adjacent_pairs


def plan_resources(
    manifest: Path,
    cases_file: Path,
    *,
    separation_bins: int = 8,
    minimum_orbits_per_bin: int = 8,
    box_factor: int = 2,
    reference_gpu_memory_gib: float = 80.0,
) -> dict:
    """Return necessary, not sufficient, sampling and memory conditions."""

    if (
        separation_bins < 1
        or minimum_orbits_per_bin < 2
        or box_factor < 2
        or reference_gpu_memory_gib <= 0.0
    ):
        raise ValueError("q-e follow-up resource parameters are invalid")
    with cases_file.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        cases = {}
        for row in reader:
            case_id = row.get("case_id")
            if not case_id or case_id in cases:
                raise ValueError("q-e case IDs must be present and unique")
            cases[case_id] = row
    rows = []
    minimum_orbits = separation_bins * minimum_orbits_per_bin
    for case_id, fine, coarse in finest_adjacent_pairs(manifest):
        if case_id not in cases:
            raise ValueError(f"q-e case is absent from physical cases: {case_id}")
        case = cases[case_id]
        period = float(case["kepler_period_myr"])
        old_duration = float(case["target_duration_myr"])
        if (
            not math.isfinite(period)
            or not math.isfinite(old_duration)
            or period <= 0.0
            or old_duration <= 0.0
        ):
            raise ValueError(f"q-e case has invalid duration or period: {case_id}")
        controls = []
        for resolution, run_id in (coarse, fine):
            control_resolution = resolution * box_factor
            memory = estimated_uniform_grid_memory_gib(control_resolution)
            controls.append({
                "original_run_id": run_id,
                "original_resolution": resolution,
                "same_cell_size_control_resolution": control_resolution,
                "uniform_grid_memory_estimate_gib": memory,
                "estimate_exceeds_reference_gpu": (
                    memory > reference_gpu_memory_gib
                ),
            })
        rows.append({
            "case_id": case_id,
            "nominal_kepler_orbits_in_existing_target": old_duration / period,
            "minimum_orbits_for_full_bin_coverage": minimum_orbits,
            "minimum_duration_myr_at_initial_kepler_period": (
                minimum_orbits * period
            ),
            "minimum_duration_over_existing_target": (
                minimum_orbits * period / old_duration
            ),
            "same_cell_size_box_controls": controls,
        })
    return {
        "status": "qe_followup_resource_design_only_no_run_authorized",
        "manifest": str(manifest.resolve()),
        "physical_cases": str(cases_file.resolve()),
        "separation_bins": separation_bins,
        "minimum_orbits_per_bin": minimum_orbits_per_bin,
        "box_factor": box_factor,
        "reference_gpu_memory_gib": reference_gpu_memory_gib,
        "memory_model": "16_double_precision_real_arrays_uniform_grid_estimate",
        "interpretation": (
            "Full-bin orbit count and uniform-grid memory are necessary planning "
            "bounds, not evidence of resolved duration, numerical convergence, "
            "doubled-box agreement, or feasible peak GPU memory."
        ),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--separation-bins", type=int, default=8)
    parser.add_argument("--minimum-orbits-per-bin", type=int, default=8)
    parser.add_argument("--box-factor", type=int, default=2)
    parser.add_argument("--reference-gpu-memory-gib", type=float, default=80.0)
    args = parser.parse_args()
    result = plan_resources(
        args.manifest.expanduser().resolve(),
        args.cases.expanduser().resolve(),
        separation_bins=args.separation_bins,
        minimum_orbits_per_bin=args.minimum_orbits_per_bin,
        box_factor=args.box_factor,
        reference_gpu_memory_gib=args.reference_gpu_memory_gib,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
