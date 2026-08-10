#!/usr/bin/env python3
"""Compare the overlapping interval of two live-wave calculations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


_ENERGY_FIELDS = (
    "binary_orbital_energy",
    "bh_com_kinetic_energy",
    "wave_intrinsic_energy",
    "wave_bh_interaction_grid",
)


def _load(run: Path) -> np.ndarray:
    return np.genfromtxt(
        run / "conservation_timeseries.csv", delimiter=",", names=True, ndmin=1
    )


def _comparison_setup(run: Path) -> dict:
    config = json.loads((run / "config.uldm").read_text(encoding="utf-8"))
    metadata = json.loads(
        (run / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
    )
    return {
        "pyul_revision": metadata["pyul_revision"],
        "particle_mass_ev": metadata["particle_mass_ev"],
        "spatial_resolution": config["Spatial Resolution"],
        "temporal_step_factor": config["Temporal Step Factor"],
        "rk_steps": config["RK Steps"],
        "simulation_box": config["Simulation Box"],
        "solitons": config["ULDM Solitons"],
        "matter_particles": config["Matter Particles"],
        "central_mass": config["Central Mass"],
        "uniform_field": config["Uniform Field Override"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference_run = args.reference.expanduser().resolve()
    comparison_run = args.comparison.expanduser().resolve()
    reference_setup = _comparison_setup(reference_run)
    comparison_setup = _comparison_setup(comparison_run)
    if reference_setup != comparison_setup:
        differing = sorted(
            key
            for key in reference_setup
            if reference_setup[key] != comparison_setup[key]
        )
        raise ValueError(
            "overlap comparison requires identical physical and numerical "
            f"setups; differing fields: {', '.join(differing)}"
        )
    reference = _load(reference_run)
    comparison = _load(comparison_run)
    overlap_end = min(
        float(reference["time_myr"][-1]), float(comparison["time_myr"][-1])
    )
    selected = reference["time_myr"] <= overlap_end
    if np.count_nonzero(selected) < 2:
        raise ValueError("the runs do not share two reference samples")
    time = reference["time_myr"][selected]
    exchange_scale = max(
        max(
            float(np.max(np.abs(reference[field][selected] - reference[field][0])))
            for field in _ENERGY_FIELDS
        ),
        np.finfo(float).tiny,
    )
    energy_mismatches: dict[str, dict[str, float]] = {}
    for field in (*_ENERGY_FIELDS, "combined_energy"):
        interpolated = np.interp(
            time, comparison["time_myr"], comparison[field]
        )
        mismatch = (
            reference[field][selected]
            - reference[field][0]
            - (interpolated - interpolated[0])
        )
        energy_mismatches[field] = {
            "maximum_over_reference_exchange": float(
                np.max(np.abs(mismatch)) / exchange_scale
            ),
            "final_over_reference_exchange": float(
                abs(mismatch[-1]) / exchange_scale
            ),
        }
    interpolated_separation = np.interp(
        time, comparison["time_myr"], comparison["separation_pc"]
    )
    separation_mismatch = (
        reference["separation_pc"][selected] - interpolated_separation
    )
    initial_separation = float(reference["separation_pc"][0])
    summary = {
        "status": "overlap_compared",
        "reference": str(reference_run),
        "comparison": str(comparison_run),
        "overlap_end_myr": overlap_end,
        "reference_samples": int(np.count_nonzero(selected)),
        "comparison_samples_in_overlap": int(
            np.count_nonzero(comparison["time_myr"] <= overlap_end)
        ),
        "comparison_to_reference_cadence_ratio": float(
            np.median(np.diff(comparison["time_myr"]))
            / np.median(np.diff(reference["time_myr"][selected]))
        ),
        "maximum_separation_mismatch_over_initial": float(
            np.max(np.abs(separation_mismatch)) / initial_separation
        ),
        "final_separation_mismatch_over_initial": float(
            abs(separation_mismatch[-1]) / initial_separation
        ),
        "energy_change_mismatch": energy_mismatches,
        "interpretation": (
            "differences include interpolation across the comparison cadence; "
            "this overlap test does not replace spatial or temporal convergence"
        ),
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
