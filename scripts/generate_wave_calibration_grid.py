#!/usr/bin/env python3
"""Generate physical cases and resolution runs for live-wave calibration."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml

from fdm_smbh_delay.calibration import (
    literature_anchor_cases,
    run_specifications,
    structured_parameter_cases,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty calibration table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", type=Path, default=Path("configs/wave_calibration_grid.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/wave_calibration_grid")
    )
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    axes = raw["axes"]
    fiducial_raw = raw["fiducial"]
    fiducial = (
        fiducial_raw["mass_ratio_q"],
        fiducial_raw["eccentricity"],
        fiducial_raw["binary_to_soliton_mass"],
        fiducial_raw["semi_major_axis_over_core_radius"],
    )
    cases = literature_anchor_cases()
    cases.extend(
        structured_parameter_cases(
            q_values=axes["mass_ratio_q"],
            eccentricities=axes["eccentricity"],
            binary_mass_fractions=axes["binary_to_soliton_mass"],
            semi_major_axis_over_core=axes["semi_major_axis_over_core_radius"],
            fiducial=fiducial,
        )
    )
    numerics = raw["numerics"]
    resolutions = {
        int(tier): [int(value) for value in values]
        for tier, values in numerics["resolutions_by_tier"].items()
    }
    runs = run_specifications(
        cases,
        box_over_core_radius=float(numerics["box_over_core_radius"]),
        resolutions_by_tier=resolutions,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    case_rows = [case.as_dict() for case in cases]
    run_rows = [run.as_dict() for run in runs]
    _write_csv(args.output / "physical_cases.csv", case_rows)
    _write_csv(args.output / "run_manifest.csv", run_rows)
    summary = {
        "physical_case_count": len(cases),
        "run_count": len(runs),
        "cases_by_tier": dict(sorted(Counter(case.tier for case in cases).items())),
        "runs_by_tier": dict(sorted(Counter(run.tier for run in runs).items())),
        "analytic_fdm_drag": False,
        "solver_status": "public_pyul_adapter_available_full_grid_pending",
        "energy_accounting": raw["energy_accounting"],
        "required_measurements": raw["measurements"],
        "configuration": str(args.config),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
