#!/usr/bin/env python3
"""Build an accepted FDM subgrid table from resolution-pair summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from fdm_smbh_delay.subgrid_table_builder import (
    CalibrationSource,
    write_calibration_table,
)


def _source(value: str) -> CalibrationSource:
    try:
        profile_id, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("source must be PROFILE=JSON") from error
    if not profile_id or not path:
        raise argparse.ArgumentTypeError("source must be PROFILE=JSON")
    return CalibrationSource(profile_id=profile_id, convergence_summary=Path(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, type=_source)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--maximum-spatial-systematic-fraction", type=float, default=0.20
    )
    parser.add_argument(
        "--maximum-energy-error-over-transfer", type=float, default=0.01
    )
    parser.add_argument("--minimum-complete-orbits-per-bin", type=int, default=8)
    parser.add_argument("--minimum-core-radius-cells", type=float, default=2.0)
    arguments = parser.parse_args()
    write_calibration_table(
        arguments.source,
        output=arguments.output,
        maximum_spatial_systematic_fraction=(
            arguments.maximum_spatial_systematic_fraction
        ),
        maximum_energy_error_over_transfer=(
            arguments.maximum_energy_error_over_transfer
        ),
        minimum_complete_orbits_per_bin=(
            arguments.minimum_complete_orbits_per_bin
        ),
        minimum_core_radius_cells=arguments.minimum_core_radius_cells,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
