#!/usr/bin/env python3
"""Release registered, raw-verified q/e rate bins with doubled-box controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from fdm_smbh_delay.subgrid_table_builder import (
    CalibrationSource,
    write_qe_calibration_table,
)


def _control(value: str) -> tuple[CalibrationSource, CalibrationSource]:
    try:
        profile, paths = value.split("=", 1)
        pair, box = paths.split(",", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "control must be PROFILE=PAIR.json,BOX.json"
        ) from error
    if not profile or not pair or not box:
        raise argparse.ArgumentTypeError(
            "control must be PROFILE=PAIR.json,BOX.json"
        )
    return (
        CalibrationSource(profile, Path(pair)),
        CalibrationSource(profile, Path(box)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="append", required=True, type=_control)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    write_qe_calibration_table(arguments.control, output=arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
