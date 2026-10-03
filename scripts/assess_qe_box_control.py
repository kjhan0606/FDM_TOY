#!/usr/bin/env python3
"""Read-only q/e fixed-bin doubled-box comparison; never releases a table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.qe_box_control import assess_qe_box_control
from fdm_smbh_delay.subgrid_table_builder import CalibrationSource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--resolution-pair", type=Path, required=True)
    parser.add_argument("--doubled-box", type=Path, required=True)
    args = parser.parse_args()
    result = assess_qe_box_control(
        CalibrationSource(args.profile_id, args.resolution_pair),
        CalibrationSource(args.profile_id, args.doubled_box),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
