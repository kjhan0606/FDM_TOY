#!/usr/bin/env python3
"""Read-only q/e fixed-bin doubled-box comparison; never releases a table."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.qe_box_control import (
    assess_qe_box_control,
    prepare_qe_calibration_candidate,
)
from fdm_smbh_delay.subgrid_table_builder import CalibrationSource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--resolution-pair", type=Path, required=True)
    parser.add_argument("--doubled-box", type=Path, required=True)
    parser.add_argument(
        "--candidate-output", type=Path,
        help="atomically create a non-production candidate JSON; never overwrite",
    )
    args = parser.parse_args()
    pair = CalibrationSource(args.profile_id, args.resolution_pair)
    box = CalibrationSource(args.profile_id, args.doubled_box)
    result = (
        prepare_qe_calibration_candidate(pair, box)
        if args.candidate_output is not None
        else assess_qe_box_control(pair, box)
    )
    if args.candidate_output is not None:
        output = args.candidate_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.partial-", dir=output.parent
        ) as staging:
            staged = Path(staging) / output.name
            with staged.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(staged, output)
            directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
