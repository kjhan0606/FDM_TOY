#!/usr/bin/env python3
"""Register one q/e resolution pair and its prospective doubled-box run."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.qe_followup_design import build_qe_followup_run_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--coarse-resolution", required=True, type=int)
    parser.add_argument("--fine-resolution", required=True, type=int)
    parser.add_argument("--box-size-pc", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = build_qe_followup_run_manifest(
        case_id=args.case_id,
        physical_cases=args.cases.expanduser().resolve(),
        coarse_resolution=args.coarse_resolution,
        fine_resolution=args.fine_resolution,
        box_size_pc=args.box_size_pc,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=output.parent, delete=False
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
