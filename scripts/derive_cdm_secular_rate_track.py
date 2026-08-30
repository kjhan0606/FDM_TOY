#!/usr/bin/env python3
"""Derive a reproducible secular CDM separation-rate track from raw orbit data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.cdm_coalescence import derive_cdm_secular_rate_track


def _write_json_atomic(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_orbit_track", type=Path)
    parser.add_argument("--stage", choices=("capture_to_hard_binary", "hard_binary_to_gw_regime"), required=True)
    parser.add_argument("--samples-per-block", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = derive_cdm_secular_rate_track(
        args.raw_orbit_track.expanduser().resolve(),
        stage=args.stage,
        samples_per_block=args.samples_per_block,
    )
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
