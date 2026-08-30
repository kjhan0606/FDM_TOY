#!/usr/bin/env python3
"""Integrate an accepted CDM resolved-rate track into one delay-stage record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.cdm_coalescence import build_cdm_delay_stage_record


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
    parser.add_argument("rate_track", type=Path)
    parser.add_argument("accepted_phase_ensemble", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = build_cdm_delay_stage_record(
        args.rate_track.expanduser().resolve(),
        args.accepted_phase_ensemble.expanduser().resolve(),
    )
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
