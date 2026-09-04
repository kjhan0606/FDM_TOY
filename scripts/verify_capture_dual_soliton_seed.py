#!/usr/bin/env python3
"""Verify that a materialized dual-soliton seed preserves capture sink rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.capture_seed_binding import (
    assess_capture_seed_materialization_binding,
)


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
    parser.add_argument("capture_sink_pair", type=Path)
    parser.add_argument("seed_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    decision = assess_capture_seed_materialization_binding(
        capture_sink_pair_path=args.capture_sink_pair,
        seed_manifest_path=args.seed_manifest,
    )
    record = decision.as_dict()
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if decision.verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
