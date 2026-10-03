#!/usr/bin/env python3
"""Bind one lagRamses capture-ledger event to a DM run-provenance sidecar."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.capture_ledger import read_capture_ledger
from fdm_smbh_delay.dm_run_provenance import (
    bind_capture_event_to_dark_matter_run,
    read_dark_matter_run_provenance,
)


def _write_json_atomic(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("event_uid")
    parser.add_argument("run_provenance", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    matches = [event for event in read_capture_ledger(args.ledger).events if event.event_uid == args.event_uid]
    if len(matches) != 1:
        raise ValueError("event_uid must identify exactly one complete capture event")
    decision = bind_capture_event_to_dark_matter_run(
        matches[0], read_dark_matter_run_provenance(args.run_provenance)
    )
    record = decision.as_dict()
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if decision.bound else 2


if __name__ == "__main__":
    raise SystemExit(main())
