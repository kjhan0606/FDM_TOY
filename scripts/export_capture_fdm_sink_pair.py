#!/usr/bin/env python3
"""Export exact dual-SMBH seed rows from one complete capture ledger event."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.capture_fdm_seed import (
    CaptureFDMSeedFrameSpecification,
    derive_dual_smbh_sink_pair_from_capture,
    verify_mass_projection_source,
)
from fdm_smbh_delay.capture_ledger import read_capture_ledger


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
    parser.add_argument("ledger", type=Path)
    parser.add_argument("frame_specification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    specification = CaptureFDMSeedFrameSpecification.from_dict(
        json.loads(args.frame_specification.read_text(encoding="utf-8"))
    )
    projection_source = verify_mass_projection_source(
        specification.mass_projection,
        reference_directory=args.frame_specification.parent,
    )
    matches = [
        event
        for event in read_capture_ledger(args.ledger).events
        if event.event_uid == specification.event_uid
    ]
    if len(matches) != 1:
        raise ValueError("frame specification event_uid must identify exactly one complete ledger event")
    pair = derive_dual_smbh_sink_pair_from_capture(
        matches[0],
        frame=specification.frame,
        assignment=specification.assignment,
        mass_projection=specification.mass_projection,
    )
    record = pair.as_dict()
    record["mass_projection_validation"] = {
        "status": "source_sha256_verified",
        "resolved_source_path": str(projection_source),
    }
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
