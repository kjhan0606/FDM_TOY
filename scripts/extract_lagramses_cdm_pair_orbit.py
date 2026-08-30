#!/usr/bin/env python3
"""Extract a raw relative-orbit series from complete non-compacting CDM outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.lagramses_cdm_orbit import extract_lagramses_cdm_pair_orbit_track


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
    parser.add_argument("--physics-id", required=True)
    parser.add_argument("--capture-event-uid", required=True)
    parser.add_argument("--capture-binding", type=Path, required=True)
    parser.add_argument("--primary-sink-id", type=int, required=True)
    parser.add_argument("--secondary-sink-id", type=int, required=True)
    parser.add_argument("--position-coordinate", choices=("comoving", "physical"), required=True)
    parser.add_argument("--time-coordinate", choices=("proper",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("outputs", nargs="+", type=Path)
    args = parser.parse_args()
    track = extract_lagramses_cdm_pair_orbit_track(
        args.outputs,
        physics_id=args.physics_id,
        capture_event_uid=args.capture_event_uid,
        capture_binding_path=args.capture_binding,
        primary_sink_id=args.primary_sink_id,
        secondary_sink_id=args.secondary_sink_id,
        position_coordinate=args.position_coordinate,
        time_coordinate=args.time_coordinate,
    )
    record = track.as_dict()
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
