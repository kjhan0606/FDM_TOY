#!/usr/bin/env python3
"""Assemble seed YAML and capture-pair evidence without submitting a run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import yaml

from fdm_smbh_delay.capture_fdm_seed import (
    CaptureFDMSeedFrameSpecification,
    derive_dual_smbh_sink_pair_from_capture,
    materialize_capture_derived_sink_pair_record,
)
from fdm_smbh_delay.capture_seed_assembly import (
    assemble_capture_derived_pure_fdm_seed,
    capture_derived_seed_mapping,
    capture_soliton_configuration_from_mapping,
)
from fdm_smbh_delay.capture_ledger import read_capture_ledger


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("frame_specification", type=Path)
    parser.add_argument("soliton_configuration", type=Path)
    parser.add_argument("seed_output", type=Path)
    parser.add_argument("capture_pair_output", type=Path)
    args = parser.parse_args()
    specification = CaptureFDMSeedFrameSpecification.from_dict(
        json.loads(args.frame_specification.read_text(encoding="utf-8"))
    )
    configuration = capture_soliton_configuration_from_mapping(
        yaml.safe_load(args.soliton_configuration.read_text(encoding="utf-8"))
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
    seed = assemble_capture_derived_pure_fdm_seed(pair, configuration)
    pair_record = materialize_capture_derived_sink_pair_record(
        pair,
        frame_specification_path=args.frame_specification,
    )
    _write_atomic(
        args.seed_output.expanduser().resolve(),
        yaml.safe_dump(capture_derived_seed_mapping(seed), sort_keys=True),
    )
    _write_atomic(
        args.capture_pair_output.expanduser().resolve(),
        json.dumps(pair_record, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "status": "capture_derived_pure_fdm_seed_assembled",
                "seed_output": str(args.seed_output.expanduser().resolve()),
                "capture_pair_output": str(args.capture_pair_output.expanduser().resolve()),
                "capture_event_uid": pair.event_uid,
                "seed_case_id": seed.case_id,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
