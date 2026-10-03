#!/usr/bin/env python3
"""Assess one completed lagRamses normal-output physics inventory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.resolved_physics_inventory import (
    assess_lagramses_resolved_physics_inventory,
    read_lagramses_resolved_physics_inventory,
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
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stars-required", action="store_true")
    parser.add_argument("--gas-required", action="store_true")
    args = parser.parse_args()
    assessment = assess_lagramses_resolved_physics_inventory(
        read_lagramses_resolved_physics_inventory(args.inventory.expanduser().resolve()),
        stars_required=args.stars_required,
        gas_required=args.gas_required,
    )
    record = assessment.as_dict()
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if assessment.ready_for_registered_analysis else 2


if __name__ == "__main__":
    raise SystemExit(main())
