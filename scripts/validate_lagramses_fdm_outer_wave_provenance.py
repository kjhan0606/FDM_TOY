#!/usr/bin/env python3
"""Validate one raw lagRamses FDM outer-wave provenance record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.lagramses_fdm_provenance import (
    read_lagramses_fdm_outer_wave_provenance,
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
    parser.add_argument("provenance", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-current-stencil-fraction", type=float, default=0.95)
    args = parser.parse_args()
    provenance = read_lagramses_fdm_outer_wave_provenance(args.provenance)
    record = provenance.as_dict()
    record["decision"] = provenance.decision(
        minimum_current_stencil_fraction=args.minimum_current_stencil_fraction
    )
    destination = args.output.expanduser().resolve()
    _write_json_atomic(destination, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
