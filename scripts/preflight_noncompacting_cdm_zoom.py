#!/usr/bin/env python3
"""Check whether one completed lagRamses output is a non-compacting CDM zoom."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.cdm_zoom import assess_noncompacting_cdm_zoom_run
from fdm_smbh_delay.dm_run_provenance import read_dark_matter_run_provenance


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
    args = parser.parse_args()
    assessment = assess_noncompacting_cdm_zoom_run(
        read_dark_matter_run_provenance(args.provenance.expanduser().resolve())
    )
    record = assessment.as_dict()
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if assessment.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
