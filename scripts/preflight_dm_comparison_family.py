#!/usr/bin/env python3
"""Write a non-submitting CDM/SIDM/FDM comparison-family preflight record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.dm_comparison import (
    preflight_dm_comparison_family,
    read_dm_comparison_family_manifest,
)


def _write_json_atomic(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assessment = preflight_dm_comparison_family(read_dm_comparison_family_manifest(args.manifest))
    record = assessment.as_dict()
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if assessment.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
