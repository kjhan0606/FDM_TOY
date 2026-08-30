#!/usr/bin/env python3
"""Verify completed CDM zoom outputs against a materialized run contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.cdm_zoom_runtime_identity import (
    assess_cdm_noncompacting_zoom_runtime_identity,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("contract", type=Path)
    parser.add_argument("outputs", nargs="+", type=Path)
    args = parser.parse_args()
    decision = assess_cdm_noncompacting_zoom_runtime_identity(args.contract, args.outputs)
    record = decision.as_dict()
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if decision.verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
