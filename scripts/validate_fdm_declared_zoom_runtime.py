#!/usr/bin/env python3
"""Verify listed FDM normal outputs against one declared seed/zoom run input."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.fdm_zoom_runtime_identity import (
    assess_fdm_declared_zoom_runtime_outputs,
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
    parser.add_argument("declared_run_input_binding", type=Path)
    parser.add_argument("normal_outputs", nargs="+", type=Path)
    args = parser.parse_args()
    decision = assess_fdm_declared_zoom_runtime_outputs(
        args.declared_run_input_binding, args.normal_outputs
    )
    record = decision.as_dict()
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if decision.verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
