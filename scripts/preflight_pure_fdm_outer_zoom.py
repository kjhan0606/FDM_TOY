#!/usr/bin/env python3
"""Validate a pure-FDM outer zoom manifest without submitting a run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.pure_fdm_zoom import preflight_pure_fdm_outer_zoom
from fdm_smbh_delay.zoom_calibration import load_zoom_grid


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
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    grid = load_zoom_grid(args.specification)
    preflight = preflight_pure_fdm_outer_zoom(grid)
    destination = args.output.expanduser().resolve()
    _write_json_atomic(destination, preflight.as_dict())
    print(json.dumps(preflight.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
