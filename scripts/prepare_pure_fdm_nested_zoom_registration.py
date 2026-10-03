#!/usr/bin/env python3
"""Prepare checkpoint-bound pure-FDM nested-zoom registrations; never submit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.pure_fdm_nested_registration import (
    build_pure_fdm_nested_registration_manifest,
)
from fdm_smbh_delay.pure_fdm_outer_evaluation import (
    evaluate_pure_fdm_outer_result_index,
    load_pure_fdm_outer_result_index,
)
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
    parser.add_argument("result_index", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    grid = load_zoom_grid(args.specification)
    result_index = load_pure_fdm_outer_result_index(args.result_index, grid=grid)
    evaluation = evaluate_pure_fdm_outer_result_index(grid, result_index)
    registration = build_pure_fdm_nested_registration_manifest(evaluation).as_dict()
    destination = args.output.expanduser().resolve()
    _write_json_atomic(destination, registration)
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
