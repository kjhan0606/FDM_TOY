#!/usr/bin/env python3
"""Evaluate registered pure-FDM outer zoom records without submitting work."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

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
    evaluation = evaluate_pure_fdm_outer_result_index(grid, result_index).as_dict()
    destination = args.output.expanduser().resolve()
    _write_json_atomic(destination, evaluation)
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
