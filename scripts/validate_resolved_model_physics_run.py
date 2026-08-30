#!/usr/bin/env python3
"""Validate one completed CDM/SIDM/FDM resolved-physics result without submission."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.model_specific_physics import read_resolved_model_physics_result
from fdm_smbh_delay.zoom_calibration import load_zoom_grid


def _write_json_atomic(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _case_for_result(grid, source: Path):
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read model-specific result: {error}") from error
    case_id = record.get("case_id") if isinstance(record, dict) else None
    matches = [case for case in grid.cases if case.case_id == case_id]
    if len(matches) != 1:
        raise ValueError("result case_id is absent from the supplied zoom manifest")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    grid = load_zoom_grid(args.specification)
    source = args.result.expanduser().resolve()
    result = read_resolved_model_physics_result(
        source,
        case=_case_for_result(grid, source),
        zoom_manifest_sha256=grid.manifest_sha256,
    )
    record = result.as_dict()
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
