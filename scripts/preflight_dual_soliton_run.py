#!/usr/bin/env python3
"""Check a dual-soliton lagRamses run input without submitting it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.dual_soliton_preflight import (
    preflight_pure_fdm_dual_soliton_run,
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
    parser.add_argument("seed_manifest", type=Path)
    parser.add_argument("run_namelist", type=Path)
    parser.add_argument("run_ic_sink", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    decision = preflight_pure_fdm_dual_soliton_run(
        seed_manifest_path=args.seed_manifest,
        run_namelist_path=args.run_namelist,
        run_ic_sink_path=args.run_ic_sink,
    )
    record = decision.as_dict()
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
