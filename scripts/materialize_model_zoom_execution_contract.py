#!/usr/bin/env python3
"""Materialize a non-submitting common CDM/SIDM/FDM zoom identity contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.model_zoom_materialization import (
    materialize_model_zoom_execution_contract,
)


def _shared_inputs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, reference = value.partition("=")
        if not separator or not name.strip() or not reference.strip() or name.strip() in result:
            raise ValueError("--shared-input must be a unique ROLE=PATH")
        result[name.strip()] = Path(reference.strip())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--capture-ledger", type=Path, required=True)
    parser.add_argument("--capture-event-uid", required=True)
    parser.add_argument("--shared-input", action="append", default=[], metavar="ROLE=PATH")
    parser.add_argument("specification", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    record = materialize_model_zoom_execution_contract(
        specification_path=args.specification,
        case_id=args.case_id,
        capture_ledger_path=args.capture_ledger,
        capture_event_uid=args.capture_event_uid,
        shared_input_paths=_shared_inputs(args.shared_input),
        output_directory=args.output_directory,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
