#!/usr/bin/env python3
"""Write a verified, non-submitting CDM non-compacting zoom run contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.cdm_zoom_materialization import (
    materialize_cdm_noncompacting_zoom_run_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--capture-binding", type=Path, required=True)
    parser.add_argument("--capture-event-uid", required=True)
    parser.add_argument("--primary-sink-id", type=int, required=True)
    parser.add_argument("--secondary-sink-id", type=int, required=True)
    parser.add_argument("--run-namelist", type=Path, required=True)
    parser.add_argument("--capture-ledger-file", required=True)
    parser.add_argument("specification", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    record = materialize_cdm_noncompacting_zoom_run_contract(
        specification_path=args.specification,
        case_id=args.case_id,
        capture_binding_path=args.capture_binding,
        capture_event_uid=args.capture_event_uid,
        primary_sink_id=args.primary_sink_id,
        secondary_sink_id=args.secondary_sink_id,
        run_namelist_path=args.run_namelist,
        capture_ledger_file=args.capture_ledger_file,
        output_directory=args.output_directory,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "ready_for_operator_submission" else 2


if __name__ == "__main__":
    raise SystemExit(main())
