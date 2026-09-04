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
from fdm_smbh_delay.lagramses_writer_audit import audit_lagramses_writer_force_accounting


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
    parser.add_argument(
        "--writer-source",
        type=Path,
        help=(
            "optional lagRamses output writer source to scan for the required "
            "CDM/SIDM force-accounting tokens"
        ),
    )
    args = parser.parse_args()
    assessment = preflight_dm_comparison_family(read_dm_comparison_family_manifest(args.manifest))
    record = assessment.as_dict()
    writer_audit = None
    if args.writer_source is not None:
        writer_audit = audit_lagramses_writer_force_accounting(args.writer_source)
        record["writer_source_audit"] = writer_audit.as_dict()
        if not writer_audit.tokens_present:
            record["reasons"] = list(record["reasons"]) + [
                "lagRamses writer source is not ready for CDM/SIDM force-accounting integration"
            ]
            record["status"] = "comparison_family_preflight_not_ready"
    _write_json_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if assessment.ready and (writer_audit is None or writer_audit.tokens_present) else 2


if __name__ == "__main__":
    raise SystemExit(main())
