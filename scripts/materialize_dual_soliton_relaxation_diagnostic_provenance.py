#!/usr/bin/env python3
"""Bind one dual-soliton diagnostic series to exact sources and extractor bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.dual_soliton_relaxation import (
    materialize_dual_soliton_relaxation_diagnostic_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_ledger", type=Path)
    parser.add_argument("diagnostic_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = materialize_dual_soliton_relaxation_diagnostic_provenance(
        sample_ledger_path=args.sample_ledger,
        diagnostic_manifest_path=args.diagnostic_manifest,
        output_path=args.output,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
