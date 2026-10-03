#!/usr/bin/env python3
"""Bind an execution-attested relaxation extractor result to its sample ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.dual_soliton_relaxation import (
    materialize_dual_soliton_relaxation_executed_diagnostic_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_ledger", type=Path)
    parser.add_argument("extractor_attestation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = materialize_dual_soliton_relaxation_executed_diagnostic_provenance(
        sample_ledger_path=args.sample_ledger,
        extractor_attestation_path=args.extractor_attestation,
        output_path=args.output,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
