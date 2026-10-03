#!/usr/bin/env python3
"""Bind every manual dual-soliton relaxation sample to its source artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.dual_soliton_relaxation import (
    materialize_dual_soliton_relaxation_sample_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_identity", type=Path)
    parser.add_argument("--runtime-output-identity", type=Path, required=True)
    parser.add_argument("sample_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = materialize_dual_soliton_relaxation_sample_ledger(
        runtime_identity_path=args.runtime_identity,
        runtime_output_identity_path=args.runtime_output_identity,
        sample_manifest_path=args.sample_manifest,
        output_path=args.output,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
