#!/usr/bin/env python3
"""Revalidate a saved pure-FDM outer submission decision; never submit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.pure_fdm_outer_submission import (
    read_verified_pure_fdm_outer_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("specification", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("preflight", type=Path)
    parser.add_argument("writer_source", type=Path)
    parser.add_argument("runtime_attestation", type=Path)
    args = parser.parse_args()
    try:
        decision = read_verified_pure_fdm_outer_submission(
            args.record,
            args.specification,
            args.manifest,
            args.preflight,
            args.writer_source,
            args.runtime_attestation,
        )
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "outer_submission_verification_failed", "reason": str(error)}))
        return 2
    print(json.dumps({"status": decision.status, "verified": True}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
