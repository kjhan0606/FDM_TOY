#!/usr/bin/env python3
"""Scan a lagRamses writer source for required CDM/SIDM force tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.lagramses_writer_audit import (
    audit_lagramses_writer_force_accounting,
    write_writer_force_accounting_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="one lagRamses writer source file")
    parser.add_argument("output", type=Path, help="JSON audit output")
    args = parser.parse_args()
    audit = audit_lagramses_writer_force_accounting(args.source)
    write_writer_force_accounting_audit(args.output, audit)
    print(json.dumps(audit.as_dict(), indent=2, sort_keys=True))
    return 0 if audit.tokens_present else 2


if __name__ == "__main__":
    raise SystemExit(main())
