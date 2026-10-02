#!/usr/bin/env python3
"""Inspect one lagRamses FDM snapshot shard's native record structure."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from fdm_smbh_delay.fdm_shard_format import inspect_fdm_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--expected-ncpu", type=int)
    parser.add_argument("--byte-order", choices=("little", "big"), default="little")
    args = parser.parse_args()
    result = inspect_fdm_shard(
        args.shard,
        expected_ncpu=args.expected_ncpu,
        byte_order="<" if args.byte_order == "little" else ">",
    )
    record = asdict(result)
    record["path"] = str(result.path)
    record["status"] = "fdm_shard_structure_verified_no_physics_extracted"
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
