#!/usr/bin/env python3
"""Inspect one lagRamses FDM snapshot shard's native record structure."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from fdm_smbh_delay.fdm_shard_format import (
    inspect_fdm_amr_shard_pair,
    inspect_fdm_shard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--amr", type=Path)
    parser.add_argument("--simple-boundary", choices=("true", "false"))
    parser.add_argument("--expected-ncpu", type=int)
    parser.add_argument("--byte-order", choices=("little", "big"), default="little")
    args = parser.parse_args()
    byte_order = "<" if args.byte_order == "little" else ">"
    if args.amr is not None:
        if args.simple_boundary is None:
            parser.error("--simple-boundary is required with --amr")
        pair = inspect_fdm_amr_shard_pair(
            args.shard,
            args.amr,
            simple_boundary=args.simple_boundary == "true",
            expected_ncpu=args.expected_ncpu,
            byte_order=byte_order,
        )
        record = {
            "status": "fdm_amr_shard_frames_aligned_no_physics_extracted",
            "wave_path": str(pair.wave.path),
            "amr_path": str(pair.amr.path),
            "ncpu": pair.wave.ncpu,
            "ndim": pair.wave.ndim,
            "nlevelmax": pair.wave.nlevelmax,
            "grid_counts_per_level": pair.wave.grid_counts_per_level,
            "amr_fine_array_records": pair.amr_fine_array_records,
        }
    else:
        if args.simple_boundary is not None:
            parser.error("--simple-boundary requires --amr")
        result = inspect_fdm_shard(
            args.shard,
            expected_ncpu=args.expected_ncpu,
            byte_order=byte_order,
        )
        record = asdict(result)
        record["path"] = str(result.path)
        record["status"] = "fdm_shard_structure_verified_no_physics_extracted"
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
