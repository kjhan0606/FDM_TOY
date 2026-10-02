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
    summarize_owned_leaf_amplitudes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--amr", type=Path)
    parser.add_argument("--simple-boundary", choices=("true", "false"))
    parser.add_argument("--owner-rank", type=int)
    parser.add_argument("--fdm-use-hjm", choices=("true", "false"))
    parser.add_argument("--first-wave-level", type=int)
    parser.add_argument("--maximum-array-mib", type=int, default=64)
    parser.add_argument("--expected-ncpu", type=int)
    parser.add_argument("--byte-order", choices=("little", "big"), default="little")
    args = parser.parse_args()
    byte_order = "<" if args.byte_order == "little" else ">"
    leaf_requested = args.owner_rank is not None
    if leaf_requested and (
        args.amr is None
        or args.simple_boundary is None
        or args.fdm_use_hjm is None
        or args.first_wave_level is None
    ):
        parser.error(
            "--owner-rank requires --amr, --simple-boundary, "
            "--fdm-use-hjm, and --first-wave-level"
        )
    if not leaf_requested and (
        args.fdm_use_hjm is not None or args.first_wave_level is not None
    ):
        parser.error("FDM mode controls require --owner-rank")
    if args.maximum_array_mib < 1:
        parser.error("--maximum-array-mib must be positive")
    if args.amr is not None:
        if args.simple_boundary is None:
            parser.error("--simple-boundary is required with --amr")
        if leaf_requested:
            summary = summarize_owned_leaf_amplitudes(
                args.shard,
                args.amr,
                owner_rank=args.owner_rank,
                simple_boundary=args.simple_boundary == "true",
                fdm_use_hjm=args.fdm_use_hjm == "true",
                fdm_first_wave_level=args.first_wave_level,
                expected_ncpu=args.expected_ncpu,
                byte_order=byte_order,
                maximum_array_bytes=args.maximum_array_mib * 1024 * 1024,
            )
            record = {
                "status": "owned_leaf_amplitude_sum_no_cell_volume_or_mass",
                "wave_path": str(summary.wave_path),
                "amr_path": str(summary.amr_path),
                "owner_rank": summary.owner_rank,
                "leaf_cells_by_level": summary.leaf_cells_by_level,
                "density_sum_by_level": summary.density_sum_by_level,
            }
        else:
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
