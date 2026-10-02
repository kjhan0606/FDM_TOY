#!/usr/bin/env python3
"""Cross-check one complete FDM output's leaf mass against raw provenance.

Run this bounded, one-process reader on Lageunha, not the shared login node.
Every MPI rank must be supplied explicitly using ``--pair WAVE AMR``.  The
result is a reader/writer identity check, not a relaxation or conservation pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from fdm_smbh_delay.fdm_leaf_mass import check_fdm_leaf_mass_identity
from fdm_smbh_delay.fdm_shard_format import summarize_owned_leaf_amplitudes
from fdm_smbh_delay.lagramses_fdm_provenance import (
    read_lagramses_fdm_outer_wave_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--coarse-cells-per-box", type=int, required=True)
    parser.add_argument("--simple-boundary", choices=("true", "false"), required=True)
    parser.add_argument("--maximum-array-mib", type=int, default=64)
    parser.add_argument("--byte-order", choices=("little", "big"), default="little")
    parser.add_argument(
        "--pair", nargs=2, action="append", metavar=("WAVE", "AMR"), required=True
    )
    args = parser.parse_args()
    provenance = read_lagramses_fdm_outer_wave_provenance(args.provenance)
    if provenance.mpi_ncpu is None or len(args.pair) != provenance.mpi_ncpu:
        parser.error("one wave/AMR pair is required for every MPI rank")
    if args.maximum_array_mib < 1:
        parser.error("--maximum-array-mib must be positive")
    ranks = []
    for wave_name, _ in args.pair:
        suffix = Path(wave_name).name[-5:]
        if len(suffix) != 5 or not suffix.isdigit():
            parser.error("every wave shard must end in a five-digit MPI rank")
        ranks.append(int(suffix))
    if sorted(ranks) != list(range(1, provenance.mpi_ncpu + 1)):
        parser.error("wave/AMR shard pairs must cover every MPI rank exactly once")
    summaries = [
        summarize_owned_leaf_amplitudes(
            wave_name,
            amr_name,
            owner_rank=rank,
            simple_boundary=args.simple_boundary == "true",
            fdm_use_hjm=provenance.fdm_use_hjm,
            fdm_first_wave_level=provenance.fdm_first_wave_level,
            expected_ncpu=provenance.mpi_ncpu,
            byte_order="<" if args.byte_order == "little" else ">",
            maximum_array_bytes=args.maximum_array_mib * 1024 * 1024,
        )
        for (wave_name, amr_name), rank in zip(args.pair, ranks, strict=True)
    ]
    result = check_fdm_leaf_mass_identity(
        summaries,
        provenance,
        coarse_cells_per_box=args.coarse_cells_per_box,
    )
    record = asdict(result)
    record["provenance_path"] = str(result.provenance_path)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if result.status == "raw_leaf_mass_reconstruction_matches_writer" else 1


if __name__ == "__main__":
    raise SystemExit(main())
