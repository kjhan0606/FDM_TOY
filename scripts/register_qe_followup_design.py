#!/usr/bin/env python3
"""Write one bounded, no-overwrite q/e follow-up design; launch nothing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.qe_followup_design import build_qe_followup_design


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coarse-resolution", required=True, type=int)
    parser.add_argument("--fine-resolution", required=True, type=int)
    parser.add_argument("--separation-bin-edges-pc", required=True)
    parser.add_argument("--duration-myr", required=True, type=float)
    parser.add_argument("--minimum-orbits-per-bin", type=int, default=8)
    parser.add_argument("--reference-gpu-memory-gib", type=float, default=80.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        edges = tuple(float(value) for value in args.separation_bin_edges_pc.split(","))
    except ValueError as error:
        parser.error(f"invalid separation-bin edges: {error}")
    payload = build_qe_followup_design(
        case_id=args.case_id,
        physical_cases=args.cases.expanduser().resolve(),
        run_manifest=args.manifest.expanduser().resolve(),
        coarse_resolution=args.coarse_resolution,
        fine_resolution=args.fine_resolution,
        separation_bin_edges_pc=edges,
        duration_myr=args.duration_myr,
        minimum_orbits_per_bin=args.minimum_orbits_per_bin,
        reference_gpu_memory_gib=args.reference_gpu_memory_gib,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output.parent, delete=False
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
