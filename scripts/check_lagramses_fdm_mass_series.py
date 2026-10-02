#!/usr/bin/env python3
"""Measure source-bound FDM leaf-mass drift from a verified sample ledger.

Run manually on Lageunha with one process and one numerical-library thread.
This reads every listed FDM/AMR shard; it submits no job and performs no FFT.
The output is mass-only evidence, never a complete relaxation acceptance.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import tempfile

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

from fdm_smbh_delay.dual_soliton_relaxation import (
    read_verified_dual_soliton_relaxation_sample_ledger,
)
from fdm_smbh_delay.fdm_leaf_mass import (
    assess_fdm_leaf_mass_series,
    check_fdm_leaf_mass_identity,
    reconstruct_fdm_radial_mass_profile,
)
from fdm_smbh_delay.fdm_shard_format import summarize_owned_leaf_amplitudes
from fdm_smbh_delay.lagramses_fdm_provenance import (
    read_lagramses_fdm_outer_wave_provenance,
)


def _publish_json_without_overwrite(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ledger", required=True, type=Path)
    parser.add_argument("--coarse-cells-per-box", required=True, type=int)
    parser.add_argument("--simple-boundary", choices=("true", "false"), required=True)
    parser.add_argument("--maximum-array-mib", type=int, default=64)
    parser.add_argument("--coarse-origin", nargs=3, type=int)
    parser.add_argument("--radial-edges-box", nargs="+", type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if "lageunha" not in socket.gethostname().lower():
        parser.error("FDM mass-series shard reading is permitted only on Lageunha")
    if args.maximum_array_mib < 1:
        parser.error("--maximum-array-mib must be positive")
    if args.coarse_cells_per_box < 1:
        parser.error("--coarse-cells-per-box must be positive")
    radial_requested = args.coarse_origin is not None or args.radial_edges_box is not None
    if radial_requested and (args.coarse_origin is None or args.radial_edges_box is None):
        parser.error("radial profiles require both --coarse-origin and --radial-edges-box")
    if args.output.expanduser().resolve().exists():
        parser.error("mass-series output already exists")
    ledger = read_verified_dual_soliton_relaxation_sample_ledger(args.sample_ledger)
    identities = []
    radial_profiles = []
    for sample in ledger.samples:
        provenance = read_lagramses_fdm_outer_wave_provenance(
            sample.raw_provenance_path
        )
        if provenance.mpi_ncpu is None or (
            len(sample.wave_snapshot_files) != provenance.mpi_ncpu
            or len(sample.amr_topology_files) != provenance.mpi_ncpu
        ):
            raise ValueError("sample shard count differs from raw FDM provenance")
        if radial_requested and (
            not provenance.fdm_dual_soliton_ic
            or provenance.fdm_dual_soliton_centres_box is None
        ):
            raise ValueError("radial profile requires declared dual-soliton seed centres")
        summaries = [
            summarize_owned_leaf_amplitudes(
                wave.path,
                amr.path,
                owner_rank=rank,
                simple_boundary=args.simple_boundary == "true",
                fdm_use_hjm=provenance.fdm_use_hjm,
                fdm_first_wave_level=provenance.fdm_first_wave_level,
                expected_ncpu=provenance.mpi_ncpu,
                maximum_array_bytes=args.maximum_array_mib * 1024 * 1024,
                radial_centres_box=(
                    provenance.fdm_dual_soliton_centres_box if radial_requested else None
                ),
                radial_edges_box=(
                    tuple(args.radial_edges_box) if radial_requested else None
                ),
                coarse_origin=(tuple(args.coarse_origin) if radial_requested else None),
                coarse_cells_per_box=(
                    args.coarse_cells_per_box if radial_requested else None
                ),
            )
            for rank, (wave, amr) in enumerate(
                zip(sample.wave_snapshot_files, sample.amr_topology_files, strict=True),
                start=1,
            )
        ]
        identity = check_fdm_leaf_mass_identity(
            summaries,
            provenance,
            coarse_cells_per_box=args.coarse_cells_per_box,
        )
        identities.append(identity)
        if radial_requested:
            radial_profiles.append(
                reconstruct_fdm_radial_mass_profile(summaries, identity)
                if identity.status == "raw_leaf_mass_reconstruction_matches_writer"
                else None
            )
    after = read_verified_dual_soliton_relaxation_sample_ledger(args.sample_ledger)
    if after.as_dict() != ledger.as_dict() or after.source_sha256 != ledger.source_sha256:
        raise ValueError("FDM sample ledger changed during mass-series extraction")
    series = assess_fdm_leaf_mass_series(ledger, identities)
    record = {
        "schema_version": 1,
        "status": series.status,
        "interpretation": (
            "verified-sample-ledger mass series only; Hamiltonian, angular "
            "momentum, core relaxation, resolution, and physical delay remain unassessed"
        ),
        "series": asdict(series),
        "snapshot_identity_checks": [asdict(item) for item in identities],
    }
    if radial_requested:
        record["two_centre_total_wave_radial_profiles"] = [
            None if item is None else asdict(item) for item in radial_profiles
        ]
    _publish_json_without_overwrite(args.output, record)
    print(json.dumps({"status": series.status, "output": str(args.output.resolve())}))
    return 0 if series.status == "mass_series_within_limit_pending_other_conservation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
