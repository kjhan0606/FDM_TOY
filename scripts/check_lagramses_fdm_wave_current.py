#!/usr/bin/env python3
"""Compare one source-verified pure-FDM snapshot with writer current.

Run manually on Lageunha, one sample and one process at a time.  This does
not measure the spectral/AMR kinetic Hamiltonian or certify relaxation.
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
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

from fdm_smbh_delay.dual_soliton_relaxation import (
    read_verified_dual_soliton_relaxation_sample_ledger,
)
from fdm_smbh_delay.fdm_shard_format import inspect_fdm_shard
from fdm_smbh_delay.fdm_wave_stencil import (
    check_fdm_wave_writer_identity,
    measure_fdm_shard_same_level_stencil,
)
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
    parser.add_argument("--sample-index", required=True, type=int)
    parser.add_argument("--simple-boundary", choices=("true", "false"), required=True)
    parser.add_argument("--maximum-grids-per-level", type=int, default=100_000)
    parser.add_argument("--maximum-array-mib", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if "lageunha" not in socket.gethostname().lower():
        parser.error("FDM wave-current shard reading is permitted only on Lageunha")
    if args.sample_index < 0 or args.maximum_grids_per_level < 1 or args.maximum_array_mib < 1:
        parser.error("sample index and memory bounds are invalid")
    if args.output.expanduser().resolve().exists():
        parser.error("wave-current output already exists")
    ledger = read_verified_dual_soliton_relaxation_sample_ledger(args.sample_ledger)
    if args.sample_index >= len(ledger.samples):
        parser.error("sample index lies outside the verified ledger")
    sample = ledger.samples[args.sample_index]
    provenance = read_lagramses_fdm_outer_wave_provenance(sample.raw_provenance_path)
    if provenance.fdm_use_hjm:
        raise ValueError("HJM phase-gradient current is outside the pure-wave reader")
    if provenance.mpi_ncpu != len(sample.wave_snapshot_files) or (
        provenance.mpi_ncpu != len(sample.amr_topology_files)
    ):
        raise ValueError("FDM wave-current shard count differs from raw provenance")
    results = []
    for rank, (wave, amr) in enumerate(
        zip(sample.wave_snapshot_files, sample.amr_topology_files, strict=True), start=1
    ):
        structure = inspect_fdm_shard(wave.path, expected_ncpu=provenance.mpi_ncpu)
        for level in range(1, structure.nlevelmax + 1):
            results.append(measure_fdm_shard_same_level_stencil(
                wave.path, amr.path, owner_rank=rank, level=level,
                simple_boundary=args.simple_boundary == "true",
                fdm_use_hjm=False,
                fdm_first_wave_level=provenance.fdm_first_wave_level,
                hbar_code=provenance.hbar_code,
                expected_ncpu=provenance.mpi_ncpu,
                maximum_grids=args.maximum_grids_per_level,
                maximum_array_bytes=args.maximum_array_mib * 1024 * 1024,
            ))
    identity = check_fdm_wave_writer_identity(results, provenance)
    # Re-read all hashes after extraction so in-place output mutation cannot
    # silently turn a measured result into an unbound one.
    after = read_verified_dual_soliton_relaxation_sample_ledger(args.sample_ledger)
    if after.source_sha256 != ledger.source_sha256:
        raise ValueError("sample ledger changed during wave-current extraction")
    payload = {
        "schema_version": 1,
        "status": identity.status,
        "interpretation": (
            "saved-wave current versus raw writer identity only; not a kinetic "
            "Hamiltonian, conservation time series, or FDM delay calibration"
        ),
        "sample_ledger": {"path": str(ledger.source_path), "sha256": ledger.source_sha256},
        "sample_index": args.sample_index,
        "raw_provenance": {
            "path": str(sample.raw_provenance_path),
            "sha256": sample.raw_provenance_sha256,
        },
        "identity": asdict(identity),
        "owner_level_results": [asdict(item) for item in results],
    }
    _publish_json_without_overwrite(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
