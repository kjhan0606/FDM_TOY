#!/usr/bin/env python3
"""Compare one source-verified pure-FDM snapshot with writer current.

Run manually on Lageunha, one sample and one process at a time.  This does
not measure the composite AMR kinetic Hamiltonian or certify relaxation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
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
from fdm_smbh_delay.dual_soliton_preflight import read_lagramses_namelist_assignment
from fdm_smbh_delay.fdm_shard_format import inspect_fdm_shard
from fdm_smbh_delay.fdm_wave_stencil import (
    assemble_uniform_fft_base_from_shards,
    check_fdm_wave_writer_identity,
    measure_fdm_shard_same_level_stencil,
    read_fdm_shard_level_fields,
)
from fdm_smbh_delay.fdm_zoom_runtime_identity import (
    read_verified_fdm_declared_zoom_runtime_outputs,
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
    parser.add_argument("--measure-uniform-fft-base", action="store_true")
    parser.add_argument("--maximum-uniform-cells", type=int, default=1_000_000)
    parser.add_argument("--maximum-total-base-grids", type=int, default=200_000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if "lageunha" not in socket.gethostname().lower():
        parser.error("FDM wave-current shard reading is permitted only on Lageunha")
    if (
        args.sample_index < 0 or args.maximum_grids_per_level < 1
        or args.maximum_array_mib < 1 or args.maximum_uniform_cells < 8
        or args.maximum_total_base_grids < 1
    ):
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
    structures = []
    for rank, (wave, amr) in enumerate(
        zip(sample.wave_snapshot_files, sample.amr_topology_files, strict=True), start=1
    ):
        structure = inspect_fdm_shard(wave.path, expected_ncpu=provenance.mpi_ncpu)
        structures.append(structure)
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
    uniform_base = None
    if args.measure_uniform_fft_base:
        outputs = read_verified_fdm_declared_zoom_runtime_outputs(
            ledger.runtime_output_identity_path
        )
        matches = [
            record for record in outputs.outputs
            if Path(record["raw_fdm_provenance"]["path"]).resolve()
            == sample.raw_provenance_path
        ]
        if len(matches) != 1:
            raise ValueError("sample does not identify one verified effective namelist")
        namelist = Path(matches[0]["namelist_copy"]["path"])
        level_token = read_lagramses_namelist_assignment(
            namelist, group="AMR_PARAMS", name="levelmin"
        )
        try:
            levelmin = int(level_token)
        except ValueError as error:
            raise ValueError("effective levelmin is not an integer") from error
        box_token = read_lagramses_namelist_assignment(
            namelist, group="AMR_PARAMS", name="boxlen"
        )
        try:
            effective_boxlen = float(box_token.replace("D", "E").replace("d", "e"))
        except ValueError as error:
            raise ValueError("effective boxlen is not numeric") from error
        if not math.isfinite(effective_boxlen) or effective_boxlen <= 0.0:
            raise ValueError("effective boxlen must be finite and positive")
        if levelmin < 1 or any(levelmin > item.nlevelmax for item in structures):
            raise ValueError("effective levelmin lies outside the wave shards")
        if sum(
            item.grid_counts_per_level[levelmin - 1] for item in structures
        ) > args.maximum_total_base_grids:
            raise ValueError("uniform FFT base exceeds the total-grid memory bound")
        base_fields = [
            read_fdm_shard_level_fields(
                wave.path, amr.path, owner_rank=rank, level=levelmin,
                simple_boundary=args.simple_boundary == "true",
                fdm_use_hjm=False,
                fdm_first_wave_level=provenance.fdm_first_wave_level,
                expected_ncpu=provenance.mpi_ncpu,
                maximum_grids=args.maximum_grids_per_level,
                maximum_array_bytes=args.maximum_array_mib * 1024 * 1024,
            )
            for rank, (wave, amr) in enumerate(
                zip(sample.wave_snapshot_files, sample.amr_topology_files, strict=True),
                start=1,
            )
        ]
        if any(not math.isclose(
            item.boxlen_code, effective_boxlen, rel_tol=1.0e-12, abs_tol=1.0e-12
        ) for item in base_fields):
            raise ValueError("effective boxlen disagrees with native AMR shard headers")
        uniform_base = assemble_uniform_fft_base_from_shards(
            base_fields, declared_levelmin=levelmin,
            hbar_code=provenance.hbar_code,
            maximum_cells=args.maximum_uniform_cells,
        )
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
        "uniform_fft_base": None if uniform_base is None else asdict(uniform_base),
        "uniform_fft_base_status": (
            "not_requested" if uniform_base is None
            else "censored_incomplete_or_writer_mismatch"
            if uniform_base.quadratic is None
            or identity.status != "saved_wave_writer_current_identity_matches"
            else "candidate_pending_use_fftw_build_and_units_verification"
        ),
    }
    _publish_json_without_overwrite(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
