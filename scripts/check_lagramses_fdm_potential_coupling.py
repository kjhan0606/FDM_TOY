#!/usr/bin/env python3
"""Measure source-bound FDM rho*phi leaf moments without claiming energy.

Run manually on Lageunha, one process/thread.  No FFT or simulation is
launched.  The Poisson zero-point and any sink contribution remain unresolved.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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
from fdm_smbh_delay.fdm_gravity_binding import read_verified_fdm_gravity_source_binding
from fdm_smbh_delay.fdm_potential_coupling import (
    reconstruct_fdm_potential_coupling,
    summarize_owned_leaf_potential_coupling,
)
from fdm_smbh_delay.lagramses_fdm_provenance import read_lagramses_fdm_outer_wave_provenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
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
    parser.add_argument("--gravity-binding", required=True, type=Path)
    parser.add_argument("--coarse-cells-per-box", required=True, type=int)
    parser.add_argument("--simple-boundary", choices=("true", "false"), required=True)
    parser.add_argument("--maximum-array-mib", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if "lageunha" not in socket.gethostname().lower():
        parser.error("FDM potential-coupling extraction is permitted only on Lageunha")
    if args.coarse_cells_per_box < 1 or args.maximum_array_mib < 1:
        parser.error("coarse geometry and array memory bound must be positive")
    if args.output.expanduser().resolve().exists():
        parser.error("potential-coupling output already exists")
    source = args.gravity_binding.expanduser().resolve()
    source_hash = _sha256(source)
    binding = read_verified_fdm_gravity_source_binding(source)
    ledger = read_verified_dual_soliton_relaxation_sample_ledger(
        binding["sample_ledger"]["path"]
    )
    if len(binding["samples"]) != len(ledger.samples):
        raise ValueError("gravity binding and sample ledger lengths disagree")
    results = []
    for sample, bound in zip(ledger.samples, binding["samples"], strict=True):
        if bound["raw_fdm_provenance"]["path"] != str(sample.raw_provenance_path):
            raise ValueError("gravity binding sample order differs from FDM ledger")
        provenance = read_lagramses_fdm_outer_wave_provenance(
            sample.raw_provenance_path
        )
        gravity_files = bound["gravity_snapshot_files"]
        if provenance.mpi_ncpu is None or len(gravity_files) != provenance.mpi_ncpu:
            raise ValueError("gravity binding rank count differs from raw provenance")
        summaries = [
            summarize_owned_leaf_potential_coupling(
                wave.path, amr.path, Path(gravity["path"]),
                owner_rank=rank,
                simple_boundary=args.simple_boundary == "true",
                fdm_use_hjm=provenance.fdm_use_hjm,
                fdm_first_wave_level=provenance.fdm_first_wave_level,
                particle_density_included=binding["particle_density_included"],
                expected_ncpu=provenance.mpi_ncpu,
                maximum_array_bytes=args.maximum_array_mib * 1024 * 1024,
            )
            for rank, (wave, amr, gravity) in enumerate(
                zip(
                    sample.wave_snapshot_files,
                    sample.amr_topology_files,
                    gravity_files,
                    strict=True,
                ), start=1,
            )
        ]
        results.append(reconstruct_fdm_potential_coupling(
            summaries, provenance,
            coarse_cells_per_box=args.coarse_cells_per_box,
        ))
    if (
        _sha256(source) != source_hash
        or read_verified_fdm_gravity_source_binding(source) != binding
    ):
        raise ValueError("gravity binding changed during potential extraction")
    status = (
        "potential_coupling_series_pending_full_hamiltonian"
        if all(item.status == "potential_coupling_measured_pending_full_hamiltonian" for item in results)
        else "censored_potential_coupling_series"
    )
    record = {
        "schema_version": 1,
        "status": status,
        "gravity_binding": {"path": str(source), "sha256": source_hash},
        "interpretation": (
            "raw gauge-dependent integral rho*phi dV only; no kinetic energy, "
            "separated sink interaction, Hamiltonian conservation, or delay"
        ),
        "samples": [asdict(item) for item in results],
    }
    _publish(args.output, record)
    print(json.dumps({"status": status, "output": str(args.output.resolve())}))
    return 0 if status == "potential_coupling_series_pending_full_hamiltonian" else 1


if __name__ == "__main__":
    raise SystemExit(main())
