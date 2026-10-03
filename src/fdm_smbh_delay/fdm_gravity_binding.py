"""Hash-bound Poisson potential sources for a verified FDM relaxation series.

The ledger checks that all gravity shards and the potential-valid marker
belong to the exact wave-output sequence.  It does not infer a Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .dual_soliton_relaxation import (
    DualSolitonRelaxationSample,
    read_verified_dual_soliton_relaxation_sample_ledger,
)
from .fdm_gravity_shard import (
    inspect_fdm_gravity_shard_pair,
    read_valid_poisson_potential_marker,
)
from .lagramses_fdm_provenance import read_lagramses_fdm_outer_wave_provenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"gravity source is not a regular file: {source}")
    return {"path": str(source), "sha256": _sha256(source)}


def _gravity_files(sample: DualSolitonRelaxationSample) -> tuple[Path, ...]:
    provenance = read_lagramses_fdm_outer_wave_provenance(sample.raw_provenance_path)
    if provenance.mpi_ncpu != len(sample.wave_snapshot_files):
        raise ValueError("gravity binding MPI rank count differs from FDM sources")
    if not provenance.psi_snapshot_prefix.startswith("fdm_"):
        raise ValueError("gravity binding FDM snapshot prefix is invalid")
    prefix = "grav_" + provenance.psi_snapshot_prefix.removeprefix("fdm_")
    output_directory = provenance.source_path.parent
    directories = [output_directory]
    for child in output_directory.iterdir():
        if child.is_dir() and child.name.startswith("group_"):
            if re.fullmatch(r"group_\d{5}", child.name) is None:
                raise ValueError("gravity output contains an invalid group directory")
            directories.append(child)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{5}})$")
    by_rank: dict[int, Path] = {}
    for directory in directories:
        for candidate in directory.iterdir():
            if not candidate.name.startswith(prefix):
                continue
            match = pattern.fullmatch(candidate.name)
            if match is None or not candidate.is_file():
                raise ValueError("gravity shard name or type is invalid")
            rank = int(match.group(1))
            if rank in by_rank:
                raise ValueError("gravity shard rank is duplicated")
            resolved = candidate.resolve()
            if not resolved.is_relative_to(output_directory.resolve()):
                raise ValueError("gravity shard resolves outside its raw output")
            by_rank[rank] = resolved
    ranks = tuple(range(1, provenance.mpi_ncpu + 1))
    if tuple(sorted(by_rank)) != ranks:
        raise ValueError("censored_missing_or_extra_gravity_shards")
    paths = tuple(by_rank[rank] for rank in ranks)
    source_paths = (
        *(item.path for item in sample.wave_snapshot_files),
        *(item.path for item in sample.amr_topology_files),
        *paths,
    )
    identities = tuple((path.stat().st_dev, path.stat().st_ino) for path in source_paths)
    if len(set(identities)) != len(identities):
        raise ValueError("gravity shard aliases a wave or AMR source")
    return paths


def _bind_sample(
    sample: DualSolitonRelaxationSample,
    *,
    particle_density_included: bool,
) -> dict[str, Any]:
    gravity_files = _gravity_files(sample)
    marker_path = sample.raw_provenance_path.parent / "POISSON_PHI_VALID"
    if marker_path.resolve().parent != sample.raw_provenance_path.parent.resolve():
        raise ValueError("Poisson potential-valid marker resolves outside its raw output")
    nlevelmax = None
    for rank, (wave, gravity_path) in enumerate(
        zip(sample.wave_snapshot_files, gravity_files, strict=True), start=1
    ):
        if not wave.path.name.endswith(f"{rank:05d}"):
            raise ValueError("gravity binding FDM rank order is invalid")
        pair = inspect_fdm_gravity_shard_pair(
            wave.path, gravity_path,
            particle_density_included=particle_density_included,
            expected_ncpu=len(gravity_files),
        )
        if nlevelmax is None:
            nlevelmax = pair.wave.nlevelmax
        elif nlevelmax != pair.wave.nlevelmax:
            raise ValueError("gravity shards disagree on AMR level count")
    assert nlevelmax is not None
    read_valid_poisson_potential_marker(
        marker_path,
        expected_nstep_coarse=sample.nstep_coarse,
        expected_nlevelmax=nlevelmax,
        expected_time_code=sample.time_code,
        expected_aexp=sample.aexp,
    )
    return {
        "raw_fdm_provenance": {
            "path": str(sample.raw_provenance_path),
            "sha256": sample.raw_provenance_sha256,
        },
        "poisson_phi_valid_marker": _artifact(marker_path),
        "gravity_snapshot_files": [_artifact(path) for path in gravity_files],
    }


def _record(sample_ledger: Any, *, particle_density_included: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "fdm_gravity_sources_bound_pending_hamiltonian",
        "interpretation": (
            "complete native Poisson source identity and phi-valid marker only; "
            "no wave kinetic energy, Hamiltonian, or angular momentum is inferred"
        ),
        "sample_ledger": _artifact(sample_ledger.source_path),
        "particle_density_included": particle_density_included,
        "samples": [
            _bind_sample(sample, particle_density_included=particle_density_included)
            for sample in sample_ledger.samples
        ],
    }


def materialize_fdm_gravity_source_binding(
    sample_ledger_path: str | Path,
    output_path: str | Path,
    *,
    particle_density_included: bool = False,
) -> dict[str, Any]:
    """Publish a verified no-overwrite source ledger for later energy extraction."""

    if not isinstance(particle_density_included, bool):
        raise ValueError("gravity particle-density layout must be explicit")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise ValueError("gravity source-binding output already exists")
    ledger = read_verified_dual_soliton_relaxation_sample_ledger(sample_ledger_path)
    record = _record(ledger, particle_density_included=particle_density_included)
    after = read_verified_dual_soliton_relaxation_sample_ledger(sample_ledger_path)
    if after.source_sha256 != ledger.source_sha256 or after.as_dict() != ledger.as_dict():
        raise ValueError("FDM sample ledger changed during gravity binding")
    if _record(after, particle_density_included=particle_density_included) != record:
        raise ValueError("FDM gravity sources changed during binding")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
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
    return record


def read_verified_fdm_gravity_source_binding(path: str | Path) -> dict[str, Any]:
    """Rehash and reconstruct every marker and gravity shard before use."""

    source = Path(path).expanduser().resolve()
    try:
        saved = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read FDM gravity binding: {error}") from error
    if (
        not isinstance(saved, dict)
        or set(saved) != {
            "schema_version", "status", "interpretation", "sample_ledger",
            "particle_density_included", "samples",
        }
        or saved.get("schema_version") != 1
        or saved.get("status") != "fdm_gravity_sources_bound_pending_hamiltonian"
        or not isinstance(saved.get("particle_density_included"), bool)
        or not isinstance(saved.get("sample_ledger"), dict)
        or set(saved["sample_ledger"]) != {"path", "sha256"}
        or not isinstance(saved["sample_ledger"].get("path"), str)
        or not isinstance(saved["sample_ledger"].get("sha256"), str)
    ):
        raise ValueError("FDM gravity binding schema is invalid")
    ledger_path = Path(saved["sample_ledger"]["path"]).expanduser().resolve()
    ledger = read_verified_dual_soliton_relaxation_sample_ledger(ledger_path)
    reconstructed = _record(
        ledger, particle_density_included=saved["particle_density_included"]
    )
    if saved != reconstructed:
        raise ValueError("FDM gravity binding no longer matches its sources")
    return reconstructed
