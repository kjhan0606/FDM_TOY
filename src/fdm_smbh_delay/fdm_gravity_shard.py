"""Structural binding for native lagRamses FDM and Poisson snapshot shards.

The selected VPATH resolves ``backup_poisson`` to ``poisson/output_poisson.f90``.
The normal build writes phi then NDIM force arrays for each AMR child.  A
compile-time particle-density prefix requires explicit declaration here.
These checks do not calculate the wave Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from .fdm_shard_format import FDMShardStructure, _RecordReader, inspect_fdm_shard


@dataclass(frozen=True)
class GravityShardStructure:
    path: Path
    ncpu: int
    ndim: int
    nlevelmax: int
    nboundary: int
    particle_density_included: bool
    grid_counts_by_level_and_domain: tuple[tuple[int, ...], ...]
    field_array_records: int


@dataclass(frozen=True)
class FDMGravityShardPair:
    wave: FDMShardStructure
    gravity: GravityShardStructure


@dataclass(frozen=True)
class ValidPoissonPotentialMarker:
    path: Path
    nstep_coarse: int
    nlevelmax: int
    time_code: float
    aexp: float


def inspect_gravity_shard(
    path: str | Path,
    *,
    ndim: int,
    particle_density_included: bool = False,
    expected_ncpu: int | None = None,
    byte_order: str = "<",
) -> GravityShardStructure:
    """Validate every Fortran frame without loading potential/force payloads."""

    if byte_order not in {"<", ">"} or ndim not in {1, 2, 3}:
        raise ValueError("gravity shard format declaration is invalid")
    if not isinstance(particle_density_included, bool):
        raise ValueError("gravity particle-density layout must be explicitly boolean")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"gravity shard is not a regular file: {source}")
    with source.open("rb") as stream:
        reader = _RecordReader(stream, byte_order=byte_order)
        ncpu = reader.scalar()
        nvar = reader.scalar()
        nlevelmax = reader.scalar()
        nboundary = reader.scalar()
        expected_nvar = ndim + 1 + int(particle_density_included)
        if (
            ncpu < 1 or nlevelmax < 1 or nboundary < 0
            or nvar != expected_nvar
            or (expected_ncpu is not None and ncpu != expected_ncpu)
        ):
            raise ValueError("gravity shard header differs from the declared writer layout")
        minimum_block_bytes = 24 * nlevelmax * (ncpu + nboundary)
        if minimum_block_bytes > source.stat().st_size - stream.tell():
            raise ValueError("gravity shard header demands more blocks than the file can hold")
        counts = []
        field_records = 0
        for level in range(1, nlevelmax + 1):
            level_counts = []
            for _ in range(ncpu + nboundary):
                recorded_level = reader.scalar()
                ncache = reader.scalar()
                if recorded_level != level or ncache < 0:
                    raise ValueError("gravity shard block level or grid count is invalid")
                level_counts.append(ncache)
                if ncache:
                    for _ in range(nvar * (1 << ndim)):
                        reader.record(8 * ncache)
                        field_records += 1
            counts.append(tuple(level_counts))
        if stream.read(1):
            raise ValueError("gravity shard has unexpected trailing bytes")
    return GravityShardStructure(
        path=source,
        ncpu=ncpu,
        ndim=ndim,
        nlevelmax=nlevelmax,
        nboundary=nboundary,
        particle_density_included=particle_density_included,
        grid_counts_by_level_and_domain=tuple(counts),
        field_array_records=field_records,
    )


def inspect_fdm_gravity_shard_pair(
    wave_path: str | Path,
    gravity_path: str | Path,
    *,
    particle_density_included: bool = False,
    expected_ncpu: int | None = None,
    byte_order: str = "<",
) -> FDMGravityShardPair:
    """Require identical per-level and per-domain FDM/Poisson topology."""

    wave = inspect_fdm_shard(
        wave_path, expected_ncpu=expected_ncpu, byte_order=byte_order
    )
    gravity = inspect_gravity_shard(
        gravity_path,
        ndim=wave.ndim,
        particle_density_included=particle_density_included,
        expected_ncpu=wave.ncpu,
        byte_order=byte_order,
    )
    if (
        gravity.nlevelmax != wave.nlevelmax
        or gravity.nboundary != wave.nboundary
        or gravity.grid_counts_by_level_and_domain != wave.grid_counts_by_level_and_domain
    ):
        raise ValueError("gravity and FDM shard grid layouts disagree")
    return FDMGravityShardPair(wave=wave, gravity=gravity)


def read_valid_poisson_potential_marker(
    path: str | Path,
    *,
    expected_nstep_coarse: int,
    expected_nlevelmax: int,
    expected_time_code: float,
    expected_aexp: float,
) -> ValidPoissonPotentialMarker:
    """Match the saved phi-valid marker to one raw FDM output's metadata."""

    source = Path(path).expanduser().resolve()
    try:
        lines = source.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read Poisson potential-valid marker: {error}") from error
    if len(lines) != 2 or lines[0].strip() != "LAGRAMSES_POISSON_PHI_VALID_V1":
        raise ValueError("Poisson potential-valid marker header is invalid")
    parts = lines[1].split()
    if len(parts) != 4:
        raise ValueError("Poisson potential-valid marker metadata is invalid")
    try:
        step = int(parts[0])
        levelmax = int(parts[1])
        time = float(parts[2].replace("D", "E").replace("d", "e"))
        aexp = float(parts[3].replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError("Poisson potential-valid marker metadata is not numeric") from error
    if (
        step < 0 or levelmax < 1 or not math.isfinite(time)
        or not math.isfinite(aexp) or aexp <= 0.0
        or step != expected_nstep_coarse
        or levelmax != expected_nlevelmax
        or not math.isclose(time, expected_time_code, rel_tol=1.0e-12, abs_tol=1.0e-14)
        or not math.isclose(aexp, expected_aexp, rel_tol=1.0e-12, abs_tol=1.0e-14)
    ):
        raise ValueError("Poisson potential-valid marker differs from the FDM output")
    return ValidPoissonPotentialMarker(
        path=source,
        nstep_coarse=step,
        nlevelmax=levelmax,
        time_code=time,
        aexp=aexp,
    )
