"""Bounded structural reader for lagRamses ``backup_psi`` snapshot shards.

The native writer stores one Fortran sequential record for each header scalar
and each real/imaginary cell array.  This reader seeks over field payloads; it
does not interpret AMR topology or derive physical diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True)
class FDMShardStructure:
    path: Path
    ncpu: int
    ndim: int
    nlevelmax: int
    nboundary: int
    blocks_per_level: tuple[int, ...]
    grid_counts_per_level: tuple[int, ...]
    grid_counts_by_level_and_domain: tuple[tuple[int, ...], ...]
    field_array_records: int


@dataclass(frozen=True)
class AMRShardHeader:
    path: Path
    ncpu: int
    ndim: int
    nlevelmax: int
    nboundary: int
    nx_ny_nz: tuple[int, int, int]
    boxlen_code: float
    ordering: str
    grids_by_level_and_cpu: tuple[tuple[int, ...], ...]
    boundary_grids_by_level: tuple[tuple[int, ...], ...] | None
    fine_payload_offset: int


@dataclass(frozen=True)
class FDMAMRShardPair:
    wave: FDMShardStructure
    amr: AMRShardHeader
    amr_fine_array_records: int


@dataclass(frozen=True)
class OwnedLeafAmplitudeSummary:
    wave_path: Path
    amr_path: Path
    owner_rank: int
    ncpu: int
    ndim: int
    boxlen_code: float
    fdm_use_hjm: bool
    fdm_first_wave_level: int
    leaf_cells_by_level: tuple[int, ...]
    density_sum_by_level: tuple[float, ...]
    radial_centres_box: tuple[tuple[float, ...], ...] | None = None
    radial_edges_box: tuple[float, ...] | None = None
    radial_density_sum_by_centre_level_bin: tuple[tuple[tuple[float, ...], ...], ...] | None = None


class _RecordReader:
    def __init__(self, stream: BinaryIO, *, byte_order: str) -> None:
        self.stream = stream
        self.integer = struct.Struct(byte_order + "i")

    def record(self, expected_bytes: int, *, read_payload: bool = False) -> bytes:
        return self.any_record(
            expected_bytes=expected_bytes, read_payload=read_payload
        )

    def any_record(
        self,
        *,
        expected_bytes: int | None = None,
        read_payload: bool = False,
    ) -> bytes:
        marker = self.stream.read(4)
        if len(marker) != 4:
            raise ValueError("FDM shard ends before a Fortran record marker")
        length = self.integer.unpack(marker)[0]
        if length < 0 or (expected_bytes is not None and length != expected_bytes):
            raise ValueError(
                f"FDM shard record length {length} differs from {expected_bytes}"
            )
        if read_payload:
            payload = self.stream.read(length)
            if len(payload) != length:
                raise ValueError("FDM shard ends inside a Fortran record")
        else:
            self.stream.seek(length, 1)
            payload = b""
        trailer = self.stream.read(4)
        if len(trailer) != 4 or self.integer.unpack(trailer)[0] != length:
            raise ValueError("FDM shard Fortran record trailer is invalid")
        return payload

    def scalar(self) -> int:
        return self.integer.unpack(self.record(4, read_payload=True))[0]

    def integer_array(self, count: int) -> tuple[int, ...]:
        if count < 0 or count > 2_000_000:
            raise ValueError("AMR header integer array is too large")
        payload = self.record(4 * count, read_payload=True)
        return struct.unpack(self.integer.format[0] + f"{count}i", payload)

    def scalar_real(self) -> float:
        return struct.unpack(self.integer.format[0] + "d", self.record(8, read_payload=True))[0]


def inspect_fdm_shard(
    path: str | Path,
    *,
    expected_ncpu: int | None = None,
    byte_order: str = "<",
) -> FDMShardStructure:
    """Validate record framing without loading wave fields into memory.

    The supported writer contract is a 4-byte record marker, 4-byte integer,
    and 8-byte real.  Other encodings must be declared explicitly rather than
    silently guessed from bytes.
    """

    if byte_order not in {"<", ">"}:
        raise ValueError("FDM shard byte order must be '<' or '>'")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"FDM shard is not a regular file: {source}")
    with source.open("rb") as stream:
        reader = _RecordReader(stream, byte_order=byte_order)
        ncpu = reader.scalar()
        ndim = reader.scalar()
        nlevelmax = reader.scalar()
        nboundary = reader.scalar()
        if (
            ncpu < 1
            or ndim not in {1, 2, 3}
            or nlevelmax < 1
            or nboundary < 0
            or (expected_ncpu is not None and ncpu != expected_ncpu)
        ):
            raise ValueError("FDM shard header is incompatible with expected topology")
        minimum_block_bytes = 24 * nlevelmax * (ncpu + nboundary)
        if minimum_block_bytes > source.stat().st_size - stream.tell():
            raise ValueError("FDM shard header demands more blocks than the file can hold")
        block_counts: list[int] = []
        grid_counts: list[int] = []
        domain_counts: list[tuple[int, ...]] = []
        field_records = 0
        for level in range(1, nlevelmax + 1):
            blocks = 0
            grids = 0
            level_counts: list[int] = []
            for _ in range(ncpu + nboundary):
                recorded_level = reader.scalar()
                ncache = reader.scalar()
                if recorded_level != level or ncache < 0:
                    raise ValueError("FDM shard block level or grid count is invalid")
                blocks += 1
                grids += ncache
                level_counts.append(ncache)
                if ncache:
                    for _ in range(2 * (1 << ndim)):
                        reader.record(8 * ncache)
                        field_records += 1
            block_counts.append(blocks)
            grid_counts.append(grids)
            domain_counts.append(tuple(level_counts))
        if stream.read(1):
            raise ValueError("FDM shard has unexpected trailing bytes")
    return FDMShardStructure(
        path=source,
        ncpu=ncpu,
        ndim=ndim,
        nlevelmax=nlevelmax,
        nboundary=nboundary,
        blocks_per_level=tuple(block_counts),
        grid_counts_per_level=tuple(grid_counts),
        grid_counts_by_level_and_domain=tuple(domain_counts),
        field_array_records=field_records,
    )


def read_amr_shard_header(
    path: str | Path,
    *,
    simple_boundary: bool,
    expected_ncpu: int | None = None,
    byte_order: str = "<",
) -> AMRShardHeader:
    """Read topology counts through the three coarse-cell records only.

    ``simple_boundary`` must come from the effective run configuration.  The
    fine-level AMR payload is deliberately not interpreted by this function.
    """

    if byte_order not in {"<", ">"}:
        raise ValueError("AMR shard byte order must be '<' or '>'")
    if not isinstance(simple_boundary, bool):
        raise ValueError("AMR simple_boundary must be explicitly boolean")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"AMR shard is not a regular file: {source}")
    with source.open("rb") as stream:
        reader = _RecordReader(stream, byte_order=byte_order)
        ncpu = reader.scalar()
        ndim = reader.scalar()
        nx_ny_nz = reader.integer_array(3)
        nlevelmax = reader.scalar()
        reader.scalar()  # ngridmax
        nboundary = reader.scalar()
        reader.scalar()  # ngrid_current
        boxlen = reader.scalar_real()
        if (
            ncpu < 1
            or ndim not in {1, 2, 3}
            or nlevelmax < 1
            or nboundary < 0
            or any(value < 1 for value in nx_ny_nz)
            or not math.isfinite(boxlen)
            or boxlen <= 0.0
            or (expected_ncpu is not None and ncpu != expected_ncpu)
        ):
            raise ValueError("AMR shard header is incompatible with expected topology")
        noutput, _, _ = reader.integer_array(3)
        if noutput < 0 or noutput > 2_000_000:
            raise ValueError("AMR output schedule length is invalid")
        reader.record(8 * noutput)  # tout
        reader.record(8 * noutput)  # aout
        reader.record(8)  # t
        reader.record(8 * nlevelmax)  # dtold
        reader.record(8 * nlevelmax)  # dtnew
        reader.record(8)  # nstep, nstep_coarse
        reader.record(24)  # const, mass_tot_0, rho_tot
        reader.record(56)  # seven cosmology scalars
        reader.record(40)  # five expansion and energy scalars
        reader.record(8)  # mass_sph
        domain_entries = ncpu * nlevelmax
        reader.record(4 * domain_entries)  # headl
        reader.record(4 * domain_entries)  # taill
        cpu_counts = reader.integer_array(domain_entries)
        reader.record(40 * nlevelmax)  # numbtot
        boundary_counts = None
        if simple_boundary:
            boundary_entries = nboundary * nlevelmax
            reader.record(4 * boundary_entries)  # headb
            reader.record(4 * boundary_entries)  # tailb
            boundary_counts = reader.integer_array(boundary_entries)
        reader.any_record()  # free-memory counters
        # The selected amr_parameters source declares character(len=128).
        ordering_bytes = reader.record(128, read_payload=True)
        try:
            ordering = ordering_bytes.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("AMR ordering record is not ASCII") from error
        if ordering not in {"hilbert", "planar", "angular", "bisection", "ksection"}:
            raise ValueError("AMR ordering record is unsupported")
        if ordering == "bisection":
            ordering_records = 5
        elif ordering == "ksection":
            ordering_records = 10
        else:
            ordering_records = 1
        for _ in range(ordering_records):
            reader.any_record()
        for _ in range(3):
            reader.any_record()  # coarse son, flag, and cpu map
        if any(count < 0 for count in cpu_counts) or (
            boundary_counts is not None and any(count < 0 for count in boundary_counts)
        ):
            raise ValueError("AMR grid counts must be non-negative")
        cpu_by_level = tuple(
            tuple(cpu_counts[level * ncpu : (level + 1) * ncpu])
            for level in range(nlevelmax)
        )
        boundary_by_level = (
            None
            if boundary_counts is None
            else tuple(
                tuple(
                    boundary_counts[
                        level * nboundary : (level + 1) * nboundary
                    ]
                )
                for level in range(nlevelmax)
            )
        )
        offset = stream.tell()
    return AMRShardHeader(
        path=source,
        ncpu=ncpu,
        ndim=ndim,
        nlevelmax=nlevelmax,
        nboundary=nboundary,
        nx_ny_nz=nx_ny_nz,
        boxlen_code=boxlen,
        ordering=ordering,
        grids_by_level_and_cpu=cpu_by_level,
        boundary_grids_by_level=boundary_by_level,
        fine_payload_offset=offset,
    )


def inspect_fdm_amr_shard_pair(
    wave_path: str | Path,
    amr_path: str | Path,
    *,
    simple_boundary: bool,
    expected_ncpu: int | None = None,
    byte_order: str = "<",
) -> FDMAMRShardPair:
    """Require equal FDM/AMR block counts and valid native AMR payload frames.

    When boundary-grid counts are omitted by ``simple_boundary=False``, only
    zero-boundary outputs can be structurally bound without another trusted
    source for those counts.  No leaf map or physical field is produced.
    """

    wave = inspect_fdm_shard(
        wave_path, expected_ncpu=expected_ncpu, byte_order=byte_order
    )
    amr = read_amr_shard_header(
        amr_path,
        simple_boundary=simple_boundary,
        expected_ncpu=wave.ncpu,
        byte_order=byte_order,
    )
    if (
        wave.ndim != amr.ndim
        or wave.nlevelmax != amr.nlevelmax
        or wave.nboundary != amr.nboundary
    ):
        raise ValueError("FDM and AMR shard topology headers disagree")
    if amr.nboundary and amr.boundary_grids_by_level is None:
        raise ValueError("AMR boundary-grid counts are unavailable for FDM binding")
    records = 0
    with amr.path.open("rb") as stream:
        stream.seek(amr.fine_payload_offset)
        reader = _RecordReader(stream, byte_order=byte_order)
        for level in range(amr.nlevelmax):
            counts = (
                *amr.grids_by_level_and_cpu[level],
                *(amr.boundary_grids_by_level[level]
                  if amr.boundary_grids_by_level is not None else ()),
            )
            if len(counts) != wave.ncpu + wave.nboundary:
                raise ValueError("AMR fine block count disagrees with FDM shard")
            if counts != wave.grid_counts_by_level_and_domain[level]:
                raise ValueError("AMR per-domain grid counts disagree with FDM shard")
            for ncache in counts:
                if not ncache:
                    continue
                for _ in range(3):  # grid index, next, prev
                    reader.record(4 * ncache)
                    records += 1
                for _ in range(amr.ndim):  # grid centre coordinates
                    reader.record(8 * ncache)
                    records += 1
                integer_records = 1 + 2 * amr.ndim + 3 * (1 << amr.ndim)
                for _ in range(integer_records):
                    reader.record(4 * ncache)
                    records += 1
        if stream.read(1):
            raise ValueError("AMR shard has unexpected trailing bytes")
    return FDMAMRShardPair(
        wave=wave, amr=amr, amr_fine_array_records=records
    )


def summarize_owned_leaf_amplitudes(
    wave_path: str | Path,
    amr_path: str | Path,
    *,
    owner_rank: int,
    simple_boundary: bool,
    fdm_use_hjm: bool,
    fdm_first_wave_level: int,
    expected_ncpu: int | None = None,
    byte_order: str = "<",
    maximum_array_bytes: int = 64 * 1024 * 1024,
    radial_centres_box: tuple[tuple[float, ...], ...] | None = None,
    radial_edges_box: tuple[float, ...] | None = None,
    coarse_origin: tuple[int, ...] | None = None,
    coarse_cells_per_box: int | None = None,
) -> OwnedLeafAmplitudeSummary:
    """Sum raw density values on owned AMR leaf cells, one block at a time.

    This sum has **no cell-volume factor** and is not a wave mass or a
    relaxation measurement.  The run's raw FDM provenance must supply the
    HJM/wave controls; no mode is inferred from the array contents.
    """

    if (
        not isinstance(owner_rank, int)
        or not isinstance(fdm_use_hjm, bool)
        or not isinstance(fdm_first_wave_level, int)
        or fdm_first_wave_level < 1
        or maximum_array_bytes < 8
    ):
        raise ValueError("owned FDM leaf extraction parameters are invalid")
    radial = radial_centres_box is not None
    if radial:
        if (
            radial_edges_box is None
            or coarse_origin is None
            or isinstance(coarse_cells_per_box, bool)
            or not isinstance(coarse_cells_per_box, int)
            or coarse_cells_per_box < 1
            or len(radial_centres_box) != 2
        ):
            raise ValueError("FDM radial geometry must declare two centres, edges, and coarse origin")
    elif any(value is not None for value in (radial_edges_box, coarse_origin, coarse_cells_per_box)):
        raise ValueError("FDM radial geometry is incomplete")
    pair = inspect_fdm_amr_shard_pair(
        wave_path,
        amr_path,
        simple_boundary=simple_boundary,
        expected_ncpu=expected_ncpu,
        byte_order=byte_order,
    )
    wave = pair.wave
    amr = pair.amr
    if radial:
        assert radial_centres_box is not None
        assert radial_edges_box is not None
        assert coarse_origin is not None
        assert coarse_cells_per_box is not None
        if (
            len(coarse_origin) != wave.ndim
            or any(isinstance(value, bool) or not isinstance(value, int) for value in coarse_origin)
            or any(len(centre) != wave.ndim or any(not math.isfinite(x) or not 0.0 <= x < 1.0 for x in centre) for centre in radial_centres_box)
            or len(radial_edges_box) < 2
            or radial_edges_box[0] != 0.0
            or any(not math.isfinite(x) or x < 0.0 for x in radial_edges_box)
            or any(right <= left for left, right in zip(radial_edges_box, radial_edges_box[1:]))
        ):
            raise ValueError("FDM radial geometry is invalid")
    if (
        owner_rank < 1
        or owner_rank > wave.ncpu
        or not wave.path.name.endswith(f"{owner_rank:05d}")
        or not amr.path.name.endswith(f"{owner_rank:05d}")
    ):
        raise ValueError("owned FDM/AMR shard rank identity is invalid")
    leaf_counts: list[int] = []
    density_sums: list[float] = []
    radial_sums = (
        np.zeros((2, wave.nlevelmax, len(radial_edges_box) - 1), dtype=np.float64)
        if radial and radial_edges_box is not None else None
    )
    integer_dtype = np.dtype(byte_order + "i4")
    real_dtype = np.dtype(byte_order + "f8")
    with wave.path.open("rb") as wave_stream, amr.path.open("rb") as amr_stream:
        wave_reader = _RecordReader(wave_stream, byte_order=byte_order)
        amr_reader = _RecordReader(amr_stream, byte_order=byte_order)
        for _ in range(4):
            wave_reader.scalar()
        amr_stream.seek(amr.fine_payload_offset)
        for level in range(1, wave.nlevelmax + 1):
            level_count = 0
            level_density = 0.0
            for domain, ncache in enumerate(
                wave.grid_counts_by_level_and_domain[level - 1], start=1
            ):
                if wave_reader.scalar() != level or wave_reader.scalar() != ncache:
                    raise ValueError("FDM block identity changed during leaf extraction")
                if not ncache:
                    continue
                owned = domain == owner_rank
                if owned and 8 * ncache > maximum_array_bytes:
                    raise ValueError("FDM/AMR block exceeds the declared array memory bound")
                for _ in range(3):
                    amr_reader.record(4 * ncache)
                grid_centres = []
                for _ in range(amr.ndim):
                    payload = amr_reader.record(8 * ncache, read_payload=owned and radial)
                    if owned and radial:
                        centre_values = np.frombuffer(payload, dtype=real_dtype)
                        if np.any(~np.isfinite(centre_values)):
                            raise ValueError("AMR grid centres must be finite")
                        grid_centres.append(centre_values)
                for _ in range(1 + 2 * amr.ndim):
                    amr_reader.record(4 * ncache)
                leaf_masks: list[np.ndarray] = []
                for _ in range(1 << amr.ndim):
                    payload = amr_reader.record(
                        4 * ncache, read_payload=owned
                    )
                    if owned:
                        sons = np.frombuffer(payload, dtype=integer_dtype)
                        if np.any(sons < 0):
                            raise ValueError("AMR son index must be non-negative")
                        leaf_masks.append(sons == 0)
                for _ in range(2 * (1 << amr.ndim)):
                    amr_reader.record(4 * ncache)
                for child in range(1 << wave.ndim):
                    real_payload = wave_reader.record(
                        8 * ncache, read_payload=owned
                    )
                    imag_payload = wave_reader.record(
                        8 * ncache, read_payload=owned
                    )
                    if not owned:
                        continue
                    real = np.frombuffer(real_payload, dtype=real_dtype)
                    imaginary = np.frombuffer(imag_payload, dtype=real_dtype)
                    if np.any(~np.isfinite(real)) or np.any(~np.isfinite(imaginary)):
                        raise ValueError("FDM owned leaf amplitudes must be finite")
                    mask = leaf_masks[child]
                    level_count += int(np.count_nonzero(mask))
                    if fdm_use_hjm and level < fdm_first_wave_level:
                        density = np.maximum(real[mask], 0.0)
                    else:
                        density = real[mask] ** 2 + imaginary[mask] ** 2
                    level_density += float(np.sum(density, dtype=np.float64))
                    if radial and radial_sums is not None and np.any(mask):
                        assert coarse_origin is not None
                        assert coarse_cells_per_box is not None
                        assert radial_centres_box is not None
                        assert radial_edges_box is not None
                        dx = 0.5 ** level
                        selected = np.flatnonzero(mask)
                        for start in range(0, len(selected), 65536):
                            selected_chunk = selected[start : start + 65536]
                            density_chunk = density[start : start + len(selected_chunk)]
                            positions = np.stack(
                                [
                                    ((grid_centres[dimension][selected_chunk]
                                      + (((child >> dimension) & 1) - 0.5) * dx
                                      - coarse_origin[dimension]) / coarse_cells_per_box) % 1.0
                                    for dimension in range(wave.ndim)
                                ], axis=1
                            )
                            for centre_index, centre in enumerate(radial_centres_box):
                                delta = np.abs(positions - np.asarray(centre))
                                distance = np.sqrt(np.sum(np.minimum(delta, 1.0 - delta) ** 2, axis=1))
                                radial_sums[centre_index, level - 1] += np.histogram(
                                    distance, bins=radial_edges_box, weights=density_chunk
                                )[0]
            if not math.isfinite(level_density):
                raise ValueError("FDM owned leaf density sum is non-finite")
            leaf_counts.append(level_count)
            density_sums.append(level_density)
        if wave_stream.read(1) or amr_stream.read(1):
            raise ValueError("FDM/AMR shard changed during leaf extraction")
    return OwnedLeafAmplitudeSummary(
        wave_path=wave.path,
        amr_path=amr.path,
        owner_rank=owner_rank,
        ncpu=wave.ncpu,
        ndim=wave.ndim,
        boxlen_code=amr.boxlen_code,
        fdm_use_hjm=fdm_use_hjm,
        fdm_first_wave_level=fdm_first_wave_level,
        leaf_cells_by_level=tuple(leaf_counts),
        density_sum_by_level=tuple(density_sums),
        radial_centres_box=radial_centres_box,
        radial_edges_box=radial_edges_box,
        radial_density_sum_by_centre_level_bin=(
            None if radial_sums is None else tuple(
                tuple(tuple(float(value) for value in bins) for bins in levels)
                for levels in radial_sums
            )
        ),
    )
