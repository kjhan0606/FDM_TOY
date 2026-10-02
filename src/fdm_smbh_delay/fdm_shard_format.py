"""Bounded structural reader for lagRamses ``backup_psi`` snapshot shards.

The native writer stores one Fortran sequential record for each header scalar
and each real/imaginary cell array.  This reader seeks over field payloads; it
does not interpret AMR topology or derive physical diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO


@dataclass(frozen=True)
class FDMShardStructure:
    path: Path
    ncpu: int
    ndim: int
    nlevelmax: int
    nboundary: int
    blocks_per_level: tuple[int, ...]
    grid_counts_per_level: tuple[int, ...]
    field_array_records: int


class _RecordReader:
    def __init__(self, stream: BinaryIO, *, byte_order: str) -> None:
        self.stream = stream
        self.integer = struct.Struct(byte_order + "i")

    def record(self, expected_bytes: int, *, read_payload: bool = False) -> bytes:
        marker = self.stream.read(4)
        if len(marker) != 4:
            raise ValueError("FDM shard ends before a Fortran record marker")
        length = self.integer.unpack(marker)[0]
        if length != expected_bytes or length < 0:
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
        field_records = 0
        for level in range(1, nlevelmax + 1):
            blocks = 0
            grids = 0
            for _ in range(ncpu + nboundary):
                recorded_level = reader.scalar()
                ncache = reader.scalar()
                if recorded_level != level or ncache < 0:
                    raise ValueError("FDM shard block level or grid count is invalid")
                blocks += 1
                grids += ncache
                if ncache:
                    for _ in range(2 * (1 << ndim)):
                        reader.record(8 * ncache)
                        field_records += 1
            block_counts.append(blocks)
            grid_counts.append(grids)
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
        field_array_records=field_records,
    )
