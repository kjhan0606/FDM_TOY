from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fdm_smbh_delay.fdm_shard_format import inspect_fdm_shard


def _record(payload: bytes, *, byte_order: str = "<") -> bytes:
    marker = struct.pack(byte_order + "i", len(payload))
    return marker + payload + marker


def _integer(value: int, *, byte_order: str = "<") -> bytes:
    return _record(struct.pack(byte_order + "i", value), byte_order=byte_order)


def _shard(*, byte_order: str = "<") -> bytes:
    data = b"".join(_integer(value, byte_order=byte_order) for value in (2, 3, 2, 1))
    for level, counts in ((1, (1, 0, 2)), (2, (0, 1, 0))):
        for ncache in counts:
            data += _integer(level, byte_order=byte_order)
            data += _integer(ncache, byte_order=byte_order)
            if ncache:
                values = struct.pack(byte_order + "d" * ncache, *range(ncache))
                for _ in range(16):
                    data += _record(values, byte_order=byte_order)
    return data


def test_inspect_native_fdm_shard_without_loading_field_arrays(tmp_path: Path) -> None:
    path = tmp_path / "fdm_00001.out00001"
    path.write_bytes(_shard())
    record = inspect_fdm_shard(path, expected_ncpu=2)
    assert record.ndim == 3
    assert record.nlevelmax == 2
    assert record.nboundary == 1
    assert record.blocks_per_level == (3, 3)
    assert record.grid_counts_per_level == (3, 1)
    assert record.field_array_records == 48


def test_big_endian_must_be_declared(tmp_path: Path) -> None:
    path = tmp_path / "fdm_00001.out00001"
    path.write_bytes(_shard(byte_order=">"))
    with pytest.raises(ValueError, match="record length"):
        inspect_fdm_shard(path)
    assert inspect_fdm_shard(path, byte_order=">").grid_counts_per_level == (3, 1)


def test_wrong_rank_count_and_corrupt_trailer_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "fdm_00001.out00001"
    payload = _shard()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="topology"):
        inspect_fdm_shard(path, expected_ncpu=3)
    path.write_bytes(payload[:-4] + b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="trailer"):
        inspect_fdm_shard(path)


def test_truncation_trailing_bytes_and_impossible_header_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fdm_00001.out00001"
    payload = _shard()
    path.write_bytes(payload[:-3])
    with pytest.raises(ValueError, match="trailer"):
        inspect_fdm_shard(path)
    path.write_bytes(payload + b"x")
    with pytest.raises(ValueError, match="trailing bytes"):
        inspect_fdm_shard(path)
    path.write_bytes(_integer(2) + _integer(3) + _integer(10_000_000) + _integer(1))
    with pytest.raises(ValueError, match="more blocks"):
        inspect_fdm_shard(path)
