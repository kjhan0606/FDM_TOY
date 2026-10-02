from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fdm_smbh_delay.fdm_shard_format import (
    inspect_fdm_amr_shard_pair,
    inspect_fdm_shard,
    read_amr_shard_header,
)


def _record(payload: bytes, *, byte_order: str = "<") -> bytes:
    marker = struct.pack(byte_order + "i", len(payload))
    return marker + payload + marker


def _integer(value: int, *, byte_order: str = "<") -> bytes:
    return _record(struct.pack(byte_order + "i", value), byte_order=byte_order)


def _shard(*, byte_order: str = "<", nboundary: int = 1) -> bytes:
    data = b"".join(
        _integer(value, byte_order=byte_order) for value in (2, 3, 2, nboundary)
    )
    for level, counts in (
        (1, (1, 0, 2) if nboundary else (1, 0)),
        (2, (0, 1, 0) if nboundary else (0, 1)),
    ):
        for ncache in counts:
            data += _integer(level, byte_order=byte_order)
            data += _integer(ncache, byte_order=byte_order)
            if ncache:
                values = struct.pack(byte_order + "d" * ncache, *range(ncache))
                for _ in range(16):
                    data += _record(values, byte_order=byte_order)
    return data


def _amr_header(
    *,
    simple_boundary: bool,
    boundary_level1: int = 2,
    nboundary: int = 1,
    cpu_counts: tuple[int, int, int, int] = (1, 0, 0, 1),
) -> bytes:
    data = _integer(2) + _integer(3)
    data += _record(struct.pack("<3i", 8, 8, 8))
    for value in (2, 100, nboundary, 2):
        data += _integer(value)
    data += _record(struct.pack("<d", 1.0))
    data += _record(struct.pack("<3i", 0, 0, 0))
    data += _record(b"") * 2  # empty tout and aout
    data += _record(struct.pack("<d", 0.0))
    data += _record(struct.pack("<2d", 0.0, 0.0)) * 2
    data += _record(struct.pack("<2i", 0, 0))
    data += _record(struct.pack("<3d", 0.0, 0.0, 0.0))
    data += _record(struct.pack("<7d", *([0.0] * 7)))
    data += _record(struct.pack("<5d", *([0.0] * 5)))
    data += _record(struct.pack("<d", 0.0))
    data += _record(struct.pack("<4i", 0, 0, 0, 0)) * 2
    data += _record(struct.pack("<4i", *cpu_counts))
    data += _record(struct.pack("<20i", *([0] * 20)))
    if simple_boundary:
        data += _record(struct.pack("<" + "i" * (2 * nboundary), *([0] * (2 * nboundary)))) * 2
        boundary_counts = (boundary_level1, 0) if nboundary else ()
        data += _record(
            struct.pack("<" + "i" * (2 * nboundary), *boundary_counts)
        )
    data += _record(struct.pack("<5i", 0, 0, 0, 0, 0))
    data += _record(b"hilbert".ljust(128, b" "))
    data += _record(struct.pack("<2d", 0.0, 1.0))
    data += _record(struct.pack("<8i", *([0] * 8))) * 3
    return data


def _amr_fine_payload(*, boundary_level1: int = 2, nboundary: int = 1) -> bytes:
    data = b""
    for counts in (
        (1, 0, boundary_level1) if nboundary else (1, 0),
        (0, 1, 0) if nboundary else (0, 1),
    ):
        for ncache in counts:
            if not ncache:
                continue
            integers = _record(struct.pack("<" + "i" * ncache, *([0] * ncache)))
            reals = _record(struct.pack("<" + "d" * ncache, *([0.0] * ncache)))
            data += integers * 3
            data += reals * 3
            data += integers * 31
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
    assert record.grid_counts_by_level_and_domain == ((1, 0, 2), (0, 1, 0))
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


@pytest.mark.parametrize("simple_boundary", [False, True])
def test_amr_header_exposes_fortran_order_domain_counts(
    tmp_path: Path, simple_boundary: bool
) -> None:
    path = tmp_path / "amr_00001.out00001"
    path.write_bytes(_amr_header(simple_boundary=simple_boundary))
    header = read_amr_shard_header(
        path, simple_boundary=simple_boundary, expected_ncpu=2
    )
    assert header.grids_by_level_and_cpu == ((1, 0), (0, 1))
    assert header.boundary_grids_by_level == (
        ((2,), (0,)) if simple_boundary else None
    )
    assert header.fine_payload_offset == path.stat().st_size
    assert header.ordering == "hilbert"


def test_amr_header_rejects_wrong_boundary_contract(tmp_path: Path) -> None:
    path = tmp_path / "amr_00001.out00001"
    path.write_bytes(_amr_header(simple_boundary=True))
    with pytest.raises(ValueError, match="record length"):
        read_amr_shard_header(path, simple_boundary=False, expected_ncpu=2)


def test_fdm_amr_pair_checks_all_block_frames(tmp_path: Path) -> None:
    wave = tmp_path / "fdm_00001.out00001"
    amr = tmp_path / "amr_00001.out00001"
    wave.write_bytes(_shard())
    amr.write_bytes(_amr_header(simple_boundary=True) + _amr_fine_payload())
    pair = inspect_fdm_amr_shard_pair(
        wave, amr, simple_boundary=True, expected_ncpu=2
    )
    assert pair.wave.grid_counts_per_level == (3, 1)
    assert pair.amr_fine_array_records == 3 * 37
    amr.write_bytes(amr.read_bytes()[:-4])
    with pytest.raises(ValueError, match="trailer"):
        inspect_fdm_amr_shard_pair(
            wave, amr, simple_boundary=True, expected_ncpu=2
        )


def test_fdm_amr_pair_rejects_count_mismatch_and_missing_boundary_counts(
    tmp_path: Path,
) -> None:
    wave = tmp_path / "fdm_00001.out00001"
    amr = tmp_path / "amr_00001.out00001"
    wave.write_bytes(_shard())
    amr.write_bytes(_amr_header(simple_boundary=True, boundary_level1=0))
    with pytest.raises(ValueError, match="grid counts disagree"):
        inspect_fdm_amr_shard_pair(wave, amr, simple_boundary=True)
    amr.write_bytes(_amr_header(simple_boundary=False))
    with pytest.raises(ValueError, match="boundary-grid counts are unavailable"):
        inspect_fdm_amr_shard_pair(wave, amr, simple_boundary=False)
    amr.write_bytes(
        _amr_header(simple_boundary=True, cpu_counts=(0, 1, 0, 1))
    )
    with pytest.raises(ValueError, match="per-domain grid counts disagree"):
        inspect_fdm_amr_shard_pair(wave, amr, simple_boundary=True)


def test_zero_boundary_pair_works_without_boundary_header_records(
    tmp_path: Path,
) -> None:
    wave = tmp_path / "fdm_00001.out00001"
    amr = tmp_path / "amr_00001.out00001"
    wave.write_bytes(_shard(nboundary=0))
    amr.write_bytes(
        _amr_header(simple_boundary=False, nboundary=0)
        + _amr_fine_payload(nboundary=0)
    )
    pair = inspect_fdm_amr_shard_pair(wave, amr, simple_boundary=False)
    assert pair.wave.grid_counts_per_level == (1, 1)
    assert pair.amr_fine_array_records == 2 * 37
