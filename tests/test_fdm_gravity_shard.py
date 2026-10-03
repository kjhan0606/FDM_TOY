from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fdm_smbh_delay.fdm_gravity_shard import (
    inspect_fdm_gravity_shard_pair,
    inspect_gravity_shard,
    read_valid_poisson_potential_marker,
)
from test_fdm_shard_format import _shard


def _record(payload: bytes) -> bytes:
    marker = struct.pack("<i", len(payload))
    return marker + payload + marker


def _integer(value: int) -> bytes:
    return _record(struct.pack("<i", value))


def _gravity(
    *, particle_density: bool = False, wrong_level_two_count: bool = False,
    density_value: float = 1.0, phi_value: float = 1.0, force_value: float = 1.0,
) -> bytes:
    nvar = 5 if particle_density else 4
    data = b"".join(_integer(value) for value in (2, nvar, 2, 1))
    for level, counts in ((1, (1, 0, 2)), (2, (0, 1, 0))):
        for domain, original_count in enumerate(counts):
            ncache = (
                2 if wrong_level_two_count and level == 2 and domain == 1
                else original_count
            )
            data += _integer(level) + _integer(ncache)
            if ncache:
                def array(value: float) -> bytes:
                    return _record(struct.pack("<" + "d" * ncache, *([value] * ncache)))

                for _ in range(8):
                    if particle_density:
                        data += array(density_value)
                    data += array(phi_value)
                    data += array(force_value) * 3
    return data


def test_gravity_frames_and_fdm_domain_counts_are_bound(tmp_path: Path) -> None:
    wave = tmp_path / "fdm_00001.out00001"
    gravity = tmp_path / "grav_00001.out00001"
    wave.write_bytes(_shard())
    gravity.write_bytes(_gravity())
    pair = inspect_fdm_gravity_shard_pair(wave, gravity, expected_ncpu=2)
    assert pair.gravity.grid_counts_by_level_and_domain == ((1, 0, 2), (0, 1, 0))
    assert pair.gravity.field_array_records == 96
    assert pair.gravity.particle_density_included is False


def test_extra_density_layout_requires_explicit_declaration(tmp_path: Path) -> None:
    wave = tmp_path / "fdm_00001.out00001"
    gravity = tmp_path / "grav_00001.out00001"
    wave.write_bytes(_shard())
    gravity.write_bytes(_gravity(particle_density=True))
    with pytest.raises(ValueError, match="declared writer layout"):
        inspect_fdm_gravity_shard_pair(wave, gravity)
    pair = inspect_fdm_gravity_shard_pair(
        wave, gravity, particle_density_included=True
    )
    assert pair.gravity.field_array_records == 120


def test_mismatch_truncation_and_trailing_bytes_fail_closed(tmp_path: Path) -> None:
    wave = tmp_path / "fdm_00001.out00001"
    gravity = tmp_path / "grav_00001.out00001"
    wave.write_bytes(_shard())
    gravity.write_bytes(_gravity(wrong_level_two_count=True))
    with pytest.raises(ValueError, match="grid layouts disagree"):
        inspect_fdm_gravity_shard_pair(wave, gravity)
    gravity.write_bytes(_gravity()[:-4])
    with pytest.raises(ValueError, match="trailer"):
        inspect_gravity_shard(gravity, ndim=3)
    gravity.write_bytes(_gravity() + b"x")
    with pytest.raises(ValueError, match="trailing bytes"):
        inspect_gravity_shard(gravity, ndim=3)


def test_phi_marker_requires_matching_step_level_and_time(tmp_path: Path) -> None:
    marker = tmp_path / "POISSON_PHI_VALID"
    marker.write_text("LAGRAMSES_POISSON_PHI_VALID_V1\n 7 2 1.25D+00 5.0D-01\n")
    record = read_valid_poisson_potential_marker(
        marker,
        expected_nstep_coarse=7,
        expected_nlevelmax=2,
        expected_time_code=1.25,
        expected_aexp=0.5,
    )
    assert record.nstep_coarse == 7
    with pytest.raises(ValueError, match="differs"):
        read_valid_poisson_potential_marker(
            marker,
            expected_nstep_coarse=8,
            expected_nlevelmax=2,
            expected_time_code=1.25,
            expected_aexp=0.5,
        )
    marker.write_text("LAGRAMSES_POISSON_PHI_VALID_V1\n 7 2 nan 5.0D-01\n")
    with pytest.raises(ValueError, match="differs"):
        read_valid_poisson_potential_marker(
            marker,
            expected_nstep_coarse=7,
            expected_nlevelmax=2,
            expected_time_code=1.25,
            expected_aexp=0.5,
        )
