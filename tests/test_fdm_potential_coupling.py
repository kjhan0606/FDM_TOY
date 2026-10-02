from __future__ import annotations

from pathlib import Path
import struct

import pytest

from fdm_smbh_delay.fdm_potential_coupling import (
    reconstruct_fdm_potential_coupling,
    summarize_owned_leaf_potential_coupling,
)
from test_fdm_gravity_shard import _gravity
from test_fdm_leaf_mass import _provenance
from test_fdm_shard_format import _amr_fine_payload, _amr_header, _shard


def _rank_sources(
    tmp_path: Path, *, phi: float = 1.0, particle_density: bool = False
):
    files = []
    gravity_payload = _gravity(
        particle_density=particle_density,
        density_value=100.0,
        phi_value=phi,
        force_value=0.0,
    )
    for rank in (1, 2):
        wave = tmp_path / f"fdm_00001.out{rank:05d}"
        amr = tmp_path / f"amr_00001.out{rank:05d}"
        gravity = tmp_path / f"grav_00001.out{rank:05d}"
        wave.write_bytes(_shard())
        amr.write_bytes(
            _amr_header(simple_boundary=True)
            + _amr_fine_payload(refine_first_owned=True)
        )
        gravity.write_bytes(gravity_payload)
        files.append((wave, amr, gravity))
    return files


def test_owned_leaf_potential_moment_matches_writer_mass_and_cell_volume(
    tmp_path: Path,
) -> None:
    files = _rank_sources(tmp_path, phi=2.0)
    summaries = [
        summarize_owned_leaf_potential_coupling(
            *paths, owner_rank=rank, simple_boundary=True,
            fdm_use_hjm=False, fdm_first_wave_level=1,
            expected_ncpu=2, maximum_array_bytes=8,
        )
        for rank, paths in enumerate(files, start=1)
    ]
    assert summaries[0].amplitude.leaf_cells_by_level == (7, 0)
    assert summaries[0].density_times_potential_sum_by_level == pytest.approx((70.0, 0.0))
    assert summaries[1].density_times_potential_sum_by_level == pytest.approx((0.0, 80.0))
    result = reconstruct_fdm_potential_coupling(
        summaries[::-1], _provenance(tmp_path, mass=5.0, count=15.0),
        coarse_cells_per_box=1,
    )
    assert result.status == "potential_coupling_measured_pending_full_hamiltonian"
    assert result.density_times_potential_code_by_level == pytest.approx((8.75, 1.25))
    assert result.integrated_density_times_potential_code == pytest.approx(10.0)
    assert "not self-energy or Hamiltonian" in result.interpretation


def test_writer_mass_mismatch_censors_potential_moment(tmp_path: Path) -> None:
    summaries = [
        summarize_owned_leaf_potential_coupling(
            *paths, owner_rank=rank, simple_boundary=True,
            fdm_use_hjm=False, fdm_first_wave_level=1,
            expected_ncpu=2,
        )
        for rank, paths in enumerate(_rank_sources(tmp_path), start=1)
    ]
    result = reconstruct_fdm_potential_coupling(
        summaries, _provenance(tmp_path, mass=6.0, count=15.0),
        coarse_cells_per_box=1,
    )
    assert result.status == "censored_potential_coupling_mass_identity"
    assert result.integrated_density_times_potential_code is None


def test_hjm_coarse_density_is_not_squared(tmp_path: Path) -> None:
    summaries = [
        summarize_owned_leaf_potential_coupling(
            *paths, owner_rank=rank, simple_boundary=True,
            fdm_use_hjm=True, fdm_first_wave_level=2,
            expected_ncpu=2,
        )
        for rank, paths in enumerate(_rank_sources(tmp_path, phi=2.0), start=1)
    ]
    provenance = _provenance(tmp_path, mass=1.5, count=15.0)
    provenance.fdm_use_hjm = True
    provenance.fdm_first_wave_level = 2
    result = reconstruct_fdm_potential_coupling(
        summaries, provenance, coarse_cells_per_box=1
    )
    assert result.status == "potential_coupling_measured_pending_full_hamiltonian"
    assert result.integrated_density_times_potential_code == pytest.approx(3.0)


def test_explicit_particle_density_record_is_not_mistaken_for_phi(tmp_path: Path) -> None:
    summaries = [
        summarize_owned_leaf_potential_coupling(
            *paths, owner_rank=rank, simple_boundary=True,
            fdm_use_hjm=False, fdm_first_wave_level=1,
            particle_density_included=True,
            expected_ncpu=2,
        )
        for rank, paths in enumerate(
            _rank_sources(tmp_path, phi=2.0, particle_density=True), start=1
        )
    ]
    result = reconstruct_fdm_potential_coupling(
        summaries, _provenance(tmp_path, mass=5.0, count=15.0),
        coarse_cells_per_box=1,
    )
    assert result.integrated_density_times_potential_code == pytest.approx(10.0)


def test_constant_potential_shift_changes_moment_by_wave_mass(tmp_path: Path) -> None:
    values = []
    for phi in (1.0, 2.0):
        summaries = [
            summarize_owned_leaf_potential_coupling(
                *paths, owner_rank=rank, simple_boundary=True,
                fdm_use_hjm=False, fdm_first_wave_level=1,
                expected_ncpu=2,
            )
            for rank, paths in enumerate(_rank_sources(tmp_path, phi=phi), start=1)
        ]
        result = reconstruct_fdm_potential_coupling(
            summaries, _provenance(tmp_path, mass=5.0, count=15.0),
            coarse_cells_per_box=1,
        )
        values.append(result.integrated_density_times_potential_code)
    assert values[1] - values[0] == pytest.approx(5.0)


def test_nonfinite_potential_and_wrong_rank_fail_closed(tmp_path: Path) -> None:
    files = _rank_sources(tmp_path)
    wave, amr, gravity = files[0]
    gravity.write_bytes(
        _gravity().replace(struct.pack("<d", 1.0), struct.pack("<d", float("nan")), 1)
    )
    with pytest.raises(ValueError, match="non-finite"):
        summarize_owned_leaf_potential_coupling(
            wave, amr, gravity, owner_rank=1,
            simple_boundary=True, fdm_use_hjm=False,
            fdm_first_wave_level=1,
        )
    gravity.write_bytes(_gravity())
    with pytest.raises(ValueError, match="owner rank"):
        summarize_owned_leaf_potential_coupling(
            wave, amr, gravity, owner_rank=2,
            simple_boundary=True, fdm_use_hjm=False,
            fdm_first_wave_level=1,
        )
