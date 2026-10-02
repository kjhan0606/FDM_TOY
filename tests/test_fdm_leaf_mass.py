from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fdm_smbh_delay.fdm_leaf_mass import check_fdm_leaf_mass_identity
from fdm_smbh_delay.fdm_shard_format import OwnedLeafAmplitudeSummary


def _summaries(tmp_path: Path) -> tuple[OwnedLeafAmplitudeSummary, ...]:
    return (
        OwnedLeafAmplitudeSummary(
            wave_path=tmp_path / "fdm_00001.out00001",
            amr_path=tmp_path / "amr_00001.out00001",
            owner_rank=1,
            ncpu=2,
            ndim=3,
            boxlen_code=1.0,
            fdm_use_hjm=False,
            fdm_first_wave_level=1,
            leaf_cells_by_level=(8, 0),
            density_sum_by_level=(40.0, 0.0),
        ),
        OwnedLeafAmplitudeSummary(
            wave_path=tmp_path / "fdm_00001.out00002",
            amr_path=tmp_path / "amr_00001.out00002",
            owner_rank=2,
            ncpu=2,
            ndim=3,
            boxlen_code=1.0,
            fdm_use_hjm=False,
            fdm_first_wave_level=1,
            leaf_cells_by_level=(0, 8),
            density_sum_by_level=(0.0, 40.0),
        ),
    )


def _provenance(tmp_path: Path, *, mass: float = 5.625, count: float = 16.0):
    return SimpleNamespace(
        source_path=tmp_path / "fdm_outer_wave_provenance_00001.txt",
        source_schema_version=5,
        mpi_ncpu=2,
        psi_snapshot_prefix="fdm_00001.out",
        fdm_use_hjm=False,
        fdm_first_wave_level=1,
        leaf_mass_code=mass,
        leaf_cell_count=count,
    )


def test_writer_cell_volume_law_and_all_rank_reconstruction(tmp_path: Path) -> None:
    result = check_fdm_leaf_mass_identity(
        _summaries(tmp_path)[::-1],
        _provenance(tmp_path),
        coarse_cells_per_box=1,
    )
    assert result.status == "raw_leaf_mass_reconstruction_matches_writer"
    assert result.owner_ranks == (1, 2)
    assert result.leaf_cells_by_level == (8, 8)
    assert result.mass_code_by_level == pytest.approx((5.0, 0.625))
    assert result.reconstructed_mass_code == pytest.approx(5.625)
    assert result.mass_relative_difference == 0.0


def test_incomplete_or_wrong_source_set_fails_closed(tmp_path: Path) -> None:
    first, second = _summaries(tmp_path)
    with pytest.raises(ValueError, match="every MPI owner"):
        check_fdm_leaf_mass_identity(
            (first,), _provenance(tmp_path), coarse_cells_per_box=1
        )
    with pytest.raises(ValueError, match="every MPI owner"):
        check_fdm_leaf_mass_identity(
            (first, first), _provenance(tmp_path), coarse_cells_per_box=1
        )
    with pytest.raises(ValueError, match="raw output identity"):
        check_fdm_leaf_mass_identity(
            (first, replace(second, wave_path=tmp_path / "wrong00002")),
            _provenance(tmp_path),
            coarse_cells_per_box=1,
        )


def test_mass_or_leaf_count_mismatch_is_censored(tmp_path: Path) -> None:
    sources = _summaries(tmp_path)
    wrong_geometry = check_fdm_leaf_mass_identity(
        sources, _provenance(tmp_path), coarse_cells_per_box=2
    )
    assert wrong_geometry.status == "censored_raw_leaf_mass_or_count_mismatch"
    wrong_count = check_fdm_leaf_mass_identity(
        sources, _provenance(tmp_path, count=15), coarse_cells_per_box=1
    )
    assert wrong_count.status == "censored_raw_leaf_mass_or_count_mismatch"


def test_hjm_controls_and_invalid_scale_are_rejected(tmp_path: Path) -> None:
    sources = _summaries(tmp_path)
    provenance = _provenance(tmp_path)
    provenance.fdm_use_hjm = True
    with pytest.raises(ValueError, match="raw output identity"):
        check_fdm_leaf_mass_identity(sources, provenance, coarse_cells_per_box=1)
    with pytest.raises(ValueError, match="parameters are invalid"):
        check_fdm_leaf_mass_identity(sources, provenance, coarse_cells_per_box=0)
    provenance.mpi_ncpu = 0
    with pytest.raises(ValueError, match="MPI rank count is invalid"):
        check_fdm_leaf_mass_identity(sources, provenance, coarse_cells_per_box=1)
