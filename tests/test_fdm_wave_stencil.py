from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fdm_smbh_delay.fdm_wave_stencil import (
    FDMSameLevelStencil,
    FDMShardStencilResult,
    check_fdm_wave_writer_identity,
    measure_fdm_same_level_stencil,
)


def _level_two_wave() -> dict:
    centres = np.array(list(itertools.product((0.25, 0.75), repeat=3)))
    real = np.empty((8, 8))
    imaginary = np.empty_like(real)
    for row, centre in enumerate(centres):
        for child in range(8):
            x = centre[0] + (((child & 1) != 0) - 0.5) * 0.25
            phase = 2.0 * np.pi * x
            real[row, child] = np.cos(phase)
            imaginary[row, child] = np.sin(phase)
    return dict(
        level=2, coarse_grid_shape=(1, 1, 1), boxlen_code=1.0,
        coarse_cells_per_box=1, hbar_code=1.0, grid_centres=centres,
        wave_real=real, wave_imag=imaginary,
        son_grid_index=np.zeros((8, 8), dtype=np.int64),
        owned_grid=np.ones(8, dtype=bool),
    )


def test_periodic_plane_wave_current_mass_and_phase_invariance() -> None:
    source = _level_two_wave()
    result = measure_fdm_same_level_stencil(**source)
    assert result.status == "same_level_current_complete_pending_writer_identity"
    assert result.owner_leaf_cells == result.complete_leaf_stencil_cells == 64
    assert result.leaf_mass_code == pytest.approx(1.0)
    assert result.integrated_current_code == pytest.approx((4.0, 0.0, 0.0))
    assert result.central_gradient_square_proxy_code == pytest.approx(8.0)

    angle = 0.37
    rotated = measure_fdm_same_level_stencil(
        **{**source,
           "wave_real": source["wave_real"] * np.cos(angle)
           - source["wave_imag"] * np.sin(angle),
           "wave_imag": source["wave_real"] * np.sin(angle)
           + source["wave_imag"] * np.cos(angle)}
    )
    assert rotated.leaf_mass_code == pytest.approx(result.leaf_mass_code)
    assert rotated.integrated_current_code == pytest.approx(result.integrated_current_code)
    assert rotated.central_gradient_square_proxy_code == pytest.approx(
        result.central_gradient_square_proxy_code
    )


def test_missing_or_refined_neighbour_censors_stencil() -> None:
    source = _level_two_wave()
    missing = measure_fdm_same_level_stencil(
        **{**source,
           "grid_centres": source["grid_centres"][:-1],
           "wave_real": source["wave_real"][:-1],
           "wave_imag": source["wave_imag"][:-1],
           "son_grid_index": source["son_grid_index"][:-1],
           "owned_grid": source["owned_grid"][:-1]}
    )
    assert missing.status.startswith("censored_")
    assert missing.owner_leaf_cells == 56
    assert missing.incomplete_leaf_stencil_cells > 0
    assert missing.leaf_mass_code == pytest.approx(0.875)

    sons = source["son_grid_index"].copy()
    sons[0, 0] = 1
    refined = measure_fdm_same_level_stencil(**{**source, "son_grid_index": sons})
    assert refined.status.startswith("censored_")
    assert refined.owner_leaf_cells == 63
    assert refined.refined_neighbour_stencil_cells > 0


def test_invalid_geometry_duplicate_grid_and_empty_owners_fail_closed() -> None:
    source = _level_two_wave()
    with pytest.raises(ValueError, match="geometry"):
        measure_fdm_same_level_stencil(**{**source, "coarse_grid_shape": (8, 8, 8)})
    with pytest.raises(ValueError, match="geometry"):
        measure_fdm_same_level_stencil(**{**source, "maximum_grids": 8.0})
    centres = source["grid_centres"].copy()
    centres[0] = centres[1]
    with pytest.raises(ValueError, match="repeats"):
        measure_fdm_same_level_stencil(**{**source, "grid_centres": centres})
    empty = measure_fdm_same_level_stencil(
        **{**source, "owned_grid": np.zeros(8, dtype=bool)}
    )
    assert empty.status.startswith("censored_")
    assert empty.owner_leaf_cells == 0


def _writer_identity_fixture(tmp_path: Path):
    results = []
    for rank in (1, 2):
        for level in (1, 2):
            measurement = FDMSameLevelStencil(
                status="same_level_current_complete_pending_writer_identity",
                level=level, owner_leaf_cells=1, complete_leaf_stencil_cells=1,
                incomplete_leaf_stencil_cells=0, refined_neighbour_stencil_cells=0,
                leaf_mass_code=0.25,
                integrated_current_code=(0.5 if rank == 1 else -0.25, 0.0, 0.0),
                central_gradient_square_proxy_code=0.0,
                interpretation="synthetic writer identity fixture",
            )
            results.append(FDMShardStencilResult(
                wave_path=tmp_path / f"fdm_00001.out{rank:05d}",
                amr_path=tmp_path / f"amr_00001.out{rank:05d}",
                owner_rank=rank, ncpu=2, nlevelmax=2, boxlen_code=1.0,
                measurement=measurement,
            ))
    provenance = SimpleNamespace(
        source_path=tmp_path / "fdm_outer_wave_provenance_00001.txt",
        mpi_ncpu=2, fdm_use_hjm=False, psi_snapshot_prefix="fdm_00001.out",
        leaf_cell_count=4.0, complete_current_stencil_cell_count=4.0,
        leaf_mass_code=1.0, integrated_current_code=(0.5, 0.0, 0.0),
    )
    return tuple(results), provenance


def test_all_rank_level_saved_wave_writer_identity_and_censoring(tmp_path: Path) -> None:
    results, provenance = _writer_identity_fixture(tmp_path)
    matched = check_fdm_wave_writer_identity(results[::-1], provenance)
    assert matched.status == "saved_wave_writer_current_identity_matches"
    assert matched.owner_ranks == (1, 2)
    assert matched.reconstructed_mass_code == pytest.approx(1.0)
    assert matched.reconstructed_current_code == pytest.approx((0.5, 0.0, 0.0))

    changed = list(results)
    changed[0] = replace(
        changed[0], measurement=replace(
            changed[0].measurement, integrated_current_code=(0.6, 0.0, 0.0)
        )
    )
    mismatch = check_fdm_wave_writer_identity(changed, provenance)
    assert mismatch.status.startswith("censored_")
    assert "saved-wave current differs from writer" in mismatch.reasons
    incomplete = list(results)
    incomplete[1] = replace(
        incomplete[1], measurement=replace(
            incomplete[1].measurement,
            status="censored_incomplete_or_refined_same_level_stencil",
            refined_neighbour_stencil_cells=1,
        )
    )
    assert check_fdm_wave_writer_identity(incomplete, provenance).status.startswith("censored_")


def test_writer_identity_requires_every_owner_and_pure_wave_contract(tmp_path: Path) -> None:
    results, provenance = _writer_identity_fixture(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        check_fdm_wave_writer_identity(results[:-1], provenance)
    with pytest.raises(ValueError, match="invalid"):
        check_fdm_wave_writer_identity((*results, results[0]), provenance)
    with pytest.raises(ValueError, match="HJM"):
        check_fdm_wave_writer_identity(results, SimpleNamespace(
            **{**vars(provenance), "fdm_use_hjm": True}
        ))
    wrong = list(results)
    wrong[0] = replace(wrong[0], wave_path=tmp_path / "different00001")
    with pytest.raises(ValueError, match="invalid"):
        check_fdm_wave_writer_identity(wrong, provenance)
