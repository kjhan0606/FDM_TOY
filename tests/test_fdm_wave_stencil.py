from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fdm_smbh_delay.fdm_wave_stencil import (
    FDMSameLevelStencil,
    FDMShardLevelFields,
    FDMShardStencilResult,
    assemble_uniform_fft_base_from_shards,
    check_fdm_wave_writer_identity,
    measure_fdm_same_level_stencil,
    measure_uniform_fft_drift_quadratic,
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


def test_uniform_fft_quadratic_matches_discrete_source_eigenvalue() -> None:
    cells = np.arange(4)
    phase = 2.0 * np.pi * cells[:, None, None] / 4.0
    wave = np.broadcast_to(np.exp(1j * phase), (4, 4, 4))
    result = measure_uniform_fft_drift_quadratic(
        level=2, boxlen_code=1.0, hbar_code=1.0,
        wave_real=wave.real, wave_imag=wave.imag,
    )
    assert result.status == "uniform_fft_drift_quadratic_pending_source_binding"
    assert result.wave_mass_code == pytest.approx(1.0)
    assert result.axis_quadratic_code == pytest.approx((16.0, 0.0, 0.0))
    assert result.drift_generator_quadratic_code == pytest.approx(16.0)
    assert result.drift_generator_quadratic_code != pytest.approx(8.0)
    rotated = measure_uniform_fft_drift_quadratic(
        level=2, boxlen_code=1.0, hbar_code=1.0,
        wave_real=(wave * np.exp(1j * 0.37)).real,
        wave_imag=(wave * np.exp(1j * 0.37)).imag,
    )
    assert rotated.drift_generator_quadratic_code == pytest.approx(16.0)

    spectrum = np.fft.fftn(wave)
    frequencies = np.arange(4)
    denominator = sum(
        2.0 * (1.0 - np.cos(2.0 * np.pi * frequencies / 4.0)).reshape(
            tuple(4 if axis == dimension else 1 for axis in range(3))
        )
        for dimension in range(3)
    )
    spectral_quadratic = (
        0.5 * (1.0 / 4.0) ** 3 / (1.0 / 4.0) ** 2
        * np.sum(denominator * np.abs(spectrum) ** 2) / 64.0
    )
    assert result.drift_generator_quadratic_code == pytest.approx(spectral_quadratic)
    alternating = np.broadcast_to((-1.0) ** cells[:, None, None], (4, 4, 4))
    nyquist = measure_uniform_fft_drift_quadratic(
        level=2, boxlen_code=1.0, hbar_code=1.0,
        wave_real=alternating, wave_imag=np.zeros((4, 4, 4)),
    )
    assert nyquist.drift_generator_quadratic_code == pytest.approx(32.0)
    assert nyquist.axis_quadratic_code == pytest.approx((32.0, 0.0, 0.0))
    doubled_hbar = measure_uniform_fft_drift_quadratic(
        level=2, boxlen_code=1.0, hbar_code=2.0,
        wave_real=wave.real, wave_imag=wave.imag,
    )
    assert doubled_hbar.drift_generator_quadratic_code == pytest.approx(64.0)


def test_uniform_fft_quadratic_refuses_incomplete_or_nonfinite_fields() -> None:
    zeros = np.zeros((4, 4, 4))
    empty = measure_uniform_fft_drift_quadratic(
        level=2, boxlen_code=1.0, hbar_code=1.0,
        wave_real=zeros, wave_imag=zeros,
    )
    assert empty.status.startswith("censored_")
    with pytest.raises(ValueError, match="incomplete"):
        measure_uniform_fft_drift_quadratic(
            level=2, boxlen_code=1.0, hbar_code=1.0,
            wave_real=zeros[:-1], wave_imag=zeros,
        )
    with pytest.raises(ValueError, match="memory bound"):
        measure_uniform_fft_drift_quadratic(
            level=2, boxlen_code=1.0, hbar_code=1.0,
            wave_real=zeros, wave_imag=zeros, maximum_cells=8,
        )
    bad = zeros.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        measure_uniform_fft_drift_quadratic(
            level=2, boxlen_code=1.0, hbar_code=1.0,
            wave_real=bad, wave_imag=zeros,
        )


def _split_uniform_native_fields(tmp_path: Path) -> tuple[FDMShardLevelFields, ...]:
    source = _level_two_wave()
    fields = []
    for rank in (1, 2):
        owned = np.zeros(8, dtype=bool)
        owned[(rank - 1) * 4:rank * 4] = True
        fields.append(FDMShardLevelFields(
            wave_path=tmp_path / f"fdm_00001.out{rank:05d}",
            amr_path=tmp_path / f"amr_00001.out{rank:05d}",
            owner_rank=rank, ncpu=2, nlevelmax=2, boxlen_code=1.0,
            level=2, grid_centres=source["grid_centres"].copy(),
            wave_real=source["wave_real"].copy(),
            wave_imag=source["wave_imag"].copy(),
            son_grid_index=source["son_grid_index"].copy(),
            owned_grid=owned,
        ))
    return tuple(fields)


def test_native_owner_lattice_assembles_uniform_fft_quadratic(tmp_path: Path) -> None:
    fields = _split_uniform_native_fields(tmp_path)
    result = assemble_uniform_fft_base_from_shards(
        fields[::-1], declared_levelmin=2, hbar_code=1.0,
    )
    assert result.status == "uniform_native_fft_quadratic_pending_ledger_binding"
    assert result.assigned_cells == result.expected_cells == 64
    assert result.duplicate_cells == result.refined_owned_cells == result.missing_cells == 0
    assert result.fft_to_fine_laplacian_spacing_ratio == pytest.approx(1.0)
    assert result.quadratic is not None
    assert result.quadratic.wave_mass_code == pytest.approx(1.0)
    assert result.quadratic.drift_generator_quadratic_code == pytest.approx(16.0)
    with pytest.raises(ValueError, match="base level"):
        assemble_uniform_fft_base_from_shards(
            fields, declared_levelmin=1, hbar_code=1.0,
        )


def test_native_uniform_assembly_censors_missing_duplicate_and_refined_cells(
    tmp_path: Path,
) -> None:
    first, second = _split_uniform_native_fields(tmp_path)
    missing = assemble_uniform_fft_base_from_shards(
        (first, replace(second, owned_grid=np.zeros(8, dtype=bool))),
        declared_levelmin=2, hbar_code=1.0,
    )
    assert missing.status.startswith("censored_")
    assert missing.missing_cells == 32
    assert missing.quadratic is None
    duplicate = assemble_uniform_fft_base_from_shards(
        (replace(first, owned_grid=np.ones(8, dtype=bool)), second),
        declared_levelmin=2, hbar_code=1.0,
    )
    assert duplicate.duplicate_cells == 32
    assert duplicate.status.startswith("censored_")
    sons = first.son_grid_index.copy()
    sons[0, 0] = 1
    refined = assemble_uniform_fft_base_from_shards(
        (replace(first, son_grid_index=sons), second),
        declared_levelmin=2, hbar_code=1.0,
    )
    assert refined.refined_owned_cells == 1
    assert refined.quadratic is None
    unit_mismatch = assemble_uniform_fft_base_from_shards(
        (replace(first, boxlen_code=2.0), replace(second, boxlen_code=2.0)),
        declared_levelmin=2, hbar_code=1.0,
    )
    assert unit_mismatch.status.startswith("censored_")
    assert unit_mismatch.fft_to_fine_laplacian_spacing_ratio == pytest.approx(4.0)
    assert unit_mismatch.quadratic is None
    assert any("boxlen" in reason for reason in unit_mismatch.reasons)
    with pytest.raises(ValueError, match="every MPI rank"):
        assemble_uniform_fft_base_from_shards(
            (first,), declared_levelmin=2, hbar_code=1.0,
        )
