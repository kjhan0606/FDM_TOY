from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fdm_smbh_delay.fdm_leaf_mass import (
    FDMLeafMassIdentity,
    assess_fdm_leaf_mass_series,
    check_fdm_leaf_mass_identity,
    reconstruct_fdm_aperture_centroids,
    reconstruct_fdm_radial_mass_profile,
)
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


def _series_inputs(tmp_path: Path):
    paths = tuple(tmp_path / f"fdm_outer_wave_provenance_{index:05d}.txt" for index in range(3))
    ledger = SimpleNamespace(
        source_path=tmp_path / "sample-ledger.json",
        source_sha256="a" * 64,
        samples=tuple(
            SimpleNamespace(time_code=float(index), raw_provenance_path=path)
            for index, path in enumerate(paths)
        ),
    )
    identities = tuple(
        FDMLeafMassIdentity(
            status="raw_leaf_mass_reconstruction_matches_writer",
            provenance_path=path,
            coarse_cells_per_box=1,
            owner_ranks=(1,),
            leaf_cells_by_level=(8,),
            mass_code_by_level=(mass,),
            reconstructed_mass_code=mass,
            writer_mass_code=mass,
            mass_relative_difference=0.0,
            reconstructed_leaf_cells=8,
            writer_leaf_cells=8.0,
        )
        for path, mass in zip(paths, (5.0, 5.001, 4.999), strict=True)
    )
    return ledger, identities


def test_mass_series_is_conditional_even_when_within_limit(tmp_path: Path) -> None:
    ledger, identities = _series_inputs(tmp_path)
    result = assess_fdm_leaf_mass_series(ledger, identities)
    assert result.status == "mass_series_within_limit_pending_other_conservation"
    assert result.maximum_relative_mass_drift == pytest.approx(0.0002)
    assert result.sample_ledger_sha256 == "a" * 64


def test_mass_series_censors_drift_and_unverified_snapshot(tmp_path: Path) -> None:
    ledger, identities = _series_inputs(tmp_path)
    changed = (*identities[:2], replace(identities[2], reconstructed_mass_code=4.9))
    assert assess_fdm_leaf_mass_series(ledger, changed).status == "censored_mass_series"
    unverified = (
        identities[0],
        replace(identities[1], status="censored_raw_leaf_mass_or_count_mismatch"),
        identities[2],
    )
    result = assess_fdm_leaf_mass_series(ledger, unverified)
    assert result.status == "censored_mass_series"
    assert result.maximum_relative_mass_drift is None


def test_mass_series_requires_exact_order_and_cannot_relax_limit(tmp_path: Path) -> None:
    ledger, identities = _series_inputs(tmp_path)
    with pytest.raises(ValueError, match="aligned"):
        assess_fdm_leaf_mass_series(ledger, identities[::-1])
    with pytest.raises(ValueError, match="cannot weaken"):
        assess_fdm_leaf_mass_series(
            ledger, identities, maximum_relative_mass_drift=0.01
        )


def test_two_centre_radial_mass_requires_verified_all_rank_leaf_identity(tmp_path: Path) -> None:
    summaries = _summaries(tmp_path)
    centres = ((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
    edges = (0.0, 0.25, 0.5)
    radial = (
        replace(
            summaries[0], radial_centres_box=centres, radial_edges_box=edges,
            radial_density_sum_by_centre_level_bin=(
                ((10.0, 30.0), (0.0, 0.0)),
                ((20.0, 20.0), (0.0, 0.0)),
            ),
        ),
        replace(
            summaries[1], radial_centres_box=centres, radial_edges_box=edges,
            radial_density_sum_by_centre_level_bin=(
                ((0.0, 0.0), (10.0, 30.0)),
                ((0.0, 0.0), (20.0, 20.0)),
            ),
        ),
    )
    identity = check_fdm_leaf_mass_identity(radial, _provenance(tmp_path), coarse_cells_per_box=1)
    profile = reconstruct_fdm_radial_mass_profile(radial[::-1], identity)
    assert profile.status == "radial_wave_mass_measured_pending_core_decomposition"
    assert profile.shell_mass_code_by_centre[0] == pytest.approx((1.40625, 4.21875))
    assert profile.enclosed_mass_code_by_centre[0] == pytest.approx((1.40625, 5.625))
    with pytest.raises(ValueError, match="passing leaf-mass"):
        reconstruct_fdm_radial_mass_profile(
            radial, replace(identity, status="censored_raw_leaf_mass_or_count_mismatch")
        )
    with pytest.raises(ValueError, match="summaries disagree"):
        reconstruct_fdm_radial_mass_profile(
            (radial[0], replace(radial[1], radial_edges_box=(0.0, 0.4, 0.5))), identity
        )
    with pytest.raises(ValueError, match="exceed their level"):
        inflated = replace(
            radial[0],
            radial_density_sum_by_centre_level_bin=(
                ((41.0, 0.0), (0.0, 0.0)),
                radial[0].radial_density_sum_by_centre_level_bin[1],
            ),
        )
        reconstruct_fdm_radial_mass_profile((inflated, radial[1]), identity)


def test_aperture_centroids_are_measured_but_not_certified_cores(tmp_path: Path) -> None:
    seed = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75))
    aperture = (
        replace(
            _summaries(tmp_path)[0], radial_centres_box=seed,
            radial_edges_box=(0.0, 0.2, 0.5),
            radial_density_sum_by_centre_level_bin=(
                ((10.0, 30.0), (0.0, 0.0)),
                ((10.0, 30.0), (0.0, 0.0)),
            ),
            aperture_radius_box=0.2,
            aperture_density_sum_by_centre_level=((10.0, 0.0), (10.0, 0.0)),
            aperture_first_moment_by_centre_level_dim=(
                ((0.8, 0.0, 0.0), (0.0, 0.0, 0.0)),
                ((-0.8, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ),
            aperture_second_moment_by_centre_level=((0.2, 0.0), (0.2, 0.0)),
        ),
        replace(
            _summaries(tmp_path)[1], radial_centres_box=seed,
            radial_edges_box=(0.0, 0.2, 0.5),
            radial_density_sum_by_centre_level_bin=(
                ((0.0, 0.0), (10.0, 30.0)),
                ((0.0, 0.0), (10.0, 30.0)),
            ),
            aperture_radius_box=0.2,
            aperture_density_sum_by_centre_level=((0.0, 10.0), (0.0, 10.0)),
            aperture_first_moment_by_centre_level_dim=(
                ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0)),
                ((0.0, 0.0, 0.0), (-0.8, 0.0, 0.0)),
            ),
            aperture_second_moment_by_centre_level=((0.0, 0.2), (0.0, 0.2)),
        ),
    )
    identity = check_fdm_leaf_mass_identity(
        aperture, _provenance(tmp_path, mass=5.625 / 16**3),
        coarse_cells_per_box=16,
    )
    result = reconstruct_fdm_aperture_centroids(aperture, identity)
    assert result.status == "aperture_centroid_candidates_pending_core_validation"
    assert result.centroid_candidates_box[0] == pytest.approx((0.33, 0.25, 0.25))
    assert result.centroid_candidates_box[1] == pytest.approx((0.67, 0.75, 0.75))
    assert result.rms_radius_box == pytest.approx((0.1166190379, 0.1166190379))
    assert result.separation_box is not None
    underresolved = check_fdm_leaf_mass_identity(
        aperture, _provenance(tmp_path), coarse_cells_per_box=1,
    )
    assert reconstruct_fdm_aperture_centroids(aperture, underresolved).status == "censored_aperture_centroids"
    inconsistent = replace(
        aperture[0], aperture_density_sum_by_centre_level=((9.0, 0.0), (10.0, 0.0))
    )
    with pytest.raises(ValueError, match="disagrees with radial shells"):
        reconstruct_fdm_aperture_centroids((inconsistent, aperture[1]), identity)
