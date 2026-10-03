from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math

import numpy as np
import pytest

from fdm_smbh_delay.fdm_leaf_mass import FDMApertureCentroids, FDMRadialMassProfile
from fdm_smbh_delay.fdm_soliton_profile import fit_disjoint_seed_soliton_profiles


def _measured_shell_mass(
    inner: float, outer: float, *, radius: float, central: float, background: float
) -> float:
    points = np.linspace(inner, outer, 2001)
    density = central / (1.0 + 0.091 * (points / radius) ** 2) ** 8 + background
    return float(4.0 * math.pi * np.trapezoid(points**2 * density, points))


def _sources(tmp_path: Path):
    edges = tuple(np.linspace(0.0, 0.1, 21))
    centres = ((0.25, 0.25, 0.25), (0.75, 0.25, 0.25))
    masses = tuple(
        tuple(
            _measured_shell_mass(left, right, radius=radius, central=central, background=background)
            for left, right in zip(edges[:-1], edges[1:], strict=True)
        )
        for radius, central, background in ((0.02, 1000.0, 1.0), (0.03, 500.0, 0.5))
    )
    source = tmp_path / "fdm_outer_wave_provenance_00001.txt"
    radial = FDMRadialMassProfile(
        status="radial_wave_mass_measured_pending_core_decomposition",
        provenance_path=source,
        centres_box=centres,
        edges_box=edges,
        shell_mass_code_by_centre=masses,
        enclosed_mass_code_by_centre=tuple(
            tuple(math.fsum(shells[:index + 1]) for index in range(len(shells)))
            for shells in masses
        ),
        total_wave_mass_code=100.0,
        shell_sampled_volume_code_by_centre=tuple(
            tuple(4.0 * math.pi / 3.0 * (right**3 - left**3)
                  for left, right in zip(edges[:-1], edges[1:], strict=True))
            for _ in range(2)
        ),
    )
    aperture = FDMApertureCentroids(
        status="aperture_centroid_candidates_pending_core_validation",
        provenance_path=source,
        seed_centres_box=centres,
        aperture_radius_box=0.05,
        aperture_mass_code=(1.0, 1.0),
        centroid_candidates_box=centres,
        centroid_shift_box=(0.0, 0.0),
        rms_radius_box=(0.01, 0.01),
        separation_box=0.5,
        reasons=(),
    )
    return radial, aperture


def test_two_separated_soliton_shell_profiles_recover_core_radii(tmp_path: Path) -> None:
    radial, aperture = _sources(tmp_path)
    fit = fit_disjoint_seed_soliton_profiles(
        radial, aperture, profile_c=0.091,
        seed_core_radii_box=(0.02, 0.03),
        maximum_contributing_cell_width_box=0.0025, boxlen_code=1.0,
    )
    assert fit.status == "seed_soliton_profile_candidates_pending_wave_validation"
    assert fit.fitted_core_radii_box == pytest.approx((0.02, 0.03), rel=0.01)
    assert fit.fitted_central_density_code == pytest.approx((1000.0, 500.0), rel=0.01)
    assert fit.fitted_background_density_code == pytest.approx((1.0, 0.5), rel=0.02)
    assert max(fit.relative_density_rmse) < 1.0e-5


def test_overlap_resolution_and_shift_censor_profile_fit(tmp_path: Path) -> None:
    radial, aperture = _sources(tmp_path)
    kwargs = dict(
        profile_c=0.091, seed_core_radii_box=(0.02, 0.03),
        maximum_contributing_cell_width_box=0.0025, boxlen_code=1.0,
    )
    shifted = replace(aperture, centroid_shift_box=(0.0, 0.005))
    fit = fit_disjoint_seed_soliton_profiles(radial, shifted, **kwargs)
    assert fit.status == "censored_seed_soliton_profile_fit"
    assert any("displaced" in reason for reason in fit.reasons)
    coarse = fit_disjoint_seed_soliton_profiles(
        radial, aperture, **{**kwargs, "maximum_contributing_cell_width_box": 0.01}
    )
    assert coarse.status == "censored_seed_soliton_profile_fit"
    assert any("four contributing cell widths" in reason for reason in coarse.reasons)
    overlapped = fit_disjoint_seed_soliton_profiles(
        replace(radial, edges_box=tuple(np.linspace(0.0, 0.25, 21))),
        aperture, **kwargs,
    )
    assert overlapped.status == "censored_seed_soliton_profile_fit"
    assert any("companion" in reason for reason in overlapped.reasons)


def test_mismatched_source_and_loosened_gate_are_rejected(tmp_path: Path) -> None:
    radial, aperture = _sources(tmp_path)
    kwargs = dict(
        profile_c=0.091, seed_core_radii_box=(0.02, 0.03),
        maximum_contributing_cell_width_box=0.0025, boxlen_code=1.0,
    )
    with pytest.raises(ValueError, match="source"):
        fit_disjoint_seed_soliton_profiles(
            radial, replace(aperture, provenance_path=tmp_path / "other.txt"), **kwargs
        )
    with pytest.raises(ValueError, match="parameters"):
        fit_disjoint_seed_soliton_profiles(
            radial, aperture, maximum_relative_density_rmse=0.2, **kwargs
        )
    uncovered = fit_disjoint_seed_soliton_profiles(
        replace(radial, shell_sampled_volume_code_by_centre=None), aperture, **kwargs
    )
    assert uncovered.status == "censored_seed_soliton_profile_fit"
    assert any("coverage is unavailable" in reason for reason in uncovered.reasons)
    partial = replace(
        radial,
        shell_sampled_volume_code_by_centre=(
            tuple(0.5 * value for value in radial.shell_sampled_volume_code_by_centre[0]),
            radial.shell_sampled_volume_code_by_centre[1],
        ),
    )
    censored = fit_disjoint_seed_soliton_profiles(partial, aperture, **kwargs)
    assert censored.status == "censored_seed_soliton_profile_fit"
    assert any("coverage is incomplete" in reason for reason in censored.reasons)
