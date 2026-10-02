"""Conditional two-core profile fits to source-bound FDM radial mass shells.

The fit tests the lagRamses seed profile shape.  A fit to seed-centred total
wave density is not, by itself, an isolated or dynamically relaxed soliton.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .fdm_leaf_mass import FDMApertureCentroids, FDMRadialMassProfile


_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(32)


@dataclass(frozen=True)
class FDMSolitonProfileCandidates:
    status: str
    provenance_path: Path
    profile_c: float
    seed_core_radii_box: tuple[float, float]
    fitted_core_radii_box: tuple[float | None, float | None]
    fitted_central_density_code: tuple[float | None, float | None]
    fitted_background_density_code: tuple[float | None, float | None]
    relative_density_rmse: tuple[float | None, float | None]
    reasons: tuple[str, ...]


def _shell_average_shape(edges: np.ndarray, core_radius: float, profile_c: float) -> np.ndarray:
    left = edges[:-1, None]
    right = edges[1:, None]
    radii = (right + left) / 2.0 + (right - left) / 2.0 * _NODES
    integral = np.sum(
        _WEIGHTS * radii**2 / (1.0 + profile_c * (radii / core_radius) ** 2) ** 8,
        axis=1,
    ) * (edges[1:] - edges[:-1]) / 2.0
    return 3.0 * integral / (edges[1:] ** 3 - edges[:-1] ** 3)


def fit_disjoint_seed_soliton_profiles(
    radial: FDMRadialMassProfile,
    apertures: FDMApertureCentroids,
    *,
    profile_c: float,
    seed_core_radii_box: tuple[float, float],
    maximum_contributing_cell_width_box: float,
    boxlen_code: float,
    maximum_relative_density_rmse: float = 0.1,
) -> FDMSolitonProfileCandidates:
    """Fit seed-profile radii only inside a deliberately narrow validity gate."""

    if (
        radial.status != "radial_wave_mass_measured_pending_core_decomposition"
        or radial.provenance_path != apertures.provenance_path
        or radial.centres_box != apertures.seed_centres_box
    ):
        raise ValueError("FDM soliton fit inputs do not share source and seed geometry")
    scalars = (profile_c, maximum_contributing_cell_width_box, boxlen_code, maximum_relative_density_rmse)
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in scalars)
        or any(not math.isfinite(value) or value <= 0.0 for value in seed_core_radii_box)
        or len(seed_core_radii_box) != 2
        or not 0.0 < maximum_relative_density_rmse <= 0.1
    ):
        raise ValueError("FDM soliton profile parameters are invalid")
    edges = np.asarray(radial.edges_box, dtype=np.float64)
    if len(edges) < 9 or edges[0] != 0.0 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("FDM soliton fit requires eight or more valid radial shells")
    if any(len(shells) != len(edges) - 1 for shells in radial.shell_mass_code_by_centre):
        raise ValueError("FDM radial shell count is invalid")
    if len(radial.shell_mass_code_by_centre) != 2:
        raise ValueError("FDM soliton fit requires two radial profiles")
    separation = math.sqrt(sum(
        min(abs(a - b), 1.0 - abs(a - b)) ** 2
        for a, b in zip(*radial.centres_box, strict=True)
    ))
    reasons: list[str] = []
    if apertures.status != "aperture_centroid_candidates_pending_core_validation":
        reasons.append("aperture centroids are censored or unavailable")
    if edges[-1] >= 0.45 * separation:
        reasons.append("radial shells approach the companion core")
    if any(radius >= separation / 8.0 for radius in seed_core_radii_box):
        reasons.append("seed cores are not sufficiently separated")
    if maximum_contributing_cell_width_box > min(seed_core_radii_box) / 4.0:
        reasons.append("seed core radius spans fewer than four contributing cell widths")
    if edges[1] > min(seed_core_radii_box) / 2.0:
        reasons.append("innermost shell does not resolve the seed core")
    if edges[-1] < 2.5 * max(seed_core_radii_box):
        reasons.append("radial coverage does not extend beyond the seed cores")
    fitted_radii: list[float | None] = []
    fitted_density: list[float | None] = []
    fitted_background: list[float | None] = []
    residuals: list[float | None] = []
    shell_volumes = 4.0 * math.pi / 3.0 * np.diff(edges**3) * boxlen_code**3
    for index, seed_radius in enumerate(seed_core_radii_box):
        local_reasons = []
        if radial.shell_sampled_volume_code_by_centre is None:
            local_reasons.append("AMR leaf-volume coverage is unavailable")
            sampled_volumes = shell_volumes
        else:
            if (
                len(radial.shell_sampled_volume_code_by_centre) != 2
                or len(radial.shell_sampled_volume_code_by_centre[index]) != len(shell_volumes)
            ):
                raise ValueError("FDM shell sampled-volume geometry is invalid")
            sampled_volumes = np.asarray(
                radial.shell_sampled_volume_code_by_centre[index], dtype=np.float64
            )
            if np.any(~np.isfinite(sampled_volumes)) or np.any(sampled_volumes < 0.0):
                raise ValueError("FDM shell sampled volumes are invalid")
            coverage = sampled_volumes / shell_volumes
            if np.any(coverage < 0.8) or np.any(coverage > 1.2):
                local_reasons.append("AMR shell-volume coverage is incomplete or overfilled")
            if sampled_volumes[0] < 16.0 * (maximum_contributing_cell_width_box * boxlen_code) ** 3:
                local_reasons.append("innermost shell has fewer than sixteen effective leaf cells")
        shift = apertures.centroid_shift_box[index]
        if shift is None or shift > 0.25 * min(seed_radius, edges[1]):
            local_reasons.append("seed-centred profile is displaced from aperture centroid")
        masses = np.asarray(radial.shell_mass_code_by_centre[index], dtype=np.float64)
        if np.any(~np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("FDM radial shell masses are invalid")
        density = masses / np.where(sampled_volumes > 0.0, sampled_volumes, shell_volumes)
        if density[0] <= 0.0 or density[0] < 3.0 * density[-1]:
            local_reasons.append("core-to-background density contrast is insufficient")
        if reasons or local_reasons:
            reasons.extend(f"core {index + 1}: {reason}" for reason in local_reasons)
            fitted_radii.append(None)
            fitted_density.append(None)
            fitted_background.append(None)
            residuals.append(None)
            continue
        best: tuple[float, float, float, float, int] | None = None
        candidates = np.geomspace(0.5 * seed_radius, 2.0 * seed_radius, 161)
        for candidate_index, candidate_radius in enumerate(candidates):
            shape = _shell_average_shape(edges, float(candidate_radius), profile_c)
            matrix = np.column_stack((shape, np.ones_like(shape)))
            amplitude, background = np.linalg.lstsq(matrix, density, rcond=None)[0]
            if amplitude <= 0.0:
                continue
            if background < 0.0:
                background = 0.0
                amplitude = float(np.dot(shape, density) / np.dot(shape, shape))
            model = amplitude * shape + background
            rmse = float(np.sqrt(np.mean((model - density) ** 2)) / density[0])
            if best is None or rmse < best[0]:
                best = (rmse, float(candidate_radius), float(amplitude), float(background), candidate_index)
        if best is None:
            reasons.append(f"core {index + 1}: no positive soliton fit exists")
            fitted_radii.append(None)
            fitted_density.append(None)
            fitted_background.append(None)
            residuals.append(None)
            continue
        rmse, radius, amplitude, background, candidate_index = best
        fitted_radii.append(radius)
        fitted_density.append(amplitude)
        fitted_background.append(background)
        residuals.append(rmse)
        if candidate_index in {0, len(candidates) - 1}:
            reasons.append(f"core {index + 1}: fitted radius reached the declared search boundary")
        if rmse > maximum_relative_density_rmse:
            reasons.append(f"core {index + 1}: soliton profile residual exceeds the limit")
        if background > 0.5 * amplitude:
            reasons.append(f"core {index + 1}: background density dominates the fit")
    return FDMSolitonProfileCandidates(
        status=(
            "seed_soliton_profile_candidates_pending_wave_validation"
            if not reasons else "censored_seed_soliton_profile_fit"
        ),
        provenance_path=radial.provenance_path,
        profile_c=profile_c,
        seed_core_radii_box=seed_core_radii_box,
        fitted_core_radii_box=tuple(fitted_radii),
        fitted_central_density_code=tuple(fitted_density),
        fitted_background_density_code=tuple(fitted_background),
        relative_density_rmse=tuple(residuals),
        reasons=tuple(reasons),
    )
