"""Code-unit leaf-mass identity check for a complete lagRamses FDM output.

This checks a snapshot reader against the solver's own compact mass ledger.
Agreement is not a time-series conservation, relaxation, or convergence pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from .fdm_shard_format import OwnedLeafAmplitudeSummary
from .dual_soliton_relaxation import (
    DualSolitonRelaxationSampleLedger,
    RelaxationConservationThresholds,
)
from .lagramses_fdm_provenance import LagRamsesFDMOuterWaveProvenance


@dataclass(frozen=True)
class FDMLeafMassIdentity:
    status: str
    provenance_path: Path
    coarse_cells_per_box: int
    owner_ranks: tuple[int, ...]
    leaf_cells_by_level: tuple[int, ...]
    mass_code_by_level: tuple[float, ...]
    reconstructed_mass_code: float
    writer_mass_code: float
    mass_relative_difference: float
    reconstructed_leaf_cells: int
    writer_leaf_cells: float


@dataclass(frozen=True)
class FDMLeafMassSeries:
    status: str
    sample_ledger_path: Path
    sample_ledger_sha256: str
    sample_times_code: tuple[float, ...]
    raw_provenance_paths: tuple[Path, ...]
    mass_code: tuple[float, ...]
    relative_mass_drift_from_initial: tuple[float, ...] | None
    maximum_relative_mass_drift: float | None
    declared_mass_drift_limit: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FDMRadialMassProfile:
    """Two-centre total-wave mass shells, not a two-component core decomposition."""

    status: str
    provenance_path: Path
    centres_box: tuple[tuple[float, ...], ...]
    edges_box: tuple[float, ...]
    shell_mass_code_by_centre: tuple[tuple[float, ...], ...]
    enclosed_mass_code_by_centre: tuple[tuple[float, ...], ...]
    total_wave_mass_code: float


@dataclass(frozen=True)
class FDMApertureCentroids:
    """Mass centroids of two disjoint seed apertures, not fitted soliton peaks."""

    status: str
    provenance_path: Path
    seed_centres_box: tuple[tuple[float, ...], ...]
    aperture_radius_box: float
    aperture_mass_code: tuple[float, float]
    centroid_candidates_box: tuple[tuple[float, ...] | None, ...]
    centroid_shift_box: tuple[float | None, float | None]
    rms_radius_box: tuple[float | None, float | None]
    separation_box: float | None
    reasons: tuple[str, ...]


def reconstruct_fdm_aperture_centroids(
    summaries: Sequence[OwnedLeafAmplitudeSummary],
    identity: FDMLeafMassIdentity,
) -> FDMApertureCentroids:
    """Reconstruct moving *candidates* from source-bound disjoint apertures.

    A density-weighted aperture centroid can follow a core but is not a
    soliton-centre measurement without background subtraction, a profile fit,
    and temporal continuity checks.
    """

    radial = reconstruct_fdm_radial_mass_profile(summaries, identity)
    ordered = tuple(sorted(summaries, key=lambda item: item.owner_rank))
    radius = ordered[0].aperture_radius_box
    if radius is None or not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("FDM aperture radius is missing")
    seed_distance = math.sqrt(sum(
        min(abs(a - b), 1.0 - abs(a - b)) ** 2
        for a, b in zip(*radial.centres_box, strict=True)
    ))
    if radius >= 0.5 * seed_distance:
        raise ValueError("FDM aperture regions must be disjoint")
    nlevelmax = len(identity.mass_code_by_level)
    ndim = ordered[0].ndim
    for item in ordered:
        mass = item.aperture_density_sum_by_centre_level
        first = item.aperture_first_moment_by_centre_level_dim
        second = item.aperture_second_moment_by_centre_level
        if (
            item.aperture_radius_box != radius
            or mass is None or first is None or second is None
            or len(mass) != 2 or len(first) != 2 or len(second) != 2
            or any(len(levels) != nlevelmax for levels in (*mass, *first, *second))
            or any(len(dimensions) != ndim for levels in first for dimensions in levels)
            or any(not math.isfinite(value) or value < 0.0 for levels in mass for value in levels)
            or any(not math.isfinite(value) for levels in first for dimensions in levels for value in dimensions)
            or any(not math.isfinite(value) or value < 0.0 for levels in second for value in levels)
        ):
            raise ValueError("FDM aperture moments disagree across owners")
    masses = []
    positions = []
    shifts = []
    rms = []
    reasons = []
    for centre_index, seed in enumerate(radial.centres_box):
        level_masses = []
        level_first = []
        level_second = []
        widest_contributing_cell = 0.0
        for level in range(1, nlevelmax + 1):
            dx = 0.5**level * ordered[0].boxlen_code / identity.coarse_cells_per_box
            volume = dx**ndim
            density_sum = math.fsum(
                item.aperture_density_sum_by_centre_level[centre_index][level - 1]  # type: ignore[index]
                for item in ordered
            )
            level_masses.append(density_sum * volume)
            if density_sum > 0.0:
                widest_contributing_cell = max(widest_contributing_cell, 0.5**level / identity.coarse_cells_per_box)
            level_first.append(tuple(
                math.fsum(
                    item.aperture_first_moment_by_centre_level_dim[centre_index][level - 1][dimension]  # type: ignore[index]
                    for item in ordered
                ) * volume
                for dimension in range(ndim)
            ))
            level_second.append(math.fsum(
                item.aperture_second_moment_by_centre_level[centre_index][level - 1]  # type: ignore[index]
                for item in ordered
            ) * volume)
            if density_sum > math.fsum(item.density_sum_by_level[level - 1] for item in ordered) * (1.0 + 1.0e-10):
                raise ValueError("FDM aperture density exceeds verified level density")
            if radius in radial.edges_box and radius != radial.edges_box[-1]:
                edge_index = radial.edges_box.index(radius)
                shell_density = math.fsum(
                    item.radial_density_sum_by_centre_level_bin[centre_index][level - 1][bin_index]  # type: ignore[index]
                    for item in ordered for bin_index in range(edge_index)
                )
                if not math.isclose(density_sum, shell_density, rel_tol=1.0e-10, abs_tol=1.0e-12):
                    raise ValueError("FDM aperture mass disagrees with radial shells")
            if level_second[-1] > level_masses[-1] * radius**2 * (1.0 + 1.0e-10):
                raise ValueError("FDM aperture second moment exceeds its radius")
            if math.sqrt(sum(value**2 for value in level_first[-1])) > level_masses[-1] * radius * (1.0 + 1.0e-10):
                raise ValueError("FDM aperture first moment exceeds its radius")
        mass = math.fsum(level_masses)
        masses.append(mass)
        if not math.isfinite(mass) or mass > identity.reconstructed_mass_code * (1.0 + 1.0e-10):
            raise ValueError("FDM aperture mass exceeds verified wave mass")
        if mass <= 0.0:
            positions.append(None)
            shifts.append(None)
            rms.append(None)
            reasons.append(f"aperture {centre_index + 1} has no positive wave mass")
            continue
        mean_shift = tuple(
            math.fsum(level[dimension] for level in level_first) / mass
            for dimension in range(ndim)
        )
        shift = math.sqrt(sum(value**2 for value in mean_shift))
        variance = math.fsum(level_second) / mass - shift**2
        if not math.isfinite(shift) or not math.isfinite(variance) or variance < -1.0e-12 * radius**2:
            raise ValueError("FDM aperture centroid moments are inconsistent")
        positions.append(tuple((coordinate + offset) % 1.0 for coordinate, offset in zip(seed, mean_shift, strict=True)))
        shifts.append(shift)
        rms.append(math.sqrt(max(variance, 0.0)))
        if shift > 0.5 * radius:
            reasons.append(f"aperture {centre_index + 1} centroid is near its search edge")
        if radius < 2.0 * widest_contributing_cell:
            reasons.append(f"aperture {centre_index + 1} spans fewer than two contributing cell widths")
    separation = None
    if positions[0] is not None and positions[1] is not None:
        separation = math.sqrt(sum(
            min(abs(a - b), 1.0 - abs(a - b)) ** 2
            for a, b in zip(positions[0], positions[1], strict=True)
        ))
    return FDMApertureCentroids(
        status=(
            "aperture_centroid_candidates_pending_core_validation"
            if not reasons else "censored_aperture_centroids"
        ),
        provenance_path=identity.provenance_path,
        seed_centres_box=radial.centres_box,
        aperture_radius_box=radius,
        aperture_mass_code=tuple(masses),
        centroid_candidates_box=tuple(positions),
        centroid_shift_box=tuple(shifts),
        rms_radius_box=tuple(rms),
        separation_box=separation,
        reasons=tuple(reasons),
    )


def reconstruct_fdm_radial_mass_profile(
    summaries: Sequence[OwnedLeafAmplitudeSummary],
    identity: FDMLeafMassIdentity,
) -> FDMRadialMassProfile:
    """Combine all owner leaf shells with the same cell volumes as the writer.

    The two centred profiles each include *all* wave mass in their shells;
    overlapping shells must never be added as if they were distinct cores.
    """

    if identity.status != "raw_leaf_mass_reconstruction_matches_writer":
        raise ValueError("FDM radial profile requires a passing leaf-mass identity")
    by_rank = {item.owner_rank: item for item in summaries}
    if len(by_rank) != len(summaries) or set(by_rank) != set(identity.owner_ranks):
        raise ValueError("FDM radial profile owner set disagrees with mass identity")
    ordered = tuple(by_rank[rank] for rank in identity.owner_ranks)
    first = ordered[0]
    centres = first.radial_centres_box
    edges = first.radial_edges_box
    if centres is None or edges is None or len(centres) != 2 or len(edges) < 2:
        raise ValueError("FDM radial profile geometry is missing")
    nlevelmax = len(identity.mass_code_by_level)
    for item in ordered:
        radial = item.radial_density_sum_by_centre_level_bin
        if (
            item.radial_centres_box != centres
            or item.radial_edges_box != edges
            or item.ndim != first.ndim
            or item.boxlen_code != first.boxlen_code
            or item.wave_path.parent != item.amr_path.parent
            or not (
                item.wave_path.parent == identity.provenance_path.parent
                or (
                    item.wave_path.parent.parent == identity.provenance_path.parent
                    and item.wave_path.parent.name.startswith("group_")
                )
            )
            or len(item.density_sum_by_level) != nlevelmax
            or radial is None
            or len(radial) != 2
            or any(len(levels) != nlevelmax for levels in radial)
            or any(len(bins) != len(edges) - 1 for levels in radial for bins in levels)
            or any(not math.isfinite(value) or value < 0.0 for levels in radial for bins in levels for value in bins)
        ):
            raise ValueError("FDM radial profile summaries disagree")
    for level in range(1, nlevelmax + 1):
        dx = 0.5**level * first.boxlen_code / identity.coarse_cells_per_box
        level_density = math.fsum(item.density_sum_by_level[level - 1] for item in ordered)
        level_mass = level_density * dx**first.ndim
        if (
            not math.isclose(level_mass, identity.mass_code_by_level[level - 1], rel_tol=1.0e-10, abs_tol=1.0e-14)
            or sum(item.leaf_cells_by_level[level - 1] for item in ordered) != identity.leaf_cells_by_level[level - 1]
        ):
            raise ValueError("FDM radial profile does not match its leaf-mass identity")
        for centre_index in range(2):
            binned_density = math.fsum(
                item.radial_density_sum_by_centre_level_bin[centre_index][level - 1][bin_index]  # type: ignore[index]
                for item in ordered for bin_index in range(len(edges) - 1)
            )
            if binned_density > level_density * (1.0 + 1.0e-10):
                raise ValueError("FDM radial shells exceed their level's wave density")
    shells = []
    enclosed = []
    for centre_index in range(2):
        centre_shells = []
        for bin_index in range(len(edges) - 1):
            level_masses = []
            for level in range(1, nlevelmax + 1):
                density_sum = math.fsum(
                    item.radial_density_sum_by_centre_level_bin[centre_index][level - 1][bin_index]  # type: ignore[index]
                    for item in ordered
                )
                dx = 0.5**level * first.boxlen_code / identity.coarse_cells_per_box
                level_masses.append(density_sum * dx**first.ndim)
            centre_shells.append(math.fsum(level_masses))
        if not all(math.isfinite(value) and value >= 0.0 for value in centre_shells):
            raise ValueError("FDM radial shell mass is invalid")
        if math.fsum(centre_shells) > identity.reconstructed_mass_code * (1.0 + 1.0e-10):
            raise ValueError("FDM radial shells exceed the reconstructed wave mass")
        shells.append(tuple(centre_shells))
        enclosed.append(tuple(math.fsum(centre_shells[:index + 1]) for index in range(len(centre_shells))))
    return FDMRadialMassProfile(
        status="radial_wave_mass_measured_pending_core_decomposition",
        provenance_path=identity.provenance_path,
        centres_box=centres,
        edges_box=edges,
        shell_mass_code_by_centre=tuple(shells),
        enclosed_mass_code_by_centre=tuple(enclosed),
        total_wave_mass_code=identity.reconstructed_mass_code,
    )


def check_fdm_leaf_mass_identity(
    summaries: Sequence[OwnedLeafAmplitudeSummary],
    provenance: LagRamsesFDMOuterWaveProvenance,
    *,
    coarse_cells_per_box: int,
    maximum_relative_difference: float = 1.0e-8,
) -> FDMLeafMassIdentity:
    """Require all MPI owners, then apply the writer's exact cell-volume law.

    ``coarse_cells_per_box`` must be the effective
    ``icoarse_max-icoarse_min+1`` from the solver geometry, not an inferred
    array shape.  A mismatch returns an explicit censored identity result.
    """

    if (
        isinstance(coarse_cells_per_box, bool)
        or not isinstance(coarse_cells_per_box, int)
        or coarse_cells_per_box < 1
        or not math.isfinite(maximum_relative_difference)
        or maximum_relative_difference < 0.0
    ):
        raise ValueError("FDM leaf-mass identity parameters are invalid")
    if provenance.source_schema_version < 3 or provenance.mpi_ncpu is None:
        raise ValueError("FDM output requires V3+ provenance with MPI rank count")
    if not provenance.psi_snapshot_prefix.startswith("fdm_"):
        raise ValueError("FDM provenance snapshot prefix is invalid")
    ncpu = provenance.mpi_ncpu
    if ncpu < 1:
        raise ValueError("FDM provenance MPI rank count is invalid")
    output_directory = provenance.source_path.parent
    by_rank = {summary.owner_rank: summary for summary in summaries}
    if len(summaries) != ncpu or set(by_rank) != set(range(1, ncpu + 1)):
        raise ValueError("FDM output requires every MPI owner exactly once")
    ordered = [by_rank[rank] for rank in range(1, ncpu + 1)]
    first = ordered[0]
    nlevelmax = len(first.leaf_cells_by_level)
    if nlevelmax < 1:
        raise ValueError("FDM output has no AMR levels")
    for rank, summary in enumerate(ordered, start=1):
        wave_parent = summary.wave_path.parent
        valid_parent = wave_parent == output_directory or (
            wave_parent.parent == output_directory
            and wave_parent.name.startswith("group_")
            and len(wave_parent.name) == 11
            and wave_parent.name[6:].isdigit()
        )
        if (
            summary.ncpu != ncpu
            or summary.ndim != first.ndim
            or summary.boxlen_code != first.boxlen_code
            or summary.fdm_use_hjm != provenance.fdm_use_hjm
            or summary.fdm_first_wave_level != provenance.fdm_first_wave_level
            or len(summary.leaf_cells_by_level) != nlevelmax
            or len(summary.density_sum_by_level) != nlevelmax
            or summary.wave_path.name != f"{provenance.psi_snapshot_prefix}{rank:05d}"
            or summary.amr_path.name != (
                f"amr_{provenance.psi_snapshot_prefix.removeprefix('fdm_')}{rank:05d}"
            )
            or not valid_parent
            or summary.amr_path.parent != wave_parent
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in summary.leaf_cells_by_level
            )
            or any(
                not math.isfinite(value) or value < 0.0
                for value in summary.density_sum_by_level
            )
        ):
            raise ValueError("FDM owner summary differs from raw output identity")
    if first.ndim not in {1, 2, 3} or not math.isfinite(first.boxlen_code) or first.boxlen_code <= 0.0:
        raise ValueError("FDM owner geometry is invalid")
    level_counts = []
    level_masses = []
    for level in range(1, nlevelmax + 1):
        count = sum(summary.leaf_cells_by_level[level - 1] for summary in ordered)
        density_sum = math.fsum(
            summary.density_sum_by_level[level - 1] for summary in ordered
        )
        if count < 0 or not math.isfinite(density_sum) or density_sum < 0.0:
            raise ValueError("FDM owner leaf summary is invalid")
        dx_code = 0.5**level * first.boxlen_code / coarse_cells_per_box
        level_mass = density_sum * dx_code**first.ndim
        if not math.isfinite(level_mass):
            raise ValueError("FDM reconstructed leaf mass is non-finite")
        level_counts.append(count)
        level_masses.append(level_mass)
    reconstructed_mass = math.fsum(level_masses)
    writer_mass = provenance.leaf_mass_code
    writer_count = provenance.leaf_cell_count
    if not math.isfinite(writer_mass) or writer_mass < 0.0 or not math.isfinite(writer_count):
        raise ValueError("FDM raw mass/count provenance is invalid")
    if writer_mass == 0.0:
        relative_difference = 0.0 if reconstructed_mass == 0.0 else math.inf
    else:
        relative_difference = abs(reconstructed_mass - writer_mass) / writer_mass
    count = sum(level_counts)
    status = (
        "raw_leaf_mass_reconstruction_matches_writer"
        if count == writer_count and relative_difference <= maximum_relative_difference
        else "censored_raw_leaf_mass_or_count_mismatch"
    )
    return FDMLeafMassIdentity(
        status=status,
        provenance_path=provenance.source_path,
        coarse_cells_per_box=coarse_cells_per_box,
        owner_ranks=tuple(range(1, ncpu + 1)),
        leaf_cells_by_level=tuple(level_counts),
        mass_code_by_level=tuple(level_masses),
        reconstructed_mass_code=reconstructed_mass,
        writer_mass_code=writer_mass,
        mass_relative_difference=relative_difference,
        reconstructed_leaf_cells=count,
        writer_leaf_cells=writer_count,
    )


def assess_fdm_leaf_mass_series(
    ledger: DualSolitonRelaxationSampleLedger,
    identities: Sequence[FDMLeafMassIdentity],
    *,
    maximum_relative_mass_drift: float | None = None,
) -> FDMLeafMassSeries:
    """Assess only wave-mass drift on one verified output sequence.

    The caller must obtain ``ledger`` with
    ``read_verified_dual_soliton_relaxation_sample_ledger`` and build each
    identity from its exact shard set.  A passing mass-only status never
    substitutes for Hamiltonian, angular momentum, or core diagnostics.
    """

    default_limit = RelaxationConservationThresholds().maximum_relative_wave_mass_error
    limit = default_limit if maximum_relative_mass_drift is None else maximum_relative_mass_drift
    if not math.isfinite(limit) or limit <= 0.0 or limit > default_limit:
        raise ValueError("FDM mass-drift limit cannot weaken the relaxation gate")
    samples = tuple(ledger.samples)
    if len(samples) < 3 or len(identities) != len(samples):
        raise ValueError("FDM mass series requires one identity for each of three or more samples")
    times = tuple(float(sample.time_code) for sample in samples)
    if any(not math.isfinite(time) for time in times) or any(
        later <= earlier for earlier, later in zip(times, times[1:])
    ):
        raise ValueError("FDM mass sample times must increase strictly")
    paths = tuple(sample.raw_provenance_path for sample in samples)
    if any(
        identity.provenance_path != path
        for identity, path in zip(identities, paths, strict=True)
    ):
        raise ValueError("FDM mass identity is not aligned with the sample ledger")
    masses = tuple(identity.reconstructed_mass_code for identity in identities)
    reasons = []
    if any(
        identity.status != "raw_leaf_mass_reconstruction_matches_writer"
        for identity in identities
    ):
        reasons.append("at least one snapshot disagrees with its raw mass/count provenance")
    if any(not math.isfinite(mass) or mass < 0.0 for mass in masses):
        reasons.append("mass series contains a non-finite or negative value")
    drifts: tuple[float, ...] | None = None
    maximum_drift: float | None = None
    if not reasons:
        if masses[0] <= 0.0:
            reasons.append("initial wave mass is not positive")
        else:
            drifts = tuple(abs(mass - masses[0]) / masses[0] for mass in masses)
            maximum_drift = max(drifts)
            if maximum_drift > limit:
                reasons.append("wave-mass drift exceeds the declared relaxation limit")
    return FDMLeafMassSeries(
        status=(
            "mass_series_within_limit_pending_other_conservation"
            if not reasons else "censored_mass_series"
        ),
        sample_ledger_path=ledger.source_path,
        sample_ledger_sha256=ledger.source_sha256,
        sample_times_code=times,
        raw_provenance_paths=paths,
        mass_code=masses,
        relative_mass_drift_from_initial=drifts,
        maximum_relative_mass_drift=maximum_drift,
        declared_mass_drift_limit=limit,
        reasons=tuple(reasons),
    )
