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
