"""Bounded same-level FDM wave-gradient diagnostics on an AMR level.

This reproduces the source writer's central-difference current where both
same-level neighbours are present.  Its gradient-square kinetic term is only
a finite-difference proxy: the base lagRamses FFT drift applies the discrete
Laplacian eigenvalue, and AMR interfaces need separately validated
ghost/reflux treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .fdm_shard_format import _RecordReader, inspect_fdm_amr_shard_pair
from .lagramses_fdm_provenance import LagRamsesFDMOuterWaveProvenance


@dataclass(frozen=True)
class FDMSameLevelStencil:
    status: str
    level: int
    owner_leaf_cells: int
    complete_leaf_stencil_cells: int
    incomplete_leaf_stencil_cells: int
    refined_neighbour_stencil_cells: int
    leaf_mass_code: float
    integrated_current_code: tuple[float, float, float]
    central_gradient_square_proxy_code: float
    interpretation: str


@dataclass(frozen=True)
class FDMShardStencilResult:
    wave_path: Path
    amr_path: Path
    owner_rank: int
    ncpu: int
    nlevelmax: int
    boxlen_code: float
    measurement: FDMSameLevelStencil


@dataclass(frozen=True)
class FDMWaveWriterIdentity:
    status: str
    provenance_path: Path
    owner_ranks: tuple[int, ...]
    nlevelmax: int
    reconstructed_leaf_cells: int
    writer_leaf_cells: float
    reconstructed_complete_stencil_cells: int
    writer_complete_stencil_cells: float
    reconstructed_mass_code: float
    writer_mass_code: float
    reconstructed_current_code: tuple[float, float, float]
    writer_current_code: tuple[float, float, float]
    mass_relative_error: float
    current_component_absolute_error: tuple[float, float, float]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FDMUniformFFTQuadratic:
    status: str
    level: int
    cell_count: int
    boxlen_code: float
    hbar_code: float
    wave_mass_code: float
    drift_generator_quadratic_code: float
    axis_quadratic_code: tuple[float, float, float]
    interpretation: str


def measure_uniform_fft_drift_quadratic(
    *,
    level: int,
    boxlen_code: float,
    hbar_code: float,
    wave_real: np.ndarray,
    wave_imag: np.ndarray,
    maximum_cells: int = 1_000_000,
) -> FDMUniformFFTQuadratic:
    """Measure the exact spatial quadratic of the base FFT drift generator.

    The selected lagRamses source applies a full C2C FFT with eigenvalue
    ``2 sum(cos(2*pi*k/N)-1) / dx_box**2``.  The equivalent spatial
    quadratic uses forward differences, not central-gradient squares or
    continuum ``k**2``.  This function accepts only an already assembled,
    uniform, periodic field; it does not verify its native source or certify
    a composite AMR Hamiltonian.
    """

    if (
        isinstance(level, bool) or not isinstance(level, int)
        or level < 1 or level > 30
        or not math.isfinite(boxlen_code) or boxlen_code <= 0.0
        or not math.isfinite(hbar_code) or hbar_code <= 0.0
        or isinstance(maximum_cells, bool) or not isinstance(maximum_cells, int)
        or maximum_cells < 8
    ):
        raise ValueError("FDM uniform FFT quadratic controls are invalid")
    shape = (1 << level,) * 3
    count = (1 << level) ** 3
    if count > maximum_cells or np.shape(wave_real) != shape or np.shape(wave_imag) != shape:
        raise ValueError("FDM uniform FFT field is incomplete or exceeds the memory bound")
    real = np.asarray(wave_real, dtype=np.float64)
    imaginary = np.asarray(wave_imag, dtype=np.float64)
    if np.any(~np.isfinite(real)) or np.any(~np.isfinite(imaginary)):
        raise ValueError("FDM uniform FFT field must be finite")
    dx_box = 0.5**level
    volume_code = (dx_box * boxlen_code) ** 3
    density_sum = np.sum(real**2 + imaginary**2, dtype=np.float64)
    axis = tuple(
        float(
            0.5 * hbar_code**2 * volume_code / dx_box**2
            * np.sum(
                (np.roll(real, -1, axis=dimension) - real) ** 2
                + (np.roll(imaginary, -1, axis=dimension) - imaginary) ** 2,
                dtype=np.float64,
            )
        )
        for dimension in range(3)
    )
    total = math.fsum(axis)
    mass = float(density_sum * volume_code)
    if not math.isfinite(mass) or not math.isfinite(total):
        raise ValueError("FDM uniform FFT quadratic is non-finite")
    return FDMUniformFFTQuadratic(
        status=("uniform_fft_drift_quadratic_pending_source_binding" if mass > 0.0
                else "censored_empty_uniform_fft_wave"),
        level=level, cell_count=count, boxlen_code=boxlen_code,
        hbar_code=hbar_code, wave_mass_code=mass,
        drift_generator_quadratic_code=total,
        axis_quadratic_code=axis,
        interpretation=(
            "exact discrete-Laplacian quadratic of the declared base FFT drift "
            "on an assembled uniform periodic field; not an AMR-composite "
            "Hamiltonian or conservation pass"
        ),
    )


def measure_fdm_same_level_stencil(
    *,
    level: int,
    coarse_grid_shape: tuple[int, int, int],
    boxlen_code: float,
    coarse_cells_per_box: int,
    hbar_code: float,
    grid_centres: np.ndarray,
    wave_real: np.ndarray,
    wave_imag: np.ndarray,
    son_grid_index: np.ndarray,
    owned_grid: np.ndarray,
    maximum_grids: int = 1_000_000,
) -> FDMSameLevelStencil:
    """Measure wave current only on complete same-level leaf-cell stencils.

    Grid centres and fields are the native saved AMR/FDM values for *one*
    output shard and level, including virtual-domain grids.  Owned grids are
    counted once; virtual grids supply neighbour values only.  Missing grids
    are censored rather than silently using zero gradients.
    """

    if (
        isinstance(level, bool) or not isinstance(level, int) or level < 1
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in coarse_grid_shape)
        or coarse_grid_shape != (1, 1, 1)
        or isinstance(coarse_cells_per_box, bool) or coarse_cells_per_box != 1
        or not math.isfinite(boxlen_code) or boxlen_code <= 0.0
        or not math.isfinite(hbar_code) or hbar_code <= 0.0
        or isinstance(maximum_grids, bool) or not isinstance(maximum_grids, int)
        or maximum_grids < 1
    ):
        raise ValueError("FDM stencil geometry or solver controls are invalid")
    centres = np.asarray(grid_centres, dtype=np.float64)
    real = np.asarray(wave_real, dtype=np.float64)
    imaginary = np.asarray(wave_imag, dtype=np.float64)
    sons = np.asarray(son_grid_index)
    owned = np.asarray(owned_grid)
    if (
        centres.ndim != 2 or centres.shape[1] != 3
        or real.shape != (len(centres), 8)
        or imaginary.shape != real.shape
        or sons.shape != real.shape
        or owned.shape != (len(centres),)
        or owned.dtype != np.dtype(bool)
        or len(centres) < 1 or len(centres) > maximum_grids
        or np.any(~np.isfinite(centres))
        or np.any(~np.isfinite(real))
        or np.any(~np.isfinite(imaginary))
        or not np.issubdtype(sons.dtype, np.integer)
        or np.any(sons < 0)
    ):
        raise ValueError("FDM same-level grid arrays are invalid or exceed the memory bound")
    grids_per_dim = 2 ** (level - 1)
    dx_box = 0.5**level
    dx_code = dx_box * boxlen_code
    grid_width = 2.0 * dx_box
    coordinates = np.rint(centres / grid_width - 0.5).astype(np.int64)
    expected = (coordinates + 0.5) * grid_width
    if (
        np.any(coordinates < 0)
        or np.any(coordinates >= grids_per_dim)
        or np.any(np.abs(centres - expected) > 1.0e-8 * dx_box)
    ):
        raise ValueError("FDM saved AMR grid centres do not match the unit-box lattice")
    index = {tuple(value): row for row, value in enumerate(coordinates)}
    if len(index) != len(centres):
        raise ValueError("FDM shard repeats a same-level grid coordinate")
    owner_count = 0
    complete_count = 0
    refined_neighbour_count = 0
    mass_sum = 0.0
    current_sum = np.zeros(3, dtype=np.float64)
    gradient_square_sum = 0.0
    for row in np.flatnonzero(owned):
        for child in range(8):
            if sons[row, child] != 0:
                continue
            owner_count += 1
            re = real[row, child]
            im = imaginary[row, child]
            mass_sum += float(re * re + im * im)
            current = np.zeros(3, dtype=np.float64)
            gradient_square = 0.0
            neighbour_refined = False
            complete = True
            for dimension in range(3):
                bit = (child >> dimension) & 1
                neighbours = []
                for direction in (-1, 1):
                    if (direction == 1 and bit == 0) or (direction == -1 and bit == 1):
                        neighbours.append((row, child ^ (1 << dimension)))
                    else:
                        neighbour_grid = coordinates[row].copy()
                        neighbour_grid[dimension] = (
                            neighbour_grid[dimension] + direction
                        ) % grids_per_dim
                        neighbour_row = index.get(tuple(neighbour_grid))
                        if neighbour_row is None:
                            complete = False
                            break
                        neighbours.append((neighbour_row, child ^ (1 << dimension)))
                if not complete:
                    break
                (left_row, left_child), (right_row, right_child) = neighbours
                neighbour_refined = neighbour_refined or (
                    sons[left_row, left_child] != 0
                    or sons[right_row, right_child] != 0
                )
                grad_real = (
                    real[right_row, right_child] - real[left_row, left_child]
                ) / (2.0 * dx_code)
                grad_imag = (
                    imaginary[right_row, right_child] - imaginary[left_row, left_child]
                ) / (2.0 * dx_code)
                current[dimension] = hbar_code * (re * grad_imag - im * grad_real)
                gradient_square += grad_real**2 + grad_imag**2
            if complete:
                complete_count += 1
                refined_neighbour_count += int(neighbour_refined)
                current_sum += current
                gradient_square_sum += gradient_square
    volume = dx_code**3
    current_integral = current_sum * volume
    kinetic_proxy = 0.5 * hbar_code**2 * gradient_square_sum * volume
    if not np.all(np.isfinite(current_integral)) or not math.isfinite(kinetic_proxy):
        raise ValueError("FDM same-level stencil diagnostic is non-finite")
    return FDMSameLevelStencil(
        status=(
            "same_level_current_complete_pending_writer_identity"
            if owner_count > 0 and complete_count == owner_count
            and refined_neighbour_count == 0
            else "censored_incomplete_or_refined_same_level_stencil"
        ),
        level=level,
        owner_leaf_cells=owner_count,
        complete_leaf_stencil_cells=complete_count,
        incomplete_leaf_stencil_cells=owner_count - complete_count,
        refined_neighbour_stencil_cells=refined_neighbour_count,
        leaf_mass_code=mass_sum * volume,
        integrated_current_code=tuple(float(value) for value in current_integral),
        central_gradient_square_proxy_code=kinetic_proxy,
        interpretation=(
            "central-difference wave current on available same-level stencils; "
            "central-gradient square is not the FFT discrete-Laplacian or "
            "AMR-composite kinetic Hamiltonian"
        ),
    )


def measure_fdm_shard_same_level_stencil(
    wave_path: str | Path,
    amr_path: str | Path,
    *,
    owner_rank: int,
    level: int,
    simple_boundary: bool,
    fdm_use_hjm: bool,
    fdm_first_wave_level: int,
    hbar_code: float,
    expected_ncpu: int | None = None,
    byte_order: str = "<",
    maximum_grids: int = 100_000,
    maximum_array_bytes: int = 16 * 1024 * 1024,
) -> FDMShardStencilResult:
    """Read one bounded native wave/AMR level and reconstruct its owner current.

    Other levels are frame-checked and skipped.  No source-writer current
    identity, cross-rank conservation, or true kinetic energy is asserted.
    """

    if (
        isinstance(owner_rank, bool) or not isinstance(owner_rank, int)
        or isinstance(level, bool) or not isinstance(level, int)
        or not isinstance(simple_boundary, bool)
        or not isinstance(fdm_use_hjm, bool)
        or isinstance(fdm_first_wave_level, bool)
        or not isinstance(fdm_first_wave_level, int)
        or fdm_first_wave_level < 1
        or isinstance(maximum_grids, bool) or not isinstance(maximum_grids, int)
        or maximum_grids < 1
        or isinstance(maximum_array_bytes, bool)
        or not isinstance(maximum_array_bytes, int)
        or maximum_array_bytes < 8
    ):
        raise ValueError("FDM shard stencil controls are invalid")
    pair = inspect_fdm_amr_shard_pair(
        wave_path, amr_path, simple_boundary=simple_boundary,
        expected_ncpu=expected_ncpu, byte_order=byte_order,
    )
    wave, amr = pair.wave, pair.amr
    if (
        wave.ndim != 3 or amr.nx_ny_nz != (1, 1, 1)
        or owner_rank < 1 or owner_rank > wave.ncpu
        or not wave.path.name.endswith(f"{owner_rank:05d}")
        or not amr.path.name.endswith(f"{owner_rank:05d}")
        or (fdm_use_hjm and level < fdm_first_wave_level)
        or level < 1 or level > wave.nlevelmax
    ):
        raise ValueError("FDM shard stencil requires a wave-level unit-box owner shard")
    selected_counts = wave.grid_counts_by_level_and_domain[level - 1]
    if sum(selected_counts) > maximum_grids:
        raise ValueError("FDM shard stencil exceeds the declared level-grid memory bound")
    centres_blocks: list[np.ndarray] = []
    real_blocks: list[np.ndarray] = []
    imaginary_blocks: list[np.ndarray] = []
    son_blocks: list[np.ndarray] = []
    owner_blocks: list[np.ndarray] = []
    real_dtype = np.dtype(byte_order + "f8")
    integer_dtype = np.dtype(byte_order + "i4")
    with wave.path.open("rb") as wave_stream, amr.path.open("rb") as amr_stream:
        wave_reader = _RecordReader(wave_stream, byte_order=byte_order)
        amr_reader = _RecordReader(amr_stream, byte_order=byte_order)
        for _ in range(4):
            wave_reader.scalar()
        amr_stream.seek(amr.fine_payload_offset)
        for current_level in range(1, wave.nlevelmax + 1):
            for domain, ncache in enumerate(
                wave.grid_counts_by_level_and_domain[current_level - 1], start=1
            ):
                if wave_reader.scalar() != current_level or wave_reader.scalar() != ncache:
                    raise ValueError("FDM shard block identity changed during stencil extraction")
                if not ncache:
                    continue
                selected = current_level == level
                if selected and 8 * ncache > maximum_array_bytes:
                    raise ValueError("FDM shard stencil array exceeds its memory bound")
                for _ in range(3):
                    amr_reader.record(4 * ncache)
                axes = []
                for _ in range(3):
                    payload = amr_reader.record(8 * ncache, read_payload=selected)
                    if selected:
                        axes.append(np.frombuffer(payload, dtype=real_dtype))
                for _ in range(7):  # father and six neighbour cells
                    amr_reader.record(4 * ncache)
                sons = []
                for _ in range(8):
                    payload = amr_reader.record(4 * ncache, read_payload=selected)
                    if selected:
                        sons.append(np.frombuffer(payload, dtype=integer_dtype))
                for _ in range(16):  # cpu and refinement maps
                    amr_reader.record(4 * ncache)
                re_children = []
                im_children = []
                for _ in range(8):
                    real_payload = wave_reader.record(8 * ncache, read_payload=selected)
                    imag_payload = wave_reader.record(8 * ncache, read_payload=selected)
                    if selected:
                        re_children.append(np.frombuffer(real_payload, dtype=real_dtype))
                        im_children.append(np.frombuffer(imag_payload, dtype=real_dtype))
                if selected:
                    centres_blocks.append(np.stack(axes, axis=1))
                    son_blocks.append(np.stack(sons, axis=1))
                    real_blocks.append(np.stack(re_children, axis=1))
                    imaginary_blocks.append(np.stack(im_children, axis=1))
                    owner_blocks.append(np.full(ncache, domain == owner_rank, dtype=bool))
        if wave_stream.read(1) or amr_stream.read(1):
            raise ValueError("FDM/AMR shard changed during stencil extraction")
    if not centres_blocks:
        measurement = FDMSameLevelStencil(
            status="censored_no_same_level_wave_grids", level=level,
            owner_leaf_cells=0, complete_leaf_stencil_cells=0,
            incomplete_leaf_stencil_cells=0, refined_neighbour_stencil_cells=0,
            leaf_mass_code=0.0, integrated_current_code=(0.0, 0.0, 0.0),
            central_gradient_square_proxy_code=0.0,
            interpretation="no same-level wave grids; no current or kinetic inference",
        )
    else:
        measurement = measure_fdm_same_level_stencil(
            level=level, coarse_grid_shape=amr.nx_ny_nz,
            boxlen_code=amr.boxlen_code, coarse_cells_per_box=1,
            hbar_code=hbar_code, grid_centres=np.concatenate(centres_blocks),
            wave_real=np.concatenate(real_blocks),
            wave_imag=np.concatenate(imaginary_blocks),
            son_grid_index=np.concatenate(son_blocks),
            owned_grid=np.concatenate(owner_blocks), maximum_grids=maximum_grids,
        )
    return FDMShardStencilResult(
        wave_path=wave.path, amr_path=amr.path, owner_rank=owner_rank,
        ncpu=wave.ncpu, nlevelmax=wave.nlevelmax,
        boxlen_code=amr.boxlen_code, measurement=measurement,
    )


def check_fdm_wave_writer_identity(
    results: Sequence[FDMShardStencilResult],
    provenance: LagRamsesFDMOuterWaveProvenance,
    *,
    maximum_relative_mass_error: float = 1.0e-10,
    maximum_relative_current_error: float = 1.0e-9,
    maximum_absolute_current_error: float = 1.0e-12,
) -> FDMWaveWriterIdentity:
    """Compare every pure-wave rank/level with the raw writer's global sums.

    Caller must first verify the immutable sample-ledger hashes.  Agreement
    is a saved-field versus writer-current identity, not wave-Hamiltonian or
    time-series conservation evidence.
    """

    thresholds = (
        maximum_relative_mass_error, maximum_relative_current_error,
        maximum_absolute_current_error,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
        raise ValueError("FDM writer-identity tolerances must be finite and non-negative")
    if provenance.fdm_use_hjm:
        raise ValueError("HJM current requires a separate phase-gradient reconstruction")
    ncpu = provenance.mpi_ncpu
    if ncpu is None or ncpu < 1 or not results:
        raise ValueError("FDM writer identity requires all MPI owner-level results")
    nlevelmax = results[0].nlevelmax
    boxlen = results[0].boxlen_code
    if nlevelmax < 1 or not math.isfinite(boxlen) or boxlen <= 0.0:
        raise ValueError("FDM writer identity has invalid level or box geometry")
    if not provenance.psi_snapshot_prefix.startswith("fdm_"):
        raise ValueError("FDM writer snapshot prefix is invalid")
    amr_prefix = "amr_" + provenance.psi_snapshot_prefix.removeprefix("fdm_")
    by_owner_level: dict[tuple[int, int], FDMShardStencilResult] = {}
    owner_paths: dict[int, tuple[Path, Path]] = {}
    for result in results:
        rank = result.owner_rank
        level = result.measurement.level
        key = rank, level
        measure = result.measurement
        if (
            key in by_owner_level or rank < 1 or rank > ncpu
            or level < 1 or level > nlevelmax
            or result.ncpu != ncpu or result.nlevelmax != nlevelmax
            or result.boxlen_code != boxlen
            or result.wave_path.name != f"{provenance.psi_snapshot_prefix}{rank:05d}"
            or result.amr_path.name != f"{amr_prefix}{rank:05d}"
            or result.wave_path == result.amr_path
            or measure.owner_leaf_cells < 0
            or measure.complete_leaf_stencil_cells < 0
            or measure.complete_leaf_stencil_cells > measure.owner_leaf_cells
            or measure.incomplete_leaf_stencil_cells != (
                measure.owner_leaf_cells - measure.complete_leaf_stencil_cells
            )
            or measure.refined_neighbour_stencil_cells < 0
            or measure.refined_neighbour_stencil_cells > measure.complete_leaf_stencil_cells
            or not math.isfinite(measure.leaf_mass_code)
            or measure.leaf_mass_code < 0.0
            or len(measure.integrated_current_code) != 3
            or any(not math.isfinite(value) for value in measure.integrated_current_code)
        ):
            raise ValueError("FDM owner-level writer identity source or measurement is invalid")
        paths = result.wave_path, result.amr_path
        if rank in owner_paths and owner_paths[rank] != paths:
            raise ValueError("FDM owner shard path changes between levels")
        owner_paths[rank] = paths
        by_owner_level[key] = result
    expected = {(rank, level) for rank in range(1, ncpu + 1)
                for level in range(1, nlevelmax + 1)}
    if set(by_owner_level) != expected:
        raise ValueError("FDM writer identity is missing an MPI owner or AMR level")
    if len({path for paths in owner_paths.values() for path in paths}) != 2 * ncpu:
        raise ValueError("FDM owner shard files alias across ranks")
    ordered = [by_owner_level[key] for key in sorted(expected)]
    cells = sum(item.measurement.owner_leaf_cells for item in ordered)
    stencils = sum(item.measurement.complete_leaf_stencil_cells for item in ordered)
    mass = math.fsum(item.measurement.leaf_mass_code for item in ordered)
    current = tuple(
        math.fsum(item.measurement.integrated_current_code[dimension] for item in ordered)
        for dimension in range(3)
    )
    writer_mass = provenance.leaf_mass_code
    writer_current = provenance.integrated_current_code
    if (
        not math.isfinite(mass) or not all(math.isfinite(x) for x in current)
        or not math.isfinite(writer_mass) or writer_mass < 0.0
        or len(writer_current) != 3
        or any(not math.isfinite(x) for x in writer_current)
        or not math.isfinite(provenance.leaf_cell_count)
        or not math.isfinite(provenance.complete_current_stencil_cell_count)
        or provenance.leaf_cell_count < 0.0
        or provenance.complete_current_stencil_cell_count < 0.0
    ):
        raise ValueError("FDM writer identity mass or current is non-finite")
    mass_relative_error = (
        abs(mass - writer_mass) / writer_mass if writer_mass > 0.0
        else (0.0 if mass == 0.0 else math.inf)
    )
    current_errors = tuple(abs(a - b) for a, b in zip(current, writer_current, strict=True))
    reasons = []
    if cells == 0:
        reasons.append("snapshot contains no owned wave leaf cells")
    if cells != provenance.leaf_cell_count:
        reasons.append("owned leaf-cell count differs from writer")
    if stencils != provenance.complete_current_stencil_cell_count:
        reasons.append("complete current-stencil count differs from writer")
    if mass_relative_error > maximum_relative_mass_error:
        reasons.append("owned wave mass differs from writer")
    if any(
        error > maximum_absolute_current_error
        + maximum_relative_current_error * max(abs(a), abs(b))
        for error, a, b in zip(current_errors, current, writer_current, strict=True)
    ):
        reasons.append("saved-wave current differs from writer")
    if any(
        item.measurement.status != "same_level_current_complete_pending_writer_identity"
        and item.measurement.owner_leaf_cells > 0
        for item in ordered
    ):
        reasons.append("at least one owner level has an incomplete or refined stencil")
    return FDMWaveWriterIdentity(
        status=("saved_wave_writer_current_identity_matches" if not reasons
                else "censored_saved_wave_writer_current_identity"),
        provenance_path=provenance.source_path,
        owner_ranks=tuple(range(1, ncpu + 1)), nlevelmax=nlevelmax,
        reconstructed_leaf_cells=cells, writer_leaf_cells=provenance.leaf_cell_count,
        reconstructed_complete_stencil_cells=stencils,
        writer_complete_stencil_cells=provenance.complete_current_stencil_cell_count,
        reconstructed_mass_code=mass, writer_mass_code=writer_mass,
        reconstructed_current_code=current, writer_current_code=writer_current,
        mass_relative_error=mass_relative_error,
        current_component_absolute_error=current_errors,
        reasons=tuple(reasons),
    )
