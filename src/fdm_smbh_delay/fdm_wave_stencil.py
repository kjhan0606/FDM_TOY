"""Bounded same-level FDM wave-gradient diagnostics on an AMR level.

This reproduces the source writer's central-difference current where both
same-level neighbours are present.  Its gradient-square kinetic term is only
a finite-difference proxy: the base lagRamses kinetic operator is spectral,
and AMR interfaces need separately validated ghost/reflux treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .fdm_shard_format import _RecordReader, inspect_fdm_amr_shard_pair


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
    measurement: FDMSameLevelStencil


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
            "gradient-square term is not the spectral/AMR kinetic Hamiltonian"
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
        or level < fdm_first_wave_level or level > wave.nlevelmax
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
        ncpu=wave.ncpu, measurement=measurement,
    )
