"""Bounded same-level FDM wave-gradient diagnostics on an AMR level.

This reproduces the source writer's central-difference current where both
same-level neighbours are present.  Its gradient-square kinetic term is only
a finite-difference proxy: the base lagRamses kinetic operator is spectral,
and AMR interfaces need separately validated ghost/reflux treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


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
