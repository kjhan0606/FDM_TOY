"""Conservative grid targets for SMBH-to-wave energy and momentum exchange.

This module maps an already-computed drag exchange onto normalized grid source
fields. It deliberately does not mutate a wavefunction: a physically closed
Schrödinger--Poisson coupling must obtain the wake force from the evolved wave
and must disable the analytic drag force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lagramses import minimum_image


@dataclass(frozen=True)
class WaveSourceFields:
    energy_lab_density_rate: np.ndarray
    excitation_energy_density_rate: np.ndarray
    momentum_density_rate: np.ndarray


def normalized_gaussian_kernel(
    grid_positions_pc: np.ndarray,
    centre_pc: np.ndarray,
    sigma_pc: float,
    cell_volume_pc3: float | np.ndarray,
    *,
    periodic_box_pc: float | np.ndarray | None = None,
) -> np.ndarray:
    """Return discrete density weights normalized to one up to roundoff."""

    positions = np.asarray(grid_positions_pc, dtype=float)
    if positions.shape[-1] != 3:
        raise ValueError("grid_positions_pc must end in a three-vector axis")
    if sigma_pc <= 0.0:
        raise ValueError("sigma_pc must be positive")
    displacement = positions - np.asarray(centre_pc, dtype=float)
    if periodic_box_pc is not None:
        displacement = minimum_image(displacement, periodic_box_pc)
    raw = np.exp(-0.5 * np.sum(displacement**2, axis=-1) / sigma_pc**2)
    cell_volume = np.asarray(cell_volume_pc3, dtype=float)
    if np.any(cell_volume <= 0.0):
        raise ValueError("cell volumes must be positive")
    normalization = float(np.sum(raw * cell_volume))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("deposition kernel has zero support on the supplied grid")
    return raw / normalization


def build_wave_source_fields(
    *,
    grid_positions_pc: np.ndarray,
    cell_volume_pc3: float | np.ndarray,
    bh_positions_pc: np.ndarray,
    bh_velocities_pc_myr: np.ndarray,
    forces_on_bh_msun_pc_myr2: np.ndarray,
    fdm_bulk_velocities_pc_myr: np.ndarray,
    sigma_pc: float,
    periodic_box_pc: float | np.ndarray | None = None,
) -> WaveSourceFields:
    """Deposit equal-and-opposite drag exchanges around the SMBHs.

    The volume integrals obey ``dP_wave/dt = -sum(F_on_BH)``,
    ``dE_wave,lab/dt = -sum(F_on_BH dot v_BH)``, and
    ``dE_wave,exc/dt = -sum(F_on_BH dot (v_BH-u_FDM))``.

    These are source *targets*. Directly applying them to ``psi`` requires a
    mass-preserving phase/amplitude operator and a matching total-Hamiltonian
    test; this function alone is not a wavefunction update.
    """

    positions = np.asarray(bh_positions_pc, dtype=float)
    velocities = np.asarray(bh_velocities_pc_myr, dtype=float)
    forces = np.asarray(forces_on_bh_msun_pc_myr2, dtype=float)
    bulk = np.asarray(fdm_bulk_velocities_pc_myr, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("bh_positions_pc must have shape (N, 3)")
    if velocities.shape != positions.shape or forces.shape != positions.shape:
        raise ValueError("BH positions, velocities, and forces must share shape (N, 3)")
    if bulk.shape == (3,):
        bulk = np.broadcast_to(bulk, positions.shape)
    if bulk.shape != positions.shape:
        raise ValueError("fdm_bulk_velocities_pc_myr must have shape (3,) or (N, 3)")

    grid_shape = np.asarray(grid_positions_pc).shape[:-1]
    energy_lab = np.zeros(grid_shape, dtype=float)
    excitation_energy = np.zeros(grid_shape, dtype=float)
    momentum = np.zeros(grid_shape + (3,), dtype=float)
    for centre, velocity, force, local_bulk in zip(
        positions, velocities, forces, bulk, strict=True
    ):
        kernel = normalized_gaussian_kernel(
            grid_positions_pc,
            centre,
            sigma_pc,
            cell_volume_pc3,
            periodic_box_pc=periodic_box_pc,
        )
        momentum += kernel[..., None] * (-force)
        energy_lab += kernel * (-float(force @ velocity))
        excitation_energy += kernel * (-float(force @ (velocity - local_bulk)))
    return WaveSourceFields(energy_lab, excitation_energy, momentum)
