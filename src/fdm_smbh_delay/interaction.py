"""Hamiltonian diagnostics for a live FDM wave coupled to compact masses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import G_INTERNAL


def unresolved_binary_potential_correction(
    *,
    grid_positions_pc: np.ndarray,
    centre_of_mass_pc: np.ndarray,
    member_displacements_pc: np.ndarray,
    member_masses_msun: np.ndarray,
    plummer_radius_pc: float,
    periodic_box_pc: float | np.ndarray | None = None,
) -> np.ndarray:
    """Return the binary potential after subtracting its centre-of-mass monopole.

    The correction exposes the rotating multipoles of an unresolved internal
    orbit while a single resolved particle supplies the total monopole.
    """

    grid = np.asarray(grid_positions_pc, dtype=float)
    centre = np.asarray(centre_of_mass_pc, dtype=float)
    displacement = np.asarray(member_displacements_pc, dtype=float)
    masses = np.asarray(member_masses_msun, dtype=float)
    if grid.ndim < 1 or grid.shape[-1] != 3:
        raise ValueError("grid positions must end in a three-vector axis")
    if centre.shape != (3,) or displacement.shape != (2, 3) or masses.shape != (2,):
        raise ValueError("binary centre, displacements, and masses have invalid shapes")
    if (
        np.any(~np.isfinite(grid))
        or np.any(~np.isfinite(centre))
        or np.any(~np.isfinite(displacement))
        or np.any(~np.isfinite(masses))
    ):
        raise ValueError("binary potential inputs must be finite")
    if (
        np.any(masses <= 0.0)
        or not np.isfinite(plummer_radius_pc)
        or plummer_radius_pc <= 0.0
    ):
        raise ValueError("masses and Plummer radius must be positive")
    centre_offset = np.sum(masses[:, None] * displacement, axis=0)
    scale = np.sum(masses) * max(
        float(np.max(np.linalg.norm(displacement, axis=1))), 1.0
    )
    if np.linalg.norm(centre_offset) > 1.0e-12 * scale:
        raise ValueError("member displacements must be centre-of-mass centred")

    def relative_position(source: np.ndarray) -> np.ndarray:
        relative = grid - source
        if periodic_box_pc is not None:
            box = np.asarray(periodic_box_pc, dtype=float)
            if box.shape not in ((), (3,)):
                raise ValueError("periodic box must be scalar or a three-vector")
            if np.any(~np.isfinite(box)) or np.any(box <= 0.0):
                raise ValueError("periodic box lengths must be finite and positive")
            relative = relative - box * np.floor(relative / box + 0.5)
        return relative

    monopole_relative = relative_position(centre)
    monopole = -G_INTERNAL * np.sum(masses) / np.sqrt(
        np.sum(monopole_relative**2, axis=-1) + plummer_radius_pc**2
    )
    pair = np.zeros(grid.shape[:-1], dtype=float)
    for mass, member_displacement in zip(masses, displacement, strict=True):
        relative = relative_position(centre + member_displacement)
        pair -= G_INTERNAL * mass / np.sqrt(
            np.sum(relative**2, axis=-1) + plummer_radius_pc**2
        )
    return pair - monopole


def wave_bh_interaction_energy(
    density_msun_pc3: np.ndarray,
    bh_potential_pc2_myr2: np.ndarray,
    cell_volume_pc3: float,
) -> float:
    """Return the cross term integral rho * Phi_BH dV, counted once."""

    density = np.asarray(density_msun_pc3, dtype=float)
    potential = np.asarray(bh_potential_pc2_myr2, dtype=float)
    if density.shape != potential.shape:
        raise ValueError("density and SMBH potential must have the same shape")
    if cell_volume_pc3 <= 0.0:
        raise ValueError("cell volume must be positive")
    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("density must be finite and non-negative")
    if np.any(~np.isfinite(potential)):
        raise ValueError("SMBH potential must be finite")
    return float(np.sum(density * potential) * cell_volume_pc3)


def coupled_hamiltonian(
    *,
    wave_kinetic: np.ndarray | float,
    wave_self_gravity: np.ndarray | float,
    wave_bh_interaction: np.ndarray | float,
    bh_kinetic: np.ndarray | float,
    bh_mutual_gravity: np.ndarray | float,
) -> np.ndarray:
    """Total energy for a reciprocally coupled wave--SMBH system.

    ``wave_self_gravity`` already contains its factor of one half.  The cross
    interaction is counted once and carries no factor of one half.
    """

    return (
        np.asarray(wave_kinetic, dtype=float)
        + np.asarray(wave_self_gravity, dtype=float)
        + np.asarray(wave_bh_interaction, dtype=float)
        + np.asarray(bh_kinetic, dtype=float)
        + np.asarray(bh_mutual_gravity, dtype=float)
    )


def gauge_align_point_estimator(
    point_energy: np.ndarray, grid_energy: np.ndarray, reference_index: int = 0
) -> tuple[np.ndarray, float]:
    """Align a point-sampled potential energy to the grid cross term.

    A periodic Poisson solve normally fixes the mean wave potential to zero,
    whereas a compact-mass potential may use zero at infinity.  Absolute cross
    energies then differ by a gauge constant.  Alignment is useful only as a
    diagnostic of energy *changes*; the returned array does not replace the
    grid Hamiltonian term.
    """

    point = np.asarray(point_energy, dtype=float)
    grid = np.asarray(grid_energy, dtype=float)
    if point.shape != grid.shape or point.ndim != 1 or point.size == 0:
        raise ValueError("point and grid energies must be non-empty 1D arrays")
    if not 0 <= reference_index < point.size:
        raise IndexError("reference_index is outside the energy arrays")
    offset = float(grid[reference_index] - point[reference_index])
    return point + offset, offset


def periodic_poisson_potential(
    density_msun_pc3: np.ndarray, cell_size_pc: float
) -> np.ndarray:
    """Solve periodic Poisson gravity with the zero Fourier mode removed."""

    density = np.asarray(density_msun_pc3, dtype=float)
    if density.ndim != 3 or len(set(density.shape)) != 1:
        raise ValueError("density must be a cubic 3D array")
    if cell_size_pc <= 0.0:
        raise ValueError("cell size must be positive")
    n = density.shape[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=cell_size_pc)[:, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=cell_size_pc)[None, :, None]
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=cell_size_pc)[None, None, :]
    k_squared = kx * kx + ky * ky + kz * kz
    k_squared[0, 0, 0] = np.inf
    potential_k = -4.0 * np.pi * G_INTERNAL * np.fft.rfftn(density) / k_squared
    return np.fft.irfftn(
        potential_k, s=density.shape, axes=(0, 1, 2)
    ).real


def trilinear_periodic(
    field: np.ndarray, position_pc: np.ndarray, box_size_pc: float
) -> float:
    """Trilinearly sample a cell-corner periodic field."""

    values = np.asarray(field, dtype=float)
    position = np.asarray(position_pc, dtype=float)
    if values.ndim != 3 or len(set(values.shape)) != 1:
        raise ValueError("field must be a cubic 3D array")
    if position.shape != (3,) or box_size_pc <= 0.0:
        raise ValueError("position must have shape (3,) and box size must be positive")
    n = values.shape[0]
    coordinate = (position / box_size_pc + 0.5) * n
    lower = np.floor(coordinate).astype(int)
    fraction = coordinate - lower
    result = 0.0
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                weight = (
                    (fraction[0] if di else 1.0 - fraction[0])
                    * (fraction[1] if dj else 1.0 - fraction[1])
                    * (fraction[2] if dk else 1.0 - fraction[2])
                )
                index = tuple((lower + (di, dj, dk)) % n)
                result += weight * values[index]
    return float(result)


@dataclass(frozen=True)
class SingleBhInteractionSnapshot:
    grid_interaction_energy: float
    point_interaction_energy: float
    force_on_bh_from_grid_energy: np.ndarray
    force_on_bh_from_point_potential: np.ndarray


def single_bh_interaction_snapshot(
    *,
    density_msun_pc3: np.ndarray,
    wave_potential_pc2_myr2: np.ndarray,
    box_size_pc: float,
    bh_mass_msun: float,
    bh_position_pc: np.ndarray,
    plummer_radius_pc: float,
    wave_gradient_pc_myr2: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> SingleBhInteractionSnapshot:
    """Evaluate both PyUL-style cross-energy estimators and their forces."""

    density = np.asarray(density_msun_pc3, dtype=float)
    wave_potential = np.asarray(wave_potential_pc2_myr2, dtype=float)
    position = np.asarray(bh_position_pc, dtype=float)
    if density.shape != wave_potential.shape:
        raise ValueError("density and wave potential must have the same shape")
    if density.ndim != 3 or len(set(density.shape)) != 1:
        raise ValueError("density must be a cubic 3D array")
    if position.shape != (3,):
        raise ValueError("SMBH position must have shape (3,)")
    if bh_mass_msun <= 0.0 or plummer_radius_pc <= 0.0:
        raise ValueError("SMBH mass and Plummer radius must be positive")
    n = density.shape[0]
    cell_size = box_size_pc / n
    coordinates = np.linspace(
        -0.5 * box_size_pc, 0.5 * box_size_pc, n, endpoint=False
    )
    dx = coordinates[:, None, None] - position[0]
    dy = coordinates[None, :, None] - position[1]
    dz = coordinates[None, None, :] - position[2]
    softened_radius_squared = (
        dx * dx + dy * dy + dz * dz + plummer_radius_pc**2
    )
    bh_potential = -G_INTERNAL * bh_mass_msun / np.sqrt(
        softened_radius_squared
    )
    grid_energy = wave_bh_interaction_energy(
        density, bh_potential, cell_size**3
    )
    point_energy = bh_mass_msun * trilinear_periodic(
        wave_potential, position, box_size_pc
    )

    factor = (
        G_INTERNAL
        * bh_mass_msun
        * density
        / softened_radius_squared**1.5
        * cell_size**3
    )
    force_from_grid = np.array(
        [np.sum(factor * dx), np.sum(factor * dy), np.sum(factor * dz)]
    )
    gradients = (
        np.gradient(wave_potential, cell_size, edge_order=2)
        if wave_gradient_pc_myr2 is None
        else wave_gradient_pc_myr2
    )
    if len(gradients) != 3 or any(
        np.asarray(component).shape != density.shape for component in gradients
    ):
        raise ValueError("wave gradient must contain three arrays matching density")
    force_from_point = -bh_mass_msun * np.array(
        [
            trilinear_periodic(component, position, box_size_pc)
            for component in gradients
        ]
    )
    return SingleBhInteractionSnapshot(
        grid_interaction_energy=grid_energy,
        point_interaction_energy=point_energy,
        force_on_bh_from_grid_energy=force_from_grid,
        force_on_bh_from_point_potential=force_from_point,
    )
