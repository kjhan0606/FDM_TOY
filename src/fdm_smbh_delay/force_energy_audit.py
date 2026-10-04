"""Fixed-density force conjugate to the solver's gridded Plummer energy.

This is a diagnostic, not a replacement for the production wave force.
"""

from __future__ import annotations

import numpy as np


def plummer_grid_conjugate_force(
    *,
    density: np.ndarray,
    coordinate: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    plummer_radius: float,
    cell_volume: float,
    slab_depth: int = 16,
) -> np.ndarray:
    """Return ``-d(sum(rho Phi_BH) dV)/dX`` in the solver's code units.

    The derivative holds the grid density fixed and uses the same non-periodic
    Plummer kernel as ``plummer_potential_torch``. Slicing the first axis keeps
    the working set bounded for a 256-cubed diagnostic.
    """

    rho = np.asarray(density, dtype=np.float64)
    axis = np.asarray(coordinate, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    centre = np.asarray(positions, dtype=np.float64)
    if (
        rho.ndim != 3 or len(set(rho.shape)) != 1
        or axis.shape != (rho.shape[0],)
        or mass.ndim != 1 or centre.shape != (mass.size, 3)
        or mass.size == 0 or slab_depth < 1
    ):
        raise ValueError("Plummer force audit has incompatible grid or body shapes")
    if (
        not np.all(np.isfinite(rho)) or np.any(rho < 0.0)
        or not np.all(np.isfinite(axis))
        or not np.all(np.isfinite(mass)) or np.any(mass <= 0.0)
        or not np.all(np.isfinite(centre))
        or not np.isfinite(plummer_radius) or plummer_radius <= 0.0
        or not np.isfinite(cell_volume) or cell_volume <= 0.0
    ):
        raise ValueError("Plummer force audit requires finite physical inputs")
    force = np.zeros((mass.size, 3), dtype=np.float64)
    for body, (body_mass, position) in enumerate(zip(mass, centre, strict=True)):
        y = axis[None, :, None] - position[1]
        z = axis[None, None, :] - position[2]
        yz_squared = y * y + z * z + plummer_radius**2
        for start in range(0, axis.size, slab_depth):
            stop = min(start + slab_depth, axis.size)
            x = axis[start:stop, None, None] - position[0]
            radius_squared = x * x + yz_squared
            weighted = rho[start:stop] * radius_squared**-1.5
            force[body, 0] += np.sum(weighted * x)
            force[body, 1] += np.sum(weighted * y)
            force[body, 2] += np.sum(weighted * z)
        force[body] *= body_mass * cell_volume
    return force
