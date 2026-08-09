"""Pure-Python audit helpers for the lagRamses numerical sink merger."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import G_INTERNAL


def numerical_merge_radius_pc(
    *, box_size_mpc_h: float, levelmax: int, hubble_h: float, rmerge: float
) -> float:
    """Physical lagRamses FOF linking radius for a cosmological run.

    ``box_size_mpc_h`` is ``boxlen_ini`` in comoving Mpc/h. The scale factor
    cancels between ``dx_min`` and ``scale_l`` in the current implementation.
    """

    if box_size_mpc_h <= 0.0 or hubble_h <= 0.0 or rmerge <= 0.0:
        raise ValueError("box size, hubble_h, and rmerge must be positive")
    if isinstance(levelmax, bool) or not isinstance(levelmax, int) or levelmax < 1:
        raise ValueError("levelmax must be a positive integer")
    return float(rmerge * box_size_mpc_h / hubble_h * 1.0e6 / 2**levelmax)


def minimum_image(displacement: np.ndarray, box_size: float | np.ndarray) -> np.ndarray:
    """Apply the periodic minimum-image convention."""

    displacement = np.asarray(displacement, dtype=float)
    box = np.asarray(box_size, dtype=float)
    if np.any(box <= 0.0):
        raise ValueError("box size must be positive")
    return displacement - box * np.floor(displacement / box + 0.5)


def com_kinetic_energy(
    mass1: float, mass2: float, velocity1: np.ndarray, velocity2: np.ndarray
) -> float:
    """Pair kinetic energy in the centre-of-mass frame."""

    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("masses must be positive")
    v1 = np.asarray(velocity1, dtype=float)
    v2 = np.asarray(velocity2, dtype=float)
    vcom = (mass1 * v1 + mass2 * v2) / (mass1 + mass2)
    return float(0.5 * mass1 * np.sum((v1 - vcom) ** 2) + 0.5 * mass2 * np.sum((v2 - vcom) ** 2))


def physically_bound_pair(
    mass1_msun: float,
    mass2_msun: float,
    separation_pc: float,
    velocity1_pc_myr: np.ndarray,
    velocity2_pc_myr: np.ndarray,
) -> bool:
    """Standard Newtonian two-body binding check in physical units."""

    kinetic = com_kinetic_energy(
        mass1_msun, mass2_msun, velocity1_pc_myr, velocity2_pc_myr
    )
    potential_magnitude = G_INTERNAL * mass1_msun * mass2_msun / separation_pc
    return bool(kinetic < potential_magnitude)


def legacy_source_binding_proxy(
    mass1_code: float,
    mass2_code: float,
    squared_separation_code: float,
    velocity1_code: np.ndarray,
    velocity2_code: np.ndarray,
    fact_g: float,
) -> tuple[float, float, bool]:
    """Reproduce the current source expression exactly for auditing.

    The code uses ``m1*m2*fact_g/rr`` where ``rr`` is squared distance. This is
    intentionally not relabelled as physical gravitational potential energy;
    the conventional expression would divide by ``sqrt(rr)``.
    """

    if squared_separation_code <= 0.0:
        raise ValueError("squared separation must be positive")
    kinetic = com_kinetic_energy(
        mass1_code, mass2_code, velocity1_code, velocity2_code
    )
    proxy = mass1_code * mass2_code * fact_g / squared_separation_code
    return kinetic, float(proxy), bool(kinetic < proxy)


@dataclass(frozen=True)
class NumericalMergeScale:
    box_size_mpc_h: float
    levelmax: int
    hubble_h: float
    rmerge: float

    @property
    def cell_size_pc(self) -> float:
        return numerical_merge_radius_pc(
            box_size_mpc_h=self.box_size_mpc_h,
            levelmax=self.levelmax,
            hubble_h=self.hubble_h,
            rmerge=1.0,
        )

    @property
    def merge_radius_pc(self) -> float:
        return numerical_merge_radius_pc(
            box_size_mpc_h=self.box_size_mpc_h,
            levelmax=self.levelmax,
            hubble_h=self.hubble_h,
            rmerge=self.rmerge,
        )
