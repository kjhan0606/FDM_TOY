"""Pure-Python audit helpers for the lagRamses numerical sink merger.

The functions in this module do not modify or call lagRamses.  They translate
the numerical linking scale and a pre-compaction two-sink record into physical
binary initial conditions for the unresolved calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

from .constants import G_INTERNAL
from .orbital_exchange import keplerian_elements_from_relative_state


ACTIVE_SOURCE_DEFAULT_RMERGE = 1.0


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

    if separation_pc <= 0.0:
        raise ValueError("separation_pc must be positive")

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


@dataclass(frozen=True)
class RamsesRunCaptureBoundary:
    """Resolved numerical linking scale and its provenance."""

    box_size_mpc_h: float
    levelmax: int
    hubble_h: float
    rmerge: float
    rmerge_origin: str

    @property
    def scale(self) -> NumericalMergeScale:
        return NumericalMergeScale(
            self.box_size_mpc_h, self.levelmax, self.hubble_h, self.rmerge
        )

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "box_size_mpc_h": self.box_size_mpc_h,
            "levelmax": self.levelmax,
            "hubble_h": self.hubble_h,
            "rmerge": self.rmerge,
            "rmerge_origin": self.rmerge_origin,
            "finest_cell_size_pc": self.scale.cell_size_pc,
            "numerical_capture_radius_pc": self.scale.merge_radius_pc,
        }


@dataclass(frozen=True)
class PairOrbitalState:
    """Osculating two-body state at a numerical binary capture.

    The Kepler elements use only the mutual SMBH potential.  Gas, stars, and
    the FDM potential must be retained separately as environmental quantities;
    therefore these are initial osculating elements rather than constants of
    the subsequent motion.
    """

    member_ids: tuple[int, int]
    masses_msun: tuple[float, float]
    separation_vector_pc: np.ndarray
    relative_velocity_pc_myr: np.ndarray
    centre_of_mass_position_pc: np.ndarray
    centre_of_mass_velocity_pc_myr: np.ndarray
    specific_energy_pc2_myr2: float
    orbital_energy_msun_pc2_myr2: float
    specific_angular_momentum_pc2_myr: np.ndarray
    angular_momentum_msun_pc2_myr: np.ndarray
    eccentricity_vector: np.ndarray
    eccentricity: float
    semi_major_axis_pc: float | None
    pericentre_pc: float | None
    apocentre_pc: float | None

    @property
    def separation_pc(self) -> float:
        return float(np.linalg.norm(self.separation_vector_pc))

    @property
    def mass_ratio(self) -> float:
        low, high = sorted(self.masses_msun)
        return low / high

    @property
    def reduced_mass_msun(self) -> float:
        m1, m2 = self.masses_msun
        return m1 * m2 / (m1 + m2)

    @property
    def bound(self) -> bool:
        return self.specific_energy_pc2_myr2 < 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_ids": list(self.member_ids),
            "masses_msun": list(self.masses_msun),
            "mass_ratio_q": self.mass_ratio,
            "reduced_mass_msun": self.reduced_mass_msun,
            "separation_pc": self.separation_pc,
            "separation_vector_pc": self.separation_vector_pc.tolist(),
            "relative_velocity_pc_myr": self.relative_velocity_pc_myr.tolist(),
            "centre_of_mass_position_pc": self.centre_of_mass_position_pc.tolist(),
            "centre_of_mass_velocity_pc_myr": self.centre_of_mass_velocity_pc_myr.tolist(),
            "specific_energy_pc2_myr2": self.specific_energy_pc2_myr2,
            "orbital_energy_msun_pc2_myr2": self.orbital_energy_msun_pc2_myr2,
            "specific_angular_momentum_pc2_myr": self.specific_angular_momentum_pc2_myr.tolist(),
            "angular_momentum_msun_pc2_myr": self.angular_momentum_msun_pc2_myr.tolist(),
            "eccentricity_vector": self.eccentricity_vector.tolist(),
            "eccentricity": self.eccentricity,
            "semi_major_axis_pc": self.semi_major_axis_pc,
            "pericentre_pc": self.pericentre_pc,
            "apocentre_pc": self.apocentre_pc,
            "physically_bound_two_body": self.bound,
        }


def pair_orbital_state(
    *,
    member_ids: tuple[int, int],
    masses_msun: tuple[float, float],
    positions_pc: np.ndarray,
    velocities_pc_myr: np.ndarray,
    periodic_box_pc: float | np.ndarray | None = None,
) -> PairOrbitalState:
    """Convert a pre-compaction two-sink record to osculating elements."""

    masses = np.asarray(masses_msun, dtype=float)
    positions = np.asarray(positions_pc, dtype=float)
    velocities = np.asarray(velocities_pc_myr, dtype=float)
    if len(member_ids) != 2 or masses.shape != (2,):
        raise ValueError("a binary capture must contain exactly two members")
    if positions.shape != (2, 3) or velocities.shape != (2, 3):
        raise ValueError("positions and velocities must have shape (2, 3)")
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("SMBH masses must be finite and positive")
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(velocities)):
        raise ValueError("positions and velocities must be finite")

    displacement = positions[0] - positions[1]
    unwrapped_position1 = positions[0]
    if periodic_box_pc is not None:
        displacement = minimum_image(displacement, periodic_box_pc)
        unwrapped_position1 = positions[1] + displacement
    separation = float(np.linalg.norm(displacement))
    if separation <= 0.0:
        raise ValueError("the two SMBHs must have non-zero separation")

    relative_velocity = velocities[0] - velocities[1]
    total_mass = float(np.sum(masses))
    reduced_mass = float(np.prod(masses) / total_mass)
    centre_position = (
        masses[0] * unwrapped_position1 + masses[1] * positions[1]
    ) / total_mass
    if periodic_box_pc is not None:
        centre_position = np.mod(centre_position, np.asarray(periodic_box_pc, dtype=float))
    centre_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass

    elements = keplerian_elements_from_relative_state(
        total_mass=total_mass,
        displacement=displacement,
        relative_velocity=relative_velocity,
    )
    angular_momentum = reduced_mass * elements.specific_angular_momentum
    if elements.semimajor_axis is not None:
        pericentre = elements.semimajor_axis * (1.0 - elements.eccentricity)
        apocentre = elements.semimajor_axis * (1.0 + elements.eccentricity)
    else:
        pericentre = None
        apocentre = None

    return PairOrbitalState(
        member_ids=member_ids,
        masses_msun=(float(masses[0]), float(masses[1])),
        separation_vector_pc=displacement,
        relative_velocity_pc_myr=relative_velocity,
        centre_of_mass_position_pc=centre_position,
        centre_of_mass_velocity_pc_myr=centre_velocity,
        specific_energy_pc2_myr2=elements.specific_energy,
        orbital_energy_msun_pc2_myr2=reduced_mass * elements.specific_energy,
        specific_angular_momentum_pc2_myr=elements.specific_angular_momentum,
        angular_momentum_msun_pc2_myr=angular_momentum,
        eccentricity_vector=elements.eccentricity_vector,
        eccentricity=elements.eccentricity,
        semi_major_axis_pc=elements.semimajor_axis,
        pericentre_pc=pericentre,
        apocentre_pc=apocentre,
    )


_ASSIGNMENT = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z0-9_]*)[ \t]*=[ \t]*([^!,/\r\n]+)",
    re.MULTILINE,
)


def read_fortran_assignments(path: str | Path) -> dict[str, str]:
    """Read scalar assignments from an ASCII RAMSES namelist or info file."""

    text = Path(path).expanduser().read_text(encoding="utf-8", errors="ignore")
    return {key.lower(): value.strip() for key, value in _ASSIGNMENT.findall(text)}


def _fortran_float(value: str, field: str) -> float:
    try:
        result = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ValueError(f"cannot parse {field}={value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def capture_boundary_from_files(
    *,
    info_path: str | Path,
    box_size_mpc_h: float,
    namelist_path: str | Path | None = None,
    rmerge_override: float | None = None,
) -> RamsesRunCaptureBoundary:
    """Resolve the numerical capture boundary from archived run metadata.

    RAMSES ``info_*.txt`` records ``levelmax`` and ``H0`` but not the initial
    cosmological box length in Mpc/h.  The latter is therefore an explicit
    argument and should come from the initial-condition header or startup log.
    """

    info = read_fortran_assignments(info_path)
    if "levelmax" not in info or "h0" not in info:
        raise ValueError("info file must contain levelmax and H0")
    level_value = _fortran_float(info["levelmax"], "levelmax")
    levelmax = int(level_value)
    if level_value != levelmax:
        raise ValueError("levelmax must be an integer")
    hubble_h = _fortran_float(info["h0"], "H0") / 100.0

    if rmerge_override is not None:
        rmerge = float(rmerge_override)
        origin = "command_line_override"
    elif namelist_path is not None:
        namelist = read_fortran_assignments(namelist_path)
        if "rmerge" in namelist:
            rmerge = _fortran_float(namelist["rmerge"], "rmerge")
            origin = "namelist"
        else:
            rmerge = ACTIVE_SOURCE_DEFAULT_RMERGE
            origin = "active_source_default"
    else:
        rmerge = ACTIVE_SOURCE_DEFAULT_RMERGE
        origin = "active_source_default"

    boundary = RamsesRunCaptureBoundary(
        box_size_mpc_h=float(box_size_mpc_h),
        levelmax=levelmax,
        hubble_h=hubble_h,
        rmerge=rmerge,
        rmerge_origin=origin,
    )
    # Trigger all validation in NumericalMergeScale before returning.
    _ = boundary.scale.merge_radius_pc
    return boundary
