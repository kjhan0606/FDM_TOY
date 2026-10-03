"""Spherical host profiles and unresolved nuclear backgrounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .constants import G_INTERNAL
from .soliton import SphericalSoliton


class SphericalMassProfile(Protocol):
    def density(self, radius_pc: float | np.ndarray) -> float | np.ndarray: ...

    def enclosed_mass(self, radius_pc: float | np.ndarray) -> float | np.ndarray: ...

    def potential(self, radius_pc: float | np.ndarray) -> float | np.ndarray: ...

    def acceleration(self, position_pc: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class DehnenProfile:
    """Finite-mass Dehnen profile with inner density slope ``gamma``."""

    total_mass_msun: float
    scale_radius_pc: float
    gamma: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.total_mass_msun, self.scale_radius_pc, self.gamma], dtype=float
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("Dehnen profile parameters must be finite")
        if self.total_mass_msun <= 0.0 or self.scale_radius_pc <= 0.0:
            raise ValueError("Dehnen mass and scale radius must be positive")
        if not 0.0 <= self.gamma < 3.0:
            raise ValueError("Dehnen gamma must satisfy 0 <= gamma < 3")

    def enclosed_mass(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        radius = np.asarray(radius_pc, dtype=float)
        if np.any(radius < 0.0):
            raise ValueError("radius must be non-negative")
        fraction = (radius / (radius + self.scale_radius_pc)) ** (3.0 - self.gamma)
        result = self.total_mass_msun * fraction
        return float(result) if np.ndim(radius_pc) == 0 else result

    def density(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        radius = np.asarray(radius_pc, dtype=float)
        if np.any(radius < 0.0):
            raise ValueError("radius must be non-negative")
        prefactor = (
            (3.0 - self.gamma)
            * self.total_mass_msun
            * self.scale_radius_pc
            / (4.0 * np.pi)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            result = prefactor / (
                radius**self.gamma
                * (radius + self.scale_radius_pc) ** (4.0 - self.gamma)
            )
        if self.gamma == 0.0:
            result = np.where(
                radius == 0.0,
                3.0 * self.total_mass_msun / (4.0 * np.pi * self.scale_radius_pc**3),
                result,
            )
        return float(result) if np.ndim(radius_pc) == 0 else result

    def potential(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        radius = np.asarray(radius_pc, dtype=float)
        if np.any(radius < 0.0):
            raise ValueError("radius must be non-negative")
        x = radius / (radius + self.scale_radius_pc)
        if np.isclose(self.gamma, 2.0, rtol=0.0, atol=1.0e-14):
            with np.errstate(divide="ignore"):
                result = G_INTERNAL * self.total_mass_msun / self.scale_radius_pc * np.log(x)
        else:
            result = (
                -G_INTERNAL
                * self.total_mass_msun
                / (self.scale_radius_pc * (2.0 - self.gamma))
                * (1.0 - x ** (2.0 - self.gamma))
            )
        return float(result) if np.ndim(radius_pc) == 0 else result

    def acceleration(self, position_pc: np.ndarray) -> np.ndarray:
        position = np.asarray(position_pc, dtype=float)
        if position.shape != (3,) or np.any(~np.isfinite(position)):
            raise ValueError("position must be a finite three-vector")
        radius = float(np.linalg.norm(position))
        if radius == 0.0:
            return np.zeros(3)
        return -G_INTERNAL * self.enclosed_mass(radius) * position / radius**3

    def logarithmic_enclosed_mass_slope(self, radius_pc: float) -> float:
        if not np.isfinite(radius_pc) or radius_pc <= 0.0:
            raise ValueError("radius must be finite and positive")
        return float(
            (3.0 - self.gamma)
            * self.scale_radius_pc
            / (radius_pc + self.scale_radius_pc)
        )


@dataclass(frozen=True)
class CompositePotential:
    components: tuple[SphericalMassProfile, ...]
    central_point_mass_msun: float = 0.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.central_point_mass_msun)
            or self.central_point_mass_msun < 0.0
        ):
            raise ValueError("central point mass must be finite and non-negative")
        if not self.components and self.central_point_mass_msun == 0.0:
            raise ValueError("a composite potential cannot be empty")

    def density(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        radius = np.asarray(radius_pc, dtype=float)
        result = np.zeros_like(radius)
        for component in self.components:
            result = result + np.asarray(component.density(radius), dtype=float)
        return float(result) if np.ndim(radius_pc) == 0 else result

    def enclosed_mass(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        radius = np.asarray(radius_pc, dtype=float)
        if np.any(radius < 0.0):
            raise ValueError("radius must be non-negative")
        result = np.full_like(radius, self.central_point_mass_msun)
        for component in self.components:
            result = result + np.asarray(component.enclosed_mass(radius), dtype=float)
        return float(result) if np.ndim(radius_pc) == 0 else result

    def potential(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        radius = np.asarray(radius_pc, dtype=float)
        if np.any(radius < 0.0):
            raise ValueError("radius must be non-negative")
        result = np.zeros_like(radius)
        for component in self.components:
            result = result + np.asarray(component.potential(radius), dtype=float)
        if self.central_point_mass_msun > 0.0:
            with np.errstate(divide="ignore"):
                result = result - G_INTERNAL * self.central_point_mass_msun / radius
        return float(result) if np.ndim(radius_pc) == 0 else result

    def acceleration(self, position_pc: np.ndarray) -> np.ndarray:
        position = np.asarray(position_pc, dtype=float)
        if position.shape != (3,) or np.any(~np.isfinite(position)):
            raise ValueError("position must be a finite three-vector")
        radius = float(np.linalg.norm(position))
        if radius == 0.0:
            return np.zeros(3)
        return -G_INTERNAL * self.enclosed_mass(radius) * position / radius**3

    def logarithmic_enclosed_mass_slope(self, radius_pc: float) -> float:
        if not np.isfinite(radius_pc) or radius_pc <= 0.0:
            raise ValueError("radius must be finite and positive")
        enclosed = float(self.enclosed_mass(radius_pc))
        if enclosed <= 0.0:
            return 0.0
        derivative_sum = 0.0
        for component in self.components:
            mass = float(component.enclosed_mass(radius_pc))
            if mass <= 0.0:
                continue
            if hasattr(component, "logarithmic_enclosed_mass_slope"):
                slope = float(component.logarithmic_enclosed_mass_slope(radius_pc))
            else:
                step = max(1.0e-6 * radius_pc, 1.0e-10)
                lower = max(radius_pc - step, 0.5 * radius_pc)
                upper = radius_pc + step
                lower_mass = float(component.enclosed_mass(lower))
                upper_mass = float(component.enclosed_mass(upper))
                slope = (
                    np.log(upper_mass) - np.log(lower_mass)
                ) / (np.log(upper) - np.log(lower))
            derivative_sum += mass * slope
        return float(derivative_sum / enclosed)


@dataclass(frozen=True)
class StellarBackground:
    profile: SphericalMassProfile
    velocity_dispersion_pc_myr: float
    bulk_velocity_pc_myr: np.ndarray

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.velocity_dispersion_pc_myr)
            or self.velocity_dispersion_pc_myr <= 0.0
        ):
            raise ValueError("stellar velocity dispersion must be positive")
        velocity = np.asarray(self.bulk_velocity_pc_myr, dtype=float)
        if velocity.shape != (3,) or np.any(~np.isfinite(velocity)):
            raise ValueError("stellar bulk velocity must be a finite three-vector")
        object.__setattr__(self, "bulk_velocity_pc_myr", velocity)


@dataclass(frozen=True)
class GasBackground:
    profile: SphericalMassProfile
    sound_speed_pc_myr: float
    rotation_fraction: float = 0.0
    rotation_axis: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.sound_speed_pc_myr) or self.sound_speed_pc_myr <= 0.0:
            raise ValueError("gas sound speed must be finite and positive")
        if not np.isfinite(self.rotation_fraction) or not 0.0 <= self.rotation_fraction <= 1.0:
            raise ValueError("gas rotation fraction must lie in [0, 1]")
        axis = np.array([0.0, 0.0, 1.0]) if self.rotation_axis is None else np.asarray(
            self.rotation_axis, dtype=float
        )
        norm = float(np.linalg.norm(axis))
        if axis.shape != (3,) or np.any(~np.isfinite(axis)) or norm == 0.0:
            raise ValueError("gas rotation axis must be a nonzero finite vector")
        object.__setattr__(self, "rotation_axis", axis / norm)

    def bulk_velocity(self, position_pc: np.ndarray, enclosed_mass_msun: float) -> np.ndarray:
        position = np.asarray(position_pc, dtype=float)
        radius = float(np.linalg.norm(position))
        tangent = np.cross(self.rotation_axis, position)
        tangent_norm = float(np.linalg.norm(tangent))
        if radius == 0.0 or tangent_norm == 0.0 or self.rotation_fraction == 0.0:
            return np.zeros(3)
        circular_speed = np.sqrt(G_INTERNAL * enclosed_mass_msun / radius)
        return self.rotation_fraction * circular_speed * tangent / tangent_norm


@dataclass(frozen=True)
class FDMBackground:
    soliton: SphericalSoliton
    particle_mass_ev: float
    alpha_df: float
    bulk_velocity_pc_myr: np.ndarray
    velocity_floor_pc_myr: float = 1.0e-12
    mode: str = "analytic_unresolved"

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.particle_mass_ev, self.alpha_df, self.velocity_floor_pc_myr],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("FDM drag parameters must be finite and positive")
        velocity = np.asarray(self.bulk_velocity_pc_myr, dtype=float)
        if velocity.shape != (3,) or np.any(~np.isfinite(velocity)):
            raise ValueError("FDM bulk velocity must be a finite three-vector")
        if self.mode not in {"analytic_unresolved", "live_resolved"}:
            raise ValueError("FDM mode must be analytic_unresolved or live_resolved")
        object.__setattr__(self, "bulk_velocity_pc_myr", velocity)


@dataclass(frozen=True)
class NuclearEnvelope:
    """Hernquist nuclear envelope carried by the secondary SMBH."""

    initial_bound_mass_msun: float
    scale_radius_pc: float
    initial_truncation_radius_pc: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.initial_bound_mass_msun,
                self.scale_radius_pc,
                self.initial_truncation_radius_pc,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("nuclear envelope parameters must be positive")

    @property
    def asymptotic_mass_msun(self) -> float:
        fraction = (
            self.initial_truncation_radius_pc
            / (self.initial_truncation_radius_pc + self.scale_radius_pc)
        ) ** 2
        return float(self.initial_bound_mass_msun / fraction)

    def mass_within(self, radius_pc: float) -> float:
        if not np.isfinite(radius_pc) or radius_pc < 0.0:
            raise ValueError("envelope radius must be finite and non-negative")
        clipped = min(radius_pc, self.initial_truncation_radius_pc)
        return float(
            self.asymptotic_mass_msun
            * (clipped / (clipped + self.scale_radius_pc)) ** 2
        )
