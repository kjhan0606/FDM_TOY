"""Static spherical soliton density and potential models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.integrate import quad
from scipy.special import beta, betainc

from .constants import G_INTERNAL, SCHIVE_A


class SphericalSoliton(Protocol):
    """Interface consumed by the orbit integrator."""

    core_radius_pc: float
    total_mass_msun: float

    def density(self, radius_pc: float | np.ndarray) -> float | np.ndarray: ...

    def enclosed_mass(self, radius_pc: float | np.ndarray) -> float | np.ndarray: ...

    def potential(self, radius_pc: float | np.ndarray) -> float | np.ndarray: ...

    def acceleration(self, position_pc: np.ndarray) -> np.ndarray: ...

    def virial_binding_energy(self) -> float: ...


@dataclass(frozen=True)
class SchiveSoliton:
    """The spherical ``[1 + 0.091 (r/rc)^2]^-8`` profile.

    ``configured_mass_msun`` follows ``mass_definition``. The actual finite
    total profile mass is exposed as ``total_mass_msun``.
    """

    core_radius_pc: float
    central_density_msun_pc3: float
    configured_mass_msun: float
    mass_definition: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.core_radius_pc, "core_radius_pc"),
            (self.central_density_msun_pc3, "central_density_msun_pc3"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.mass_definition not in {"total_profile", "within_rc", "central_density"}:
            raise ValueError(f"unsupported mass_definition={self.mass_definition!r}")

    @staticmethod
    def _dimensionless_integral_total() -> float:
        return float(beta(1.5, 6.5) / (2.0 * SCHIVE_A**1.5))

    @staticmethod
    def _mass_fraction(radius_over_rc: float | np.ndarray) -> float | np.ndarray:
        x = np.asarray(radius_over_rc, dtype=float)
        y = SCHIVE_A * x * x / (1.0 + SCHIVE_A * x * x)
        fraction = betainc(1.5, 6.5, y)
        if np.ndim(radius_over_rc) == 0:
            return float(fraction)
        return fraction

    @classmethod
    def from_mass(
        cls, mass_msun: float, core_radius_pc: float, mass_definition: str
    ) -> "SchiveSoliton":
        if not np.isfinite(mass_msun) or mass_msun <= 0.0:
            raise ValueError("mass_msun must be finite and positive")
        if mass_definition == "total_profile":
            normalization_mass = mass_msun
        elif mass_definition == "within_rc":
            fraction_rc = float(cls._mass_fraction(1.0))
            normalization_mass = mass_msun / fraction_rc
        else:
            raise ValueError("mass_definition must be 'total_profile' or 'within_rc'")
        integral = cls._dimensionless_integral_total()
        rho0 = normalization_mass / (4.0 * np.pi * core_radius_pc**3 * integral)
        return cls(core_radius_pc, rho0, mass_msun, mass_definition)

    @classmethod
    def from_central_density(
        cls, central_density_msun_pc3: float, core_radius_pc: float
    ) -> "SchiveSoliton":
        integral = cls._dimensionless_integral_total()
        total_mass = (
            4.0
            * np.pi
            * central_density_msun_pc3
            * core_radius_pc**3
            * integral
        )
        return cls(
            core_radius_pc,
            central_density_msun_pc3,
            total_mass,
            "central_density",
        )

    @property
    def total_mass_msun(self) -> float:
        integral = self._dimensionless_integral_total()
        return float(
            4.0
            * np.pi
            * self.central_density_msun_pc3
            * self.core_radius_pc**3
            * integral
        )

    def density(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        radius = np.asarray(radius_pc, dtype=float)
        if np.any(radius < 0.0):
            raise ValueError("radius must be non-negative")
        result = self.central_density_msun_pc3 * (
            1.0 + SCHIVE_A * (radius / self.core_radius_pc) ** 2
        ) ** -8
        if np.ndim(radius_pc) == 0:
            return float(result)
        return result

    def enclosed_mass(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        radius = np.asarray(radius_pc, dtype=float)
        if np.any(radius < 0.0):
            raise ValueError("radius must be non-negative")
        result = self.total_mass_msun * self._mass_fraction(radius / self.core_radius_pc)
        if np.ndim(radius_pc) == 0:
            return float(result)
        return result

    def potential(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        """Potential with zero at infinity, in pc^2/Myr^2."""

        radius = np.asarray(radius_pc, dtype=float)
        if np.any(radius < 0.0):
            raise ValueError("radius must be non-negative")
        x = radius / self.core_radius_pc
        enclosed = np.asarray(self.enclosed_mass(radius), dtype=float)
        first = np.divide(enclosed, radius, out=np.zeros_like(radius), where=radius > 0.0)
        outer_shell = (
            4.0
            * np.pi
            * self.central_density_msun_pc3
            * self.core_radius_pc**2
            / (14.0 * SCHIVE_A)
            * (1.0 + SCHIVE_A * x * x) ** -7
        )
        result = -G_INTERNAL * (first + outer_shell)
        if np.ndim(radius_pc) == 0:
            return float(result)
        return result

    def acceleration(self, position_pc: np.ndarray) -> np.ndarray:
        position = np.asarray(position_pc, dtype=float)
        radius = float(np.linalg.norm(position))
        if radius == 0.0:
            return np.zeros(3, dtype=float)
        return -G_INTERNAL * self.enclosed_mass(radius) * position / radius**3

    def virial_binding_energy(self) -> float:
        """Positive proxy for the energy required to unbind the soliton.

        For a virialized self-gravitating profile this is ``|W|/2``, where
        ``W = -4 pi G integral(rho M_enc r dr)``. It is a backreaction scale,
        not a substitute for a Schrödinger--Poisson energy calculation.
        """

        def integrand(x: float) -> float:
            density_shape = (1.0 + SCHIVE_A * x * x) ** -8
            mass = self.total_mass_msun * float(self._mass_fraction(x))
            return density_shape * mass * x

        dimensionless, _ = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1.0e-10)
        gravitational_energy = (
            -4.0
            * np.pi
            * G_INTERNAL
            * self.central_density_msun_pc3
            * self.core_radius_pc**2
            * dimensionless
        )
        return float(0.5 * abs(gravitational_energy))
