"""No-extrapolation radial profiles for the kpc-to-pc bridge."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator

from .constants import G_INTERNAL
from .fdm_outer_halo import FDMOuterHaloClosure


def _array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size < 2 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional array with at least two values")
    return result


@dataclass(frozen=True)
class TabulatedSphericalProfile:
    """A spherical profile that raises instead of extrapolating."""

    radii_pc: np.ndarray
    density_msun_pc3: np.ndarray
    enclosed_mass_msun: np.ndarray
    potential_pc2_myr2: np.ndarray
    core_radius_pc: float | None = None
    virial_binding_energy_msun_pc2_myr2: float | None = None

    def __post_init__(self) -> None:
        radii = _array(self.radii_pc, "radii_pc")
        density = _array(self.density_msun_pc3, "density_msun_pc3")
        enclosed = _array(self.enclosed_mass_msun, "enclosed_mass_msun")
        potential = _array(self.potential_pc2_myr2, "potential_pc2_myr2")
        if not (
            radii.size == density.size == enclosed.size == potential.size
        ):
            raise ValueError("tabulated profile arrays must have equal lengths")
        if np.any(radii <= 0.0) or np.any(np.diff(radii) <= 0.0):
            raise ValueError("profile radii must be strictly increasing and positive")
        if np.any(density < 0.0) or np.any(enclosed < 0.0):
            raise ValueError("density and enclosed mass must be non-negative")
        if np.any(np.diff(enclosed) < -1.0e-12 * max(enclosed[-1], 1.0)):
            raise ValueError("enclosed mass must be non-decreasing")
        core = (
            None
            if self.core_radius_pc is None
            else float(self.core_radius_pc)
        )
        if core is not None and (not math.isfinite(core) or core <= 0.0):
            raise ValueError("core_radius_pc must be finite and positive")
        binding = self.virial_binding_energy_msun_pc2_myr2
        if binding is not None and not math.isfinite(float(binding)):
            raise ValueError("virial binding energy must be finite")
        object.__setattr__(self, "radii_pc", radii)
        object.__setattr__(self, "density_msun_pc3", density)
        object.__setattr__(self, "enclosed_mass_msun", enclosed)
        object.__setattr__(self, "potential_pc2_myr2", potential)
        object.__setattr__(self, "core_radius_pc", core)
        object.__setattr__(self, "virial_binding_energy_msun_pc2_myr2", binding)
        object.__setattr__(
            self,
            "_density_interpolator",
            PchipInterpolator(radii, density, extrapolate=False),
        )
        object.__setattr__(
            self,
            "_enclosed_interpolator",
            PchipInterpolator(radii, enclosed, extrapolate=False),
        )
        object.__setattr__(
            self,
            "_potential_interpolator",
            PchipInterpolator(radii, potential, extrapolate=False),
        )

    @property
    def total_mass_msun(self) -> float:
        return float(self.enclosed_mass_msun[-1])

    def _evaluate(self, name: str, radius_pc: float | np.ndarray) -> float | np.ndarray:
        radius = np.asarray(radius_pc, dtype=float)
        if np.any(~np.isfinite(radius)) or np.any(radius < self.radii_pc[0]) or np.any(radius > self.radii_pc[-1]):
            raise ValueError(
                f"{name} requested outside tabulated support "
                f"[{self.radii_pc[0]}, {self.radii_pc[-1]}] pc"
            )
        interpolator = getattr(self, f"_{name}_interpolator")
        values = np.asarray(interpolator(radius), dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} interpolation produced non-finite values")
        if name in {"density", "enclosed"} and np.any(values < -1.0e-10 * max(self.total_mass_msun, 1.0)):
            raise ValueError(f"{name} interpolation produced negative values")
        values = np.maximum(values, 0.0) if name in {"density", "enclosed"} else values
        return float(values) if np.ndim(radius_pc) == 0 else values

    def density(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        return self._evaluate("density", radius_pc)

    def enclosed_mass(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        return self._evaluate("enclosed", radius_pc)

    def potential(self, radius_pc: float | np.ndarray) -> float | np.ndarray:
        return self._evaluate("potential", radius_pc)

    def acceleration(self, position_pc: np.ndarray) -> np.ndarray:
        position = np.asarray(position_pc, dtype=float)
        if position.shape != (3,) or np.any(~np.isfinite(position)):
            raise ValueError("position must be a finite three-vector")
        radius = float(np.linalg.norm(position))
        if radius == 0.0:
            return np.zeros(3)
        enclosed = float(self.enclosed_mass(radius))
        return -G_INTERNAL * enclosed * position / radius**3

    def virial_binding_energy(self) -> float:
        if self.virial_binding_energy_msun_pc2_myr2 is None:
            raise ValueError("profile does not provide a virial binding energy")
        return float(self.virial_binding_energy_msun_pc2_myr2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "radii_pc": self.radii_pc.tolist(),
            "density_msun_pc3": self.density_msun_pc3.tolist(),
            "enclosed_mass_msun": self.enclosed_mass_msun.tolist(),
            "potential_pc2_myr2": self.potential_pc2_myr2.tolist(),
            "core_radius_pc": self.core_radius_pc,
            "virial_binding_energy_msun_pc2_myr2": self.virial_binding_energy_msun_pc2_myr2,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "TabulatedSphericalProfile":
        if not isinstance(record, Mapping):
            raise ValueError("radial profile must be an object")
        return cls(
            radii_pc=record.get("radii_pc"),
            density_msun_pc3=record.get("density_msun_pc3"),
            enclosed_mass_msun=record.get("enclosed_mass_msun"),
            potential_pc2_myr2=record.get("potential_pc2_myr2"),
            core_radius_pc=record.get("core_radius_pc"),
            virial_binding_energy_msun_pc2_myr2=record.get(
                "virial_binding_energy_msun_pc2_myr2"
            ),
        )


@dataclass(frozen=True)
class EnvironmentProfileBundle:
    """Named stellar, gas, and FDM profile tables with provenance."""

    event_uid: str
    stellar: TabulatedSphericalProfile | None
    gas: TabulatedSphericalProfile | None
    fdm: TabulatedSphericalProfile | None
    source_case_id: str
    source_sha256: str
    source_path: str
    outer_fdm_closure: FDMOuterHaloClosure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_uid, str) or not self.event_uid.strip():
            raise ValueError("profile bundle event_uid is required")
        if not isinstance(self.source_case_id, str) or not self.source_case_id.strip():
            raise ValueError("profile bundle source_case_id is required")
        if not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64:
            raise ValueError("profile bundle source_sha256 must be 64 characters")
        if any(character not in "0123456789abcdefABCDEF" for character in self.source_sha256):
            raise ValueError("profile bundle source_sha256 must be hexadecimal")
        if not isinstance(self.source_path, str) or not self.source_path.strip():
            raise ValueError("profile bundle source_path is required")

    def profile(self, channel: str) -> TabulatedSphericalProfile | None:
        if channel not in {"stellar", "gas", "fdm"}:
            raise KeyError(channel)
        return getattr(self, channel)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event_uid": self.event_uid,
            "profiles": {
                name: None if self.profile(name) is None else self.profile(name).as_dict()
                for name in ("stellar", "gas", "fdm")
            },
            "source_case_id": self.source_case_id,
            "source_sha256": self.source_sha256,
            "source_path": self.source_path,
            "outer_fdm_closure": (
                None
                if self.outer_fdm_closure is None
                else self.outer_fdm_closure.as_dict()
            ),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "EnvironmentProfileBundle":
        if not isinstance(record, Mapping) or record.get("schema_version") != 1:
            raise ValueError("unsupported environment profile bundle schema")
        profiles = record.get("profiles")
        if not isinstance(profiles, Mapping):
            raise ValueError("profile bundle lacks profiles")
        return cls(
            event_uid=record.get("event_uid", ""),
            stellar=(
                None
                if profiles.get("stellar") is None
                else TabulatedSphericalProfile.from_dict(profiles["stellar"])
            ),
            gas=(
                None
                if profiles.get("gas") is None
                else TabulatedSphericalProfile.from_dict(profiles["gas"])
            ),
            fdm=(
                None
                if profiles.get("fdm") is None
                else TabulatedSphericalProfile.from_dict(profiles["fdm"])
            ),
            source_case_id=record.get("source_case_id", ""),
            source_sha256=record.get("source_sha256", ""),
            source_path=record.get("source_path", ""),
            outer_fdm_closure=(
                None
                if record.get("outer_fdm_closure") is None
                else FDMOuterHaloClosure.from_dict(record["outer_fdm_closure"])
            ),
        )
