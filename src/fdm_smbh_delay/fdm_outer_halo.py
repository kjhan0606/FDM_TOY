"""Versioned radial diagnostics for an outer pure-FDM halo.

These data are a closure contract only.  They do not assert a dynamical
friction law; an outer-halo drift/diffusion model must be calibrated against
the live wave data before it can consume them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator


def _finite_array(value: Any, name: str, *, size: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size < 2 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional array with at least two values")
    if size is not None and result.size != size:
        raise ValueError(f"{name} must have length {size}")
    return result


@dataclass(frozen=True)
class FDMOuterHaloClosure:
    """Radial wave diagnostics needed before an outer-FDM closure is usable.

    ``mass_current_msun_pc2_myr`` is the Madelung mass current, not a
    dimensionless phase gradient.  ``velocity_diffusion_pc2_myr3`` is the
    coefficient of a velocity-space diffusion process.  Both are stored as
    measured diagnostics; no stochastic evolution is implied by this class.
    """

    radii_pc: np.ndarray
    mass_current_msun_pc2_myr: np.ndarray
    coherence_time_myr: np.ndarray
    de_broglie_wavelength_pc: np.ndarray
    velocity_diffusion_pc2_myr3: np.ndarray
    density_gradient_scale_pc: np.ndarray
    closure_status: str = "uncalibrated"

    def __post_init__(self) -> None:
        radii = _finite_array(self.radii_pc, "radii_pc")
        if np.any(radii <= 0.0) or np.any(np.diff(radii) <= 0.0):
            raise ValueError("outer-halo radii must be strictly increasing and positive")

        current = np.asarray(self.mass_current_msun_pc2_myr, dtype=float)
        if current.shape != (radii.size, 3) or np.any(~np.isfinite(current)):
            raise ValueError("mass_current_msun_pc2_myr must have shape (N, 3) and be finite")
        coherence = _finite_array(
            self.coherence_time_myr, "coherence_time_myr", size=radii.size
        )
        wavelength = _finite_array(
            self.de_broglie_wavelength_pc,
            "de_broglie_wavelength_pc",
            size=radii.size,
        )
        diffusion = _finite_array(
            self.velocity_diffusion_pc2_myr3,
            "velocity_diffusion_pc2_myr3",
            size=radii.size,
        )
        gradient = _finite_array(
            self.density_gradient_scale_pc,
            "density_gradient_scale_pc",
            size=radii.size,
        )
        if np.any(coherence <= 0.0) or np.any(wavelength <= 0.0):
            raise ValueError("coherence times and de Broglie wavelengths must be positive")
        if np.any(diffusion < 0.0) or np.any(gradient <= 0.0):
            raise ValueError("diffusion must be non-negative and gradient scales positive")
        if self.closure_status not in {"calibrated", "uncalibrated", "censored"}:
            raise ValueError("unsupported outer-halo closure status")

        object.__setattr__(self, "radii_pc", radii)
        object.__setattr__(self, "mass_current_msun_pc2_myr", current)
        object.__setattr__(self, "coherence_time_myr", coherence)
        object.__setattr__(self, "de_broglie_wavelength_pc", wavelength)
        object.__setattr__(self, "velocity_diffusion_pc2_myr3", diffusion)
        object.__setattr__(self, "density_gradient_scale_pc", gradient)
        object.__setattr__(
            self,
            "_current_interpolators",
            tuple(
                PchipInterpolator(radii, current[:, axis], extrapolate=False)
                for axis in range(3)
            ),
        )
        for name, values in (
            ("coherence", coherence),
            ("wavelength", wavelength),
            ("diffusion", diffusion),
            ("gradient", gradient),
        ):
            object.__setattr__(
                self,
                f"_{name}_interpolator",
                PchipInterpolator(radii, values, extrapolate=False),
            )

    def _radius(self, radius_pc: float) -> float:
        radius = float(radius_pc)
        if (
            not np.isfinite(radius)
            or radius < self.radii_pc[0]
            or radius > self.radii_pc[-1]
        ):
            raise ValueError(
                "outer-FDM closure requested outside tabulated support "
                f"[{self.radii_pc[0]}, {self.radii_pc[-1]}] pc"
            )
        return radius

    def evaluate(self, radius_pc: float) -> dict[str, Any]:
        """Return closure diagnostics at a supported radius, without extrapolation."""

        radius = self._radius(radius_pc)
        current = np.array(
            [float(interpolator(radius)) for interpolator in self._current_interpolators],
            dtype=float,
        )
        values = {
            "mass_current_msun_pc2_myr": current,
            "coherence_time_myr": float(self._coherence_interpolator(radius)),
            "de_broglie_wavelength_pc": float(self._wavelength_interpolator(radius)),
            "velocity_diffusion_pc2_myr3": float(self._diffusion_interpolator(radius)),
            "density_gradient_scale_pc": float(self._gradient_interpolator(radius)),
        }
        if any(not np.all(np.isfinite(value)) for value in values.values()):
            raise ValueError("outer-FDM closure interpolation produced non-finite values")
        return values

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "radii_pc": self.radii_pc.tolist(),
            "mass_current_msun_pc2_myr": self.mass_current_msun_pc2_myr.tolist(),
            "coherence_time_myr": self.coherence_time_myr.tolist(),
            "de_broglie_wavelength_pc": self.de_broglie_wavelength_pc.tolist(),
            "velocity_diffusion_pc2_myr3": self.velocity_diffusion_pc2_myr3.tolist(),
            "density_gradient_scale_pc": self.density_gradient_scale_pc.tolist(),
            "closure_status": self.closure_status,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "FDMOuterHaloClosure":
        if not isinstance(record, Mapping) or record.get("schema_version") != 1:
            raise ValueError("unsupported outer-FDM closure schema")
        return cls(
            radii_pc=record.get("radii_pc"),
            mass_current_msun_pc2_myr=record.get("mass_current_msun_pc2_myr"),
            coherence_time_myr=record.get("coherence_time_myr"),
            de_broglie_wavelength_pc=record.get("de_broglie_wavelength_pc"),
            velocity_diffusion_pc2_myr3=record.get("velocity_diffusion_pc2_myr3"),
            density_gradient_scale_pc=record.get("density_gradient_scale_pc"),
            closure_status=record.get("closure_status", "uncalibrated"),
        )
