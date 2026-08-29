"""Censor-first calibrated drift/diffusion response for an outer FDM halo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class FDMOuterResponseTable:
    """Measured outer-halo response; this class does not define a drag law."""

    radii_pc: np.ndarray
    drift_acceleration_pc_myr2: np.ndarray
    diffusion_tensor_pc2_myr3: np.ndarray
    response_status: str = "uncalibrated"

    def __post_init__(self) -> None:
        radii = np.asarray(self.radii_pc, dtype=float)
        drift = np.asarray(self.drift_acceleration_pc_myr2, dtype=float)
        diffusion = np.asarray(self.diffusion_tensor_pc2_myr3, dtype=float)
        if radii.ndim != 1 or radii.size < 2 or np.any(~np.isfinite(radii)):
            raise ValueError("response radii must be a finite one-dimensional array")
        if np.any(radii <= 0.0) or np.any(np.diff(radii) <= 0.0):
            raise ValueError("response radii must be strictly increasing and positive")
        if drift.shape != (radii.size, 3) or np.any(~np.isfinite(drift)):
            raise ValueError("drift response must have shape (N, 3) and be finite")
        if diffusion.shape != (radii.size, 3, 3) or np.any(~np.isfinite(diffusion)):
            raise ValueError("diffusion response must have shape (N, 3, 3) and be finite")
        if np.max(np.abs(diffusion - np.swapaxes(diffusion, 1, 2))) > 1.0e-10:
            raise ValueError("diffusion tensors must be symmetric")
        eigenvalues = np.linalg.eigvalsh(diffusion)
        if np.any(eigenvalues < -1.0e-12):
            raise ValueError("diffusion tensors must be positive semidefinite")
        if self.response_status not in {"calibrated", "uncalibrated", "censored"}:
            raise ValueError("unsupported outer response status")
        object.__setattr__(self, "radii_pc", radii)
        object.__setattr__(self, "drift_acceleration_pc_myr2", drift)
        object.__setattr__(self, "diffusion_tensor_pc2_myr3", diffusion)
        object.__setattr__(
            self,
            "_drift_interpolators",
            tuple(
                PchipInterpolator(radii, drift[:, axis], extrapolate=False)
                for axis in range(3)
            ),
        )
        object.__setattr__(
            self,
            "_diffusion_interpolators",
            tuple(
                tuple(
                    PchipInterpolator(radii, diffusion[:, row, col], extrapolate=False)
                    for col in range(3)
                )
                for row in range(3)
            ),
        )

    def evaluate(self, radius_pc: float) -> dict[str, Any]:
        """Evaluate a calibrated response, rejecting unsupported radii/status."""

        if self.response_status != "calibrated":
            raise ValueError(
                "outer FDM response is not calibrated; physical delay is censored"
            )
        radius = float(radius_pc)
        if (
            not np.isfinite(radius)
            or radius < self.radii_pc[0]
            or radius > self.radii_pc[-1]
        ):
            raise ValueError("outer FDM response requested outside tabulated support")
        drift = np.array(
            [float(interpolator(radius)) for interpolator in self._drift_interpolators],
            dtype=float,
        )
        diffusion = np.array(
            [
                [float(self._diffusion_interpolators[row][col](radius)) for col in range(3)]
                for row in range(3)
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(drift)) or np.any(~np.isfinite(diffusion)):
            raise ValueError("outer FDM response interpolation produced non-finite values")
        return {
            "drift_acceleration_pc_myr2": drift,
            "diffusion_tensor_pc2_myr3": diffusion,
        }

    def decision(self, radius_pc: float) -> dict[str, Any]:
        """Return a censor-preserving decision without inventing a response."""

        if self.response_status != "calibrated":
            return {
                "status": "censored",
                "reason": f"outer FDM response status is {self.response_status}",
            }
        try:
            values = self.evaluate(radius_pc)
        except ValueError as error:
            return {"status": "censored", "reason": str(error)}
        return {"status": "available", **values}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "radii_pc": self.radii_pc.tolist(),
            "drift_acceleration_pc_myr2": self.drift_acceleration_pc_myr2.tolist(),
            "diffusion_tensor_pc2_myr3": self.diffusion_tensor_pc2_myr3.tolist(),
            "response_status": self.response_status,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "FDMOuterResponseTable":
        if not isinstance(record, Mapping) or record.get("schema_version") != 1:
            raise ValueError("unsupported outer FDM response schema")
        return cls(
            radii_pc=record.get("radii_pc"),
            drift_acceleration_pc_myr2=record.get("drift_acceleration_pc_myr2"),
            diffusion_tensor_pc2_myr3=record.get("diffusion_tensor_pc2_myr3"),
            response_status=record.get("response_status", "uncalibrated"),
        )
