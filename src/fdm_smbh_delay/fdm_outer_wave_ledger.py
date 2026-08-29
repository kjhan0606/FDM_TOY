"""Compact provenance for time-resolved FDM outer-zoom wave observables."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np


_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_FORCE_ACCOUNTING = {"live_wave_only", "resolved_wake_plus_measured_residual"}


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _path(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _vector_array(value: Any, name: str, samples: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (samples, 3) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must have shape ({samples}, 3) and finite values")
    return array


def _scalar_array(value: Any, name: str, samples: int, *, positive: bool) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (samples,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must have {samples} finite values")
    invalid = array <= 0.0 if positive else array < 0.0
    if np.any(invalid):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return array


def _complex_array(value: Any, name: str, samples: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (samples, 2) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must have shape ({samples}, 2) and finite values")
    return array


@dataclass(frozen=True)
class FDMOuterWaveLedger:
    """Small index to the resolved wave evidence needed by an outer handoff.

    It preserves hashes rather than duplicating 3-D wave snapshots.  The
    complex l=1 and l=2 mode pairs, current, granule statistic, and protected
    wake clearances prevent a static soliton profile from being registered as
    a complete outer-halo description.
    """

    source_path: str
    source_sha256: str
    force_ledger_sha256: str
    field_snapshot_index_sha256: str
    profile_snapshot_index_sha256: str
    sample_times_myr: np.ndarray
    core_centres_pc: np.ndarray
    fdm_current_msun_pc2_myr: np.ndarray
    coherence_time_myr: np.ndarray
    de_broglie_wavelength_pc: np.ndarray
    granule_power: np.ndarray
    dipole_l1_real_imag: np.ndarray
    quadrupole_l2_real_imag: np.ndarray
    wake_extent_pc: np.ndarray
    hjm_wave_seam_clearance_pc: np.ndarray
    boundary_clearance_pc: np.ndarray
    maximum_relative_hamiltonian_error: float
    maximum_relative_mass_error: float
    maximum_relative_angular_momentum_error: float
    force_accounting: str = "live_wave_only"

    def __post_init__(self) -> None:
        for name in (
            "source_sha256",
            "force_ledger_sha256",
            "field_snapshot_index_sha256",
            "profile_snapshot_index_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "source_path", _path(self.source_path, "source_path"))
        if self.force_accounting not in _FORCE_ACCOUNTING:
            raise ValueError("force_accounting must preserve a resolved-wave force")
        times = np.asarray(self.sample_times_myr, dtype=float)
        if times.ndim != 1 or times.size < 2 or np.any(~np.isfinite(times)):
            raise ValueError("sample_times_myr must contain at least two finite samples")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("sample_times_myr must be strictly increasing")
        object.__setattr__(self, "sample_times_myr", times)
        samples = int(times.size)
        for name in ("core_centres_pc", "fdm_current_msun_pc2_myr"):
            object.__setattr__(
                self, name, _vector_array(getattr(self, name), name, samples)
            )
        for name in (
            "coherence_time_myr",
            "de_broglie_wavelength_pc",
            "wake_extent_pc",
            "hjm_wave_seam_clearance_pc",
            "boundary_clearance_pc",
        ):
            object.__setattr__(
                self,
                name,
                _scalar_array(getattr(self, name), name, samples, positive=True),
            )
        object.__setattr__(
            self,
            "granule_power",
            _scalar_array(self.granule_power, "granule_power", samples, positive=False),
        )
        for name in ("dipole_l1_real_imag", "quadrupole_l2_real_imag"):
            object.__setattr__(
                self, name, _complex_array(getattr(self, name), name, samples)
            )
        protected_extent = np.maximum(self.wake_extent_pc, self.de_broglie_wavelength_pc)
        if np.any(self.hjm_wave_seam_clearance_pc <= protected_extent):
            raise ValueError("HJM/wave seam lies inside the protected wave extent")
        if np.any(self.boundary_clearance_pc <= self.wake_extent_pc):
            raise ValueError("wave boundary lies inside the resolved wake extent")
        for name in (
            "maximum_relative_hamiltonian_error",
            "maximum_relative_mass_error",
            "maximum_relative_angular_momentum_error",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite and non-negative")
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be finite and non-negative") from error
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "force_ledger_sha256": self.force_ledger_sha256,
            "field_snapshot_index_sha256": self.field_snapshot_index_sha256,
            "profile_snapshot_index_sha256": self.profile_snapshot_index_sha256,
            "sample_times_myr": self.sample_times_myr.tolist(),
            "core_centres_pc": self.core_centres_pc.tolist(),
            "fdm_current_msun_pc2_myr": self.fdm_current_msun_pc2_myr.tolist(),
            "coherence_time_myr": self.coherence_time_myr.tolist(),
            "de_broglie_wavelength_pc": self.de_broglie_wavelength_pc.tolist(),
            "granule_power": self.granule_power.tolist(),
            "dipole_l1_real_imag": self.dipole_l1_real_imag.tolist(),
            "quadrupole_l2_real_imag": self.quadrupole_l2_real_imag.tolist(),
            "wake_extent_pc": self.wake_extent_pc.tolist(),
            "hjm_wave_seam_clearance_pc": self.hjm_wave_seam_clearance_pc.tolist(),
            "boundary_clearance_pc": self.boundary_clearance_pc.tolist(),
            "maximum_relative_hamiltonian_error": self.maximum_relative_hamiltonian_error,
            "maximum_relative_mass_error": self.maximum_relative_mass_error,
            "maximum_relative_angular_momentum_error": (
                self.maximum_relative_angular_momentum_error
            ),
            "force_accounting": self.force_accounting,
        }

    @classmethod
    def from_dict(cls, record: Any) -> "FDMOuterWaveLedger":
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ValueError("unsupported FDM outer-wave ledger schema")
        payload = dict(record)
        payload.pop("schema_version")
        return cls(**payload)
