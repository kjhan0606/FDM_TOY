"""Assess the initial relaxation/conservation window of a dual-soliton seed.

The evidence table is deliberately small and is expected to be produced from
time-resolved FDM outputs during bounded manual postprocessing.  It does not
read full wave fields, run an FFT, or infer an unrecorded conservation law.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DUAL_SOLITON_RELAXATION_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _positive_series(
    value: Any, name: str, samples: int | None = None, *, allow_zero: bool = False
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    invalid = array < 0.0 if allow_zero else array <= 0.0
    qualifier = "non-negative" if allow_zero else "positive"
    if array.ndim != 1 or array.size < 3 or np.any(~np.isfinite(array)) or np.any(invalid):
        raise ValueError(f"{name} must contain at least three finite {qualifier} values")
    if samples is not None and array.shape != (samples,):
        raise ValueError(f"{name} must have {samples} samples")
    return array


def _nonnegative_series(value: Any, name: str, samples: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (samples,) or np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must have {samples} finite non-negative values")
    return array


def _periodic_separation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = first - second
    delta -= np.rint(delta)
    return np.linalg.norm(delta, axis=1)


@dataclass(frozen=True)
class RelaxationConservationThresholds:
    maximum_relative_component_mass_drift: float = 0.02
    maximum_relative_component_radius_drift: float = 0.02
    maximum_relative_wave_mass_error: float = 1.0e-3
    maximum_relative_hamiltonian_error: float = 1.0e-3
    maximum_relative_angular_momentum_error: float = 1.0e-3
    minimum_core_separation_to_radius_ratio: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_relative_component_mass_drift",
            "maximum_relative_component_radius_drift",
            "maximum_relative_wave_mass_error",
            "maximum_relative_hamiltonian_error",
            "maximum_relative_angular_momentum_error",
            "minimum_core_separation_to_radius_ratio",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))

    def as_dict(self) -> dict[str, float]:
        return {
            "maximum_relative_component_mass_drift": self.maximum_relative_component_mass_drift,
            "maximum_relative_component_radius_drift": self.maximum_relative_component_radius_drift,
            "maximum_relative_wave_mass_error": self.maximum_relative_wave_mass_error,
            "maximum_relative_hamiltonian_error": self.maximum_relative_hamiltonian_error,
            "maximum_relative_angular_momentum_error": self.maximum_relative_angular_momentum_error,
            "minimum_core_separation_to_radius_ratio": self.minimum_core_separation_to_radius_ratio,
        }


@dataclass(frozen=True)
class DualSolitonRelaxationEvidence:
    """A postprocessed initial window, tied to a verified runtime seed record."""

    seed_case_id: str
    runtime_identity_path: Path
    relaxation_window_start_code: float
    sample_times_code: np.ndarray
    component_core_mass_code: np.ndarray
    component_core_radius_box: np.ndarray
    component_centres_box: np.ndarray
    relative_wave_mass_error: np.ndarray
    relative_hamiltonian_error: np.ndarray
    relative_angular_momentum_error: np.ndarray
    thresholds: RelaxationConservationThresholds

    def __post_init__(self) -> None:
        if not isinstance(self.seed_case_id, str) or not self.seed_case_id.strip():
            raise ValueError("seed_case_id is required")
        object.__setattr__(
            self,
            "runtime_identity_path",
            Path(self.runtime_identity_path).expanduser().resolve(),
        )
        times = _positive_series(
            self.sample_times_code, "sample_times_code", allow_zero=True
        )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("sample_times_code must be strictly increasing")
        start = _finite(self.relaxation_window_start_code, "relaxation_window_start_code")
        if start < times[0] or start >= times[-2]:
            raise ValueError("relaxation window must contain at least three samples")
        mask = times >= start
        if int(np.count_nonzero(mask)) < 3:
            raise ValueError("relaxation window must contain at least three samples")
        object.__setattr__(self, "sample_times_code", times)
        object.__setattr__(self, "relaxation_window_start_code", start)
        samples = int(times.size)
        for name in ("component_core_mass_code", "component_core_radius_box"):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (samples, 2) or np.any(~np.isfinite(array)) or np.any(array <= 0.0):
                raise ValueError(f"{name} must have shape ({samples}, 2) and positive values")
            object.__setattr__(self, name, array)
        centres = np.asarray(self.component_centres_box, dtype=float)
        if centres.shape != (samples, 2, 3) or np.any(~np.isfinite(centres)):
            raise ValueError(f"component_centres_box must have shape ({samples}, 2, 3)")
        if np.any(centres < 0.0) or np.any(centres >= 1.0):
            raise ValueError("component_centres_box must lie in [0, 1)")
        object.__setattr__(self, "component_centres_box", centres)
        for name in (
            "relative_wave_mass_error",
            "relative_hamiltonian_error",
            "relative_angular_momentum_error",
        ):
            object.__setattr__(
                self, name, _nonnegative_series(getattr(self, name), name, samples)
            )
        if not isinstance(self.thresholds, RelaxationConservationThresholds):
            raise ValueError("thresholds must be a relaxation/conservation threshold record")

    @property
    def relaxation_mask(self) -> np.ndarray:
        return self.sample_times_code >= self.relaxation_window_start_code

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DUAL_SOLITON_RELAXATION_SCHEMA_VERSION,
            "seed_case_id": self.seed_case_id,
            "runtime_identity_path": str(self.runtime_identity_path),
            "relaxation_window_start_code": self.relaxation_window_start_code,
            "sample_times_code": self.sample_times_code.tolist(),
            "component_core_mass_code": self.component_core_mass_code.tolist(),
            "component_core_radius_box": self.component_core_radius_box.tolist(),
            "component_centres_box": self.component_centres_box.tolist(),
            "relative_wave_mass_error": self.relative_wave_mass_error.tolist(),
            "relative_hamiltonian_error": self.relative_hamiltonian_error.tolist(),
            "relative_angular_momentum_error": self.relative_angular_momentum_error.tolist(),
            "thresholds": self.thresholds.as_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "DualSolitonRelaxationEvidence":
        if not isinstance(record, Mapping) or record.get("schema_version") != DUAL_SOLITON_RELAXATION_SCHEMA_VERSION:
            raise ValueError("unsupported dual-soliton relaxation evidence schema")
        expected = set(cls.__dataclass_fields__) | {"schema_version"}
        unknown = set(record) - expected
        if unknown:
            raise ValueError(f"dual-soliton relaxation evidence has unknown fields: {sorted(unknown)}")
        thresholds = record.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("dual-soliton relaxation evidence lacks thresholds")
        try:
            return cls(
                seed_case_id=record.get("seed_case_id"),
                runtime_identity_path=record.get("runtime_identity_path"),
                relaxation_window_start_code=record.get("relaxation_window_start_code"),
                sample_times_code=record.get("sample_times_code"),
                component_core_mass_code=record.get("component_core_mass_code"),
                component_core_radius_box=record.get("component_core_radius_box"),
                component_centres_box=record.get("component_centres_box"),
                relative_wave_mass_error=record.get("relative_wave_mass_error"),
                relative_hamiltonian_error=record.get("relative_hamiltonian_error"),
                relative_angular_momentum_error=record.get("relative_angular_momentum_error"),
                thresholds=RelaxationConservationThresholds(**thresholds),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"dual-soliton relaxation evidence is invalid: {error}") from error


@dataclass(frozen=True)
class DualSolitonRelaxationAssessment:
    source_path: Path
    source_sha256: str
    runtime_identity_path: Path
    runtime_identity_sha256: str
    seed_case_id: str
    status: str
    metrics: dict[str, float]
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "relaxation_conservation_passed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "initial-state relaxation/conservation only; this is neither an "
                "outer-merger calibration nor a physical coalescence delay"
            ),
            "source": {"path": str(self.source_path), "sha256": self.source_sha256},
            "runtime_identity": {
                "path": str(self.runtime_identity_path),
                "sha256": self.runtime_identity_sha256,
            },
            "seed_case_id": self.seed_case_id,
            "metrics": self.metrics,
            "reasons": list(self.reasons),
        }


def _verified_runtime_identity(path: Path, seed_case_id: str) -> None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read runtime identity record: {error}") from error
    if not isinstance(record, Mapping):
        raise ValueError("runtime identity record must be an object")
    if record.get("status") != "runtime_seed_identity_verified":
        raise ValueError("relaxation evidence requires a verified runtime seed identity")
    if record.get("seed_case_id") != seed_case_id:
        raise ValueError("runtime identity seed_case_id does not match relaxation evidence")


def assess_dual_soliton_relaxation(
    evidence_path: str | Path,
) -> DualSolitonRelaxationAssessment:
    """Gate an explicitly selected initial window without reading live outputs."""

    source = Path(evidence_path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dual-soliton relaxation evidence: {error}") from error
    evidence = DualSolitonRelaxationEvidence.from_dict(record)
    _verified_runtime_identity(evidence.runtime_identity_path, evidence.seed_case_id)
    window = evidence.relaxation_mask
    masses = evidence.component_core_mass_code[window]
    radii = evidence.component_core_radius_box[window]
    mass_drift = np.max(np.abs(masses / masses[0] - 1.0))
    radius_drift = np.max(np.abs(radii / radii[0] - 1.0))
    separations = _periodic_separation(
        evidence.component_centres_box[window, 0],
        evidence.component_centres_box[window, 1],
    )
    core_scale = np.maximum(radii[:, 0], radii[:, 1])
    minimum_separation_ratio = float(np.min(separations / core_scale))
    metrics = {
        "maximum_relative_component_mass_drift": float(mass_drift),
        "maximum_relative_component_radius_drift": float(radius_drift),
        "maximum_relative_wave_mass_error": float(np.max(evidence.relative_wave_mass_error[window])),
        "maximum_relative_hamiltonian_error": float(np.max(evidence.relative_hamiltonian_error[window])),
        "maximum_relative_angular_momentum_error": float(np.max(evidence.relative_angular_momentum_error[window])),
        "minimum_core_separation_to_radius_ratio": minimum_separation_ratio,
    }
    thresholds = evidence.thresholds
    reasons: list[str] = []
    for name in (
        "maximum_relative_component_mass_drift",
        "maximum_relative_component_radius_drift",
        "maximum_relative_wave_mass_error",
        "maximum_relative_hamiltonian_error",
        "maximum_relative_angular_momentum_error",
    ):
        if metrics[name] > getattr(thresholds, name):
            reasons.append(f"{name} exceeds the declared relaxation threshold")
    if minimum_separation_ratio < thresholds.minimum_core_separation_to_radius_ratio:
        reasons.append("two core centres enter the declared relaxation separation guard")
    return DualSolitonRelaxationAssessment(
        source_path=source,
        source_sha256=_sha256(source),
        runtime_identity_path=evidence.runtime_identity_path,
        runtime_identity_sha256=_sha256(evidence.runtime_identity_path),
        seed_case_id=evidence.seed_case_id,
        status=(
            "relaxation_conservation_passed"
            if not reasons
            else "relaxation_conservation_not_passed"
        ),
        metrics=metrics,
        reasons=tuple(reasons),
    )
