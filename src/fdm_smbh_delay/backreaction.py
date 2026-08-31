"""Paired live-versus-frozen backreaction decisions for kpc-to-coalescence.

The resolved simulation remains the authority for a live wake, scattering
response, or wave backreaction.  This module only decides whether a measured
paired comparison is sufficient to use a frozen/offline closure.  It never
creates a force or a delay and it never treats an unresolved rate as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from .delay_budget import DelaySegment


BACKREACTION_SCHEMA_VERSION = 1
_MODELS = {"cdm", "sidm", "fdm"}
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_TRACK_FIELDS = {
    "separation_pc",
    "orbital_power_pc2_myr3",
    "orbital_torque_msun_pc2_myr",
    "eccentricity",
}
_DECISION_INTERPRETATION = (
    "paired live/frozen force-treatment decision only; this is not "
    "a coalescence-time estimate"
)


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


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be exactly 64 hexadecimal characters")
    return value.lower()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be a JSON object")
    return record


@dataclass(frozen=True)
class BackreactionTrackPoint:
    """One orbit-averaged point from either a live or frozen track."""

    separation_pc: float
    orbital_power_pc2_myr3: float
    orbital_torque_msun_pc2_myr: float
    eccentricity: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.separation_pc,
                self.orbital_power_pc2_myr3,
                self.orbital_torque_msun_pc2_myr,
                self.eccentricity,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("backreaction track point must be finite")
        if self.separation_pc <= 0.0:
            raise ValueError("backreaction separation must be positive")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("backreaction eccentricity must lie in [0, 1)")

    def as_dict(self) -> dict[str, float]:
        return {
            "separation_pc": self.separation_pc,
            "orbital_power_pc2_myr3": self.orbital_power_pc2_myr3,
            "orbital_torque_msun_pc2_myr": self.orbital_torque_msun_pc2_myr,
            "eccentricity": self.eccentricity,
        }


def read_backreaction_track(path: str | Path) -> tuple[BackreactionTrackPoint, ...]:
    """Read one measured track after validating its exact schema."""

    source = Path(path).expanduser().resolve()
    record = _read_json(source, "backreaction track")
    if set(record) != {"schema_version", "status", "track"}:
        raise ValueError("backreaction track fields are invalid")
    if record["schema_version"] != 1 or record["status"] != "measured_track":
        raise ValueError("backreaction track is not a measured_track schema-v1 record")
    raw_points = record["track"]
    if not isinstance(raw_points, list):
        raise ValueError("backreaction track must contain a list")
    points: list[BackreactionTrackPoint] = []
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, Mapping) or set(raw) != _TRACK_FIELDS:
            raise ValueError(f"backreaction track point {index} fields are invalid")
        try:
            points.append(BackreactionTrackPoint(**raw))
        except (TypeError, ValueError) as error:
            raise ValueError(f"backreaction track point {index} is invalid: {error}") from error
    _validate_points(tuple(points), "backreaction track")
    return tuple(points)


@dataclass(frozen=True)
class BackreactionEvidence:
    """Provenance and quality metadata for one live/frozen pair."""

    model: str
    live_checkpoint_id: str
    frozen_checkpoint_id: str
    live_source_path: str
    live_source_sha256: str
    frozen_source_path: str
    frozen_source_sha256: str
    live_force_accounting: str
    frozen_force_accounting: str
    maximum_live_relative_energy_error: float
    maximum_frozen_relative_energy_error: float
    minimum_live_orbital_resolution_cells: float
    minimum_frozen_orbital_resolution_cells: float

    def __post_init__(self) -> None:
        if self.model not in _MODELS:
            raise ValueError("backreaction model must be cdm, sidm, or fdm")
        for name in (
            "live_checkpoint_id",
            "frozen_checkpoint_id",
            "live_source_path",
            "frozen_source_path",
            "live_force_accounting",
            "frozen_force_accounting",
        ):
            _nonempty(getattr(self, name), name)
        object.__setattr__(
            self,
            "live_source_sha256",
            _sha256(self.live_source_sha256, "live_source_sha256"),
        )
        object.__setattr__(
            self,
            "frozen_source_sha256",
            _sha256(self.frozen_source_sha256, "frozen_source_sha256"),
        )
        for name in (
            "maximum_live_relative_energy_error",
            "maximum_frozen_relative_energy_error",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "minimum_live_orbital_resolution_cells",
            "minimum_frozen_orbital_resolution_cells",
        ):
            value = _finite(getattr(self, name), name, positive=True)
            object.__setattr__(self, name, value)
        if self.model == "fdm" and self.live_force_accounting != "live_wave_only":
            raise ValueError(
                "FDM live track must use live_wave_only force accounting"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "live_checkpoint_id": self.live_checkpoint_id,
            "frozen_checkpoint_id": self.frozen_checkpoint_id,
            "live_source": {
                "path": self.live_source_path,
                "sha256": self.live_source_sha256,
            },
            "frozen_source": {
                "path": self.frozen_source_path,
                "sha256": self.frozen_source_sha256,
            },
            "live_force_accounting": self.live_force_accounting,
            "frozen_force_accounting": self.frozen_force_accounting,
            "maximum_live_relative_energy_error": self.maximum_live_relative_energy_error,
            "maximum_frozen_relative_energy_error": self.maximum_frozen_relative_energy_error,
            "minimum_live_orbital_resolution_cells": self.minimum_live_orbital_resolution_cells,
            "minimum_frozen_orbital_resolution_cells": self.minimum_frozen_orbital_resolution_cells,
        }


@dataclass(frozen=True)
class BackreactionGateConfig:
    """Acceptance controls for a paired live/frozen comparison."""

    minimum_overlap_factor: float = 2.0
    minimum_overlap_points: int = 3
    maximum_rate_fractional_difference: float = 0.20
    maximum_eccentricity_difference: float = 0.02
    maximum_relative_energy_error: float = 1.0e-3
    minimum_orbital_resolution_cells: float = 4.0
    maximum_log_separation_match: float = 0.15
    rate_floor_fraction: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in (
            "minimum_overlap_factor",
            "maximum_rate_fractional_difference",
            "maximum_eccentricity_difference",
            "maximum_relative_energy_error",
            "minimum_orbital_resolution_cells",
            "maximum_log_separation_match",
            "rate_floor_fraction",
        ):
            _finite(getattr(self, name), name, positive=True)
        if self.minimum_overlap_factor <= 1.0:
            raise ValueError("minimum_overlap_factor must exceed one")
        if self.minimum_overlap_points < 2:
            raise ValueError("minimum_overlap_points must be at least two")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "minimum_overlap_factor": self.minimum_overlap_factor,
            "minimum_overlap_points": self.minimum_overlap_points,
            "maximum_rate_fractional_difference": self.maximum_rate_fractional_difference,
            "maximum_eccentricity_difference": self.maximum_eccentricity_difference,
            "maximum_relative_energy_error": self.maximum_relative_energy_error,
            "minimum_orbital_resolution_cells": self.minimum_orbital_resolution_cells,
            "maximum_log_separation_match": self.maximum_log_separation_match,
            "rate_floor_fraction": self.rate_floor_fraction,
        }


def _gate_config_from_mapping(
    record: Any, *, enforce_project_floor: bool = True
) -> BackreactionGateConfig:
    if record is None:
        return BackreactionGateConfig()
    if not isinstance(record, Mapping):
        raise ValueError("backreaction gates must be an object")
    allowed = {
        "minimum_overlap_factor",
        "minimum_overlap_points",
        "maximum_rate_fractional_difference",
        "maximum_eccentricity_difference",
        "maximum_relative_energy_error",
        "minimum_orbital_resolution_cells",
        "maximum_log_separation_match",
        "rate_floor_fraction",
    }
    if set(record) != allowed:
        raise ValueError("backreaction gates fields are invalid")
    config = BackreactionGateConfig(**dict(record))
    if enforce_project_floor:
        baseline = BackreactionGateConfig()
        if (
            config.minimum_overlap_factor < baseline.minimum_overlap_factor
            or config.minimum_overlap_points < baseline.minimum_overlap_points
            or config.maximum_rate_fractional_difference > baseline.maximum_rate_fractional_difference
            or config.maximum_eccentricity_difference > baseline.maximum_eccentricity_difference
            or config.maximum_relative_energy_error > baseline.maximum_relative_energy_error
            or config.minimum_orbital_resolution_cells < baseline.minimum_orbital_resolution_cells
            or config.maximum_log_separation_match > baseline.maximum_log_separation_match
            or config.rate_floor_fraction < baseline.rate_floor_fraction
        ):
            raise ValueError(
                "backreaction gates may only be stricter than the project defaults"
            )
    return config


@dataclass(frozen=True)
class BackreactionManifest:
    """A fully re-readable live/frozen manifest and its source tracks."""

    path: Path
    sha256: str
    model: str
    live_points: tuple[BackreactionTrackPoint, ...]
    frozen_points: tuple[BackreactionTrackPoint, ...]
    evidence: BackreactionEvidence
    config: BackreactionGateConfig


def _manifest_side(
    record: Any,
    *,
    label: str,
    base: Path,
) -> tuple[dict[str, Any], tuple[BackreactionTrackPoint, ...]]:
    expected = {
        "checkpoint_id",
        "source",
        "force_accounting",
        "maximum_relative_energy_error",
        "minimum_orbital_resolution_cells",
    }
    if not isinstance(record, Mapping) or set(record) != expected:
        raise ValueError(f"{label} evidence fields are invalid")
    source = record.get("source")
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
        raise ValueError(f"{label} source fields are invalid")
    source_path = Path(_nonempty(source.get("path"), f"{label} source path")).expanduser()
    source_path = (source_path if source_path.is_absolute() else base / source_path).resolve()
    declared_sha = _sha256(source.get("sha256"), f"{label} source SHA-256")
    try:
        actual_sha = _file_sha256(source_path)
    except OSError as error:
        raise ValueError(f"cannot read {label} source: {error}") from error
    if actual_sha != declared_sha:
        raise ValueError(f"{label} source SHA-256 differs")
    parsed = dict(record)
    parsed["checkpoint_id"] = _nonempty(parsed["checkpoint_id"], f"{label} checkpoint_id")
    parsed["source_path"] = str(source_path)
    parsed["source_sha256"] = actual_sha
    parsed["force_accounting"] = _nonempty(
        parsed["force_accounting"], f"{label} force_accounting"
    )
    return parsed, read_backreaction_track(source_path)


def read_backreaction_manifest(path: str | Path) -> BackreactionManifest:
    """Read a manifest and re-hash both measured source track files."""

    source = Path(path).expanduser().resolve()
    record = _read_json(source, "backreaction manifest")
    expected = {"schema_version", "model", "live", "frozen"}
    optional = {"gates"}
    if set(record) - expected - optional or not expected.issubset(record):
        raise ValueError("backreaction manifest fields are invalid")
    if record["schema_version"] != 1:
        raise ValueError("unsupported backreaction manifest schema")
    model = _nonempty(record.get("model"), "backreaction model")
    live_record, live_points = _manifest_side(
        record.get("live"), label="live", base=source.parent
    )
    frozen_record, frozen_points = _manifest_side(
        record.get("frozen"), label="frozen", base=source.parent
    )
    evidence = BackreactionEvidence(
        model=model,
        live_checkpoint_id=live_record["checkpoint_id"],
        frozen_checkpoint_id=frozen_record["checkpoint_id"],
        live_source_path=live_record["source_path"],
        live_source_sha256=live_record["source_sha256"],
        frozen_source_path=frozen_record["source_path"],
        frozen_source_sha256=frozen_record["source_sha256"],
        live_force_accounting=live_record["force_accounting"],
        frozen_force_accounting=frozen_record["force_accounting"],
        maximum_live_relative_energy_error=live_record["maximum_relative_energy_error"],
        maximum_frozen_relative_energy_error=frozen_record["maximum_relative_energy_error"],
        minimum_live_orbital_resolution_cells=live_record["minimum_orbital_resolution_cells"],
        minimum_frozen_orbital_resolution_cells=frozen_record["minimum_orbital_resolution_cells"],
    )
    return BackreactionManifest(
        path=source,
        sha256=_file_sha256(source),
        model=model,
        live_points=live_points,
        frozen_points=frozen_points,
        evidence=evidence,
        config=_gate_config_from_mapping(record.get("gates")),
    )


@dataclass(frozen=True)
class BackreactionDecision:
    """Non-throwing recommendation for the force-treatment boundary."""

    status: str
    model: str
    overlap_low_pc: float | None
    overlap_high_pc: float | None
    matched_points: int
    maximum_power_fractional_difference: float | None
    maximum_torque_fractional_difference: float | None
    maximum_eccentricity_difference: float | None
    reasons: tuple[str, ...]
    evidence: BackreactionEvidence
    config: BackreactionGateConfig

    @property
    def offline_acceptable(self) -> bool:
        return self.status == "offline_acceptable"

    @property
    def runtime_required(self) -> bool:
        return self.status == "runtime_required"

    @property
    def censored(self) -> bool:
        return self.status == "censored"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BACKREACTION_SCHEMA_VERSION,
            "status": self.status,
            "interpretation": _DECISION_INTERPRETATION,
            "model": self.model,
            "overlap": {
                "low_pc": self.overlap_low_pc,
                "high_pc": self.overlap_high_pc,
                "matched_points": self.matched_points,
            },
            "maximum_power_fractional_difference": self.maximum_power_fractional_difference,
            "maximum_torque_fractional_difference": self.maximum_torque_fractional_difference,
            "maximum_eccentricity_difference": self.maximum_eccentricity_difference,
            "reasons": list(self.reasons),
            "evidence": self.evidence.as_dict(),
            "gates": self.config.as_dict(),
        }


def materialize_backreaction_delay_segment(
    decision: BackreactionDecision,
    *,
    name: str,
    offline_delay_myr: float | None,
    source_case_id: str | None = None,
) -> DelaySegment:
    """Turn a backreaction decision into one censor-preserving delay segment.

    A candidate delay is materialized only for ``offline_acceptable``.  For a
    resolved but materially different pair (``runtime_required``), or for
    inadequate evidence (``censored``), the candidate is discarded and the
    returned segment has no delay.  The local import keeps this module
    independent from the generic delay composer at import time.
    """

    name = _nonempty(name, "delay segment name")
    if source_case_id is not None:
        source_case_id = _nonempty(source_case_id, "source_case_id")
    decision_digest = hashlib.sha256(
        json.dumps(
            decision.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    from .delay_budget import DelaySegment

    if decision.offline_acceptable:
        if offline_delay_myr is None:
            raise ValueError(
                "offline_acceptable backreaction requires an offline delay"
            )
        delay = _finite(offline_delay_myr, "offline_delay_myr")
        if delay < 0.0:
            raise ValueError("offline_delay_myr must be non-negative")
        return DelaySegment(
            name,
            "complete",
            delay,
            reason=(
                "paired live/frozen backreaction accepted an offline closure "
                "over the measured support"
            ),
            source_case_id=source_case_id,
            source_sha256=decision_digest,
        )

    reason = (
        f"backreaction status={decision.status}; offline delay is not usable"
    )
    if decision.reasons:
        reason += ": " + "; ".join(decision.reasons)
    return DelaySegment(
        name,
        "censored",
        None,
        reason=reason,
        source_case_id=source_case_id,
        source_sha256=decision_digest,
    )


def _validate_points(
    points: tuple[BackreactionTrackPoint, ...], name: str
) -> None:
    if len(points) < 2:
        raise ValueError(f"{name} must contain at least two points")
    separations = np.asarray([point.separation_pc for point in points], dtype=float)
    if np.any(np.diff(separations) <= 0.0):
        raise ValueError(f"{name} separations must be strictly increasing")


def _fractional_difference(left: float, right: float, floor: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), floor)


def _match_points(
    live: tuple[BackreactionTrackPoint, ...],
    frozen: tuple[BackreactionTrackPoint, ...],
    *,
    overlap_low: float,
    overlap_high: float,
    maximum_log_separation_match: float,
) -> list[tuple[BackreactionTrackPoint, BackreactionTrackPoint]]:
    live_candidates = [
        point for point in live if overlap_low <= point.separation_pc <= overlap_high
    ]
    frozen_candidates = [
        point
        for point in frozen
        if overlap_low <= point.separation_pc <= overlap_high
    ]
    pairs: list[tuple[BackreactionTrackPoint, BackreactionTrackPoint]] = []
    used_frozen: set[int] = set()
    for live_point in live_candidates:
        choices = [
            (abs(math.log(point.separation_pc / live_point.separation_pc)), index)
            for index, point in enumerate(frozen_candidates)
            if index not in used_frozen
        ]
        if not choices:
            continue
        distance, index = min(choices)
        if distance <= maximum_log_separation_match:
            pairs.append((live_point, frozen_candidates[index]))
            used_frozen.add(index)
    return pairs


def _decision(
    *,
    status: str,
    model: str,
    evidence: BackreactionEvidence,
    overlap_low: float | None,
    overlap_high: float | None,
    matched_points: int,
    maximum_power: float | None,
    maximum_torque: float | None,
    maximum_eccentricity: float | None,
    reasons: list[str],
    config: BackreactionGateConfig,
) -> BackreactionDecision:
    return BackreactionDecision(
        status=status,
        model=model,
        overlap_low_pc=overlap_low,
        overlap_high_pc=overlap_high,
        matched_points=matched_points,
        maximum_power_fractional_difference=maximum_power,
        maximum_torque_fractional_difference=maximum_torque,
        maximum_eccentricity_difference=maximum_eccentricity,
        reasons=tuple(dict.fromkeys(reasons)),
        evidence=evidence,
        config=config,
    )


def assess_live_frozen_backreaction(
    *,
    model: str,
    live_points: tuple[BackreactionTrackPoint, ...],
    frozen_points: tuple[BackreactionTrackPoint, ...],
    evidence: BackreactionEvidence,
    config: BackreactionGateConfig = BackreactionGateConfig(),
) -> BackreactionDecision:
    """Classify whether an offline closure is safe for a measured pair.

    ``offline_acceptable`` means that the paired tracks agree within the
    configured gates and a frozen/offline treatment may be used for that
    tested support.  ``runtime_required`` means the tracks are resolved but
    live backreaction changes the result enough that the runtime treatment (or
    a model-specific live residual) must be retained.  ``censored`` means the
    evidence is not adequate to decide; no force or delay is inferred.
    """

    if model not in _MODELS:
        raise ValueError("model must be cdm, sidm, or fdm")
    if evidence.model != model:
        raise ValueError("backreaction evidence model differs from requested model")
    live = tuple(live_points)
    frozen = tuple(frozen_points)
    _validate_points(live, "live_points")
    _validate_points(frozen, "frozen_points")

    reasons: list[str] = []
    if evidence.live_checkpoint_id != evidence.frozen_checkpoint_id:
        reasons.append("live and frozen tracks do not share one checkpoint")
    if evidence.live_source_sha256 == evidence.frozen_source_sha256:
        reasons.append("live and frozen source artifacts have the same SHA-256")
    if model == "fdm" and evidence.live_force_accounting != "live_wave_only":
        reasons.append("FDM live track is not resolved-wave-only")
    if (
        evidence.maximum_live_relative_energy_error
        > config.maximum_relative_energy_error
    ):
        reasons.append("live conservation error exceeds the backreaction gate")
    if (
        evidence.maximum_frozen_relative_energy_error
        > config.maximum_relative_energy_error
    ):
        reasons.append("frozen conservation error exceeds the backreaction gate")
    if (
        evidence.minimum_live_orbital_resolution_cells
        < config.minimum_orbital_resolution_cells
    ):
        reasons.append("live orbital resolution is below the backreaction gate")
    if (
        evidence.minimum_frozen_orbital_resolution_cells
        < config.minimum_orbital_resolution_cells
    ):
        reasons.append("frozen orbital resolution is below the backreaction gate")

    overlap_low = max(live[0].separation_pc, frozen[0].separation_pc)
    overlap_high = min(live[-1].separation_pc, frozen[-1].separation_pc)
    if overlap_high <= overlap_low:
        reasons.append("live and frozen supports do not overlap")
        return _decision(
            status="censored",
            model=model,
            evidence=evidence,
            overlap_low=None,
            overlap_high=None,
            matched_points=0,
            maximum_power=None,
            maximum_torque=None,
            maximum_eccentricity=None,
            reasons=reasons,
            config=config,
        )
    if overlap_high / overlap_low < config.minimum_overlap_factor:
        reasons.append("live/frozen overlap has insufficient separation width")

    pairs = _match_points(
        live,
        frozen,
        overlap_low=overlap_low,
        overlap_high=overlap_high,
        maximum_log_separation_match=config.maximum_log_separation_match,
    )
    if len(pairs) < config.minimum_overlap_points:
        reasons.append("live/frozen overlap has too few matched rate points")

    all_power = [
        abs(point.orbital_power_pc2_myr3)
        for point in live + frozen
    ]
    all_torque = [
        abs(point.orbital_torque_msun_pc2_myr)
        for point in live + frozen
    ]
    power_floor = (
        max(max(all_power, default=0.0), np.finfo(float).tiny)
        * config.rate_floor_fraction
    )
    torque_floor = (
        max(max(all_torque, default=0.0), np.finfo(float).tiny)
        * config.rate_floor_fraction
    )
    power_differences: list[float] = []
    torque_differences: list[float] = []
    eccentricity_differences: list[float] = []
    unresolved = False
    for live_point, frozen_point in pairs:
        live_power = live_point.orbital_power_pc2_myr3
        frozen_power = frozen_point.orbital_power_pc2_myr3
        live_torque = live_point.orbital_torque_msun_pc2_myr
        frozen_torque = frozen_point.orbital_torque_msun_pc2_myr
        if abs(live_power) <= power_floor or abs(frozen_power) <= power_floor:
            unresolved = True
        if abs(live_torque) <= torque_floor or abs(frozen_torque) <= torque_floor:
            unresolved = True
        if np.sign(live_power) != np.sign(frozen_power):
            reasons.append("live/frozen orbital power changes sign")
        if np.sign(live_torque) != np.sign(frozen_torque):
            reasons.append("live/frozen orbital torque changes sign")
        power_differences.append(
            _fractional_difference(live_power, frozen_power, power_floor)
        )
        torque_differences.append(
            _fractional_difference(live_torque, frozen_torque, torque_floor)
        )
        eccentricity_differences.append(
            abs(live_point.eccentricity - frozen_point.eccentricity)
        )

    if unresolved:
        reasons.append("one or more paired power/torque rates are unresolved")
    maximum_power = max(power_differences, default=None)
    maximum_torque = max(torque_differences, default=None)
    maximum_eccentricity = max(eccentricity_differences, default=None)
    if (
        maximum_power is not None
        and maximum_power > config.maximum_rate_fractional_difference
    ):
        reasons.append("live/frozen orbital power differs beyond the gate")
    if (
        maximum_torque is not None
        and maximum_torque > config.maximum_rate_fractional_difference
    ):
        reasons.append("live/frozen orbital torque differs beyond the gate")
    if (
        maximum_eccentricity is not None
        and maximum_eccentricity > config.maximum_eccentricity_difference
    ):
        reasons.append("live/frozen eccentricity differs beyond the gate")

    censor_markers = (
        "checkpoint",
        "do not overlap",
        "insufficient separation width",
        "too few matched",
        "same SHA-256",
        "not resolved-wave-only",
        "conservation error",
        "resolution is below",
        "unresolved",
    )
    has_censor_reason = any(
        any(marker in reason for marker in censor_markers) for reason in reasons
    )
    if has_censor_reason:
        status = "censored"
    elif reasons:
        status = "runtime_required"
    else:
        status = "offline_acceptable"
        reasons.append(
            "paired live/frozen rates agree within the offline backreaction gate"
        )
    return _decision(
        status=status,
        model=model,
        evidence=evidence,
        overlap_low=overlap_low,
        overlap_high=overlap_high,
        matched_points=len(pairs),
        maximum_power=maximum_power,
        maximum_torque=maximum_torque,
        maximum_eccentricity=maximum_eccentricity,
        reasons=reasons,
        config=config,
    )


def read_verified_backreaction_decision(
    path: str | Path,
) -> BackreactionDecision:
    """Rebuild a saved CLI decision from its manifest and current track bytes."""

    source = Path(path).expanduser().resolve()
    record = _read_json(source, "backreaction decision")
    expected_fields = {
        "schema_version",
        "status",
        "interpretation",
        "model",
        "overlap",
        "maximum_power_fractional_difference",
        "maximum_torque_fractional_difference",
        "maximum_eccentricity_difference",
        "reasons",
        "evidence",
        "gates",
        "input_manifest",
    }
    if set(record) != expected_fields:
        raise ValueError("backreaction decision fields are invalid")
    if record.get("schema_version") != BACKREACTION_SCHEMA_VERSION:
        raise ValueError("unsupported backreaction decision schema")
    if record.get("interpretation") != _DECISION_INTERPRETATION:
        raise ValueError("backreaction decision interpretation is invalid")
    manifest_record = record.get("input_manifest")
    if not isinstance(manifest_record, Mapping) or set(manifest_record) != {
        "path",
        "sha256",
    }:
        raise ValueError("backreaction input manifest fields are invalid")
    manifest_path = Path(
        _nonempty(manifest_record.get("path"), "backreaction input manifest path")
    ).expanduser()
    manifest_path = (
        manifest_path if manifest_path.is_absolute() else source.parent / manifest_path
    ).resolve()
    manifest_sha = _sha256(
        manifest_record.get("sha256"), "backreaction input manifest SHA-256"
    )
    try:
        actual_manifest_sha = _file_sha256(manifest_path)
    except OSError as error:
        raise ValueError(f"cannot read backreaction input manifest: {error}") from error
    if actual_manifest_sha != manifest_sha:
        raise ValueError("backreaction input manifest SHA-256 differs")
    manifest = read_backreaction_manifest(manifest_path)
    if manifest.sha256 != manifest_sha or manifest.path != manifest_path:
        raise ValueError("backreaction input manifest identity differs")
    decision = assess_live_frozen_backreaction(
        model=manifest.model,
        live_points=manifest.live_points,
        frozen_points=manifest.frozen_points,
        evidence=manifest.evidence,
        config=manifest.config,
    )
    expected_record = decision.as_dict()
    expected_record["input_manifest"] = {
        "path": str(manifest.path),
        "sha256": manifest.sha256,
    }
    if record != expected_record:
        raise ValueError("saved backreaction decision differs from current evidence")
    return decision
