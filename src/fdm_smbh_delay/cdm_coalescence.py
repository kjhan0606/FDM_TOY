"""Provenance-gated CDM capture-to-coalescence delay composition.

This module intentionally does not consume HR5 disappearance histories or a
population-level fixed delay.  A completed CDM environmental interval must be
an explicitly measured, same-model rate integration whose independent phase
ensemble was accepted first.  Until those records exist, the result remains
missing or censored rather than becoming a physical coalescence prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .delay_budget import DelaySegment, TrueMergeEstimate, compose_delay_segments
from .gw import peters_time_myr


CDM_DELAY_STAGE_SCHEMA_VERSION = 1
CDM_DELAY_STAGES = ("capture_to_hard_binary", "hard_binary_to_gw_regime")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_STATUSES = {"complete", "timeout", "censored", "missing", "invalid"}
_CALIBRATION_METHOD = "resolved_cdm_rate_integration"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value.lower()


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and non-negative") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


def _positive(value: Any, label: str) -> float:
    parsed = _nonnegative(value, label)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return parsed


def _resolve(reference: str, base: Path) -> Path:
    path = Path(reference).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True)
class CDMDelayStageSummary:
    """One environmental CDM interval, or an explicit non-complete outcome."""

    source_path: Path
    source_sha256: str
    stage: str
    segment: DelaySegment
    start_separation_pc: float | None
    end_separation_pc: float | None
    physics_id: str | None
    phase_ensemble_path: Path | None
    phase_ensemble_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "summary_path": str(self.source_path),
            "summary_sha256": self.source_sha256,
            "segment": {
                "name": self.segment.name,
                "status": self.segment.status,
                "delay_myr": self.segment.delay_myr,
                "elapsed_lower_bound_myr": self.segment.elapsed_lower_bound_myr,
                "reason": self.segment.reason,
                "source_case_id": self.segment.source_case_id,
                "source_sha256": self.segment.source_sha256,
            },
            "interval_pc": None
            if self.start_separation_pc is None
            else {
                "start_separation_pc": self.start_separation_pc,
                "end_separation_pc": self.end_separation_pc,
            },
            "physics_id": self.physics_id,
            "phase_ensemble": None
            if self.phase_ensemble_path is None
            else {
                "path": str(self.phase_ensemble_path),
                "sha256": self.phase_ensemble_sha256,
            },
        }


def _accepted_phase_ensemble(
    path: Path,
    expected_sha256: str,
    expected_physics_id: str,
) -> None:
    try:
        actual_sha256 = _file_sha256(path)
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM phase ensemble: {error}") from error
    if actual_sha256 != expected_sha256:
        raise ValueError("CDM phase ensemble SHA-256 differs from stage summary")
    if not isinstance(record, Mapping):
        raise ValueError("CDM phase ensemble must be a JSON object")
    if record.get("schema_version") != 1:
        raise ValueError("CDM phase ensemble schema_version is unsupported")
    if record.get("status") != "accepted_model_specific_phase_ensemble":
        raise ValueError("CDM phase ensemble was not accepted")
    if record.get("dark_matter_model") != "cdm":
        raise ValueError("CDM delay stages cannot use a non-CDM phase ensemble")
    if record.get("physics_id") != expected_physics_id:
        raise ValueError("CDM phase ensemble physics_id differs from stage summary")


def _read_completed_calibration(
    record: Any,
    *,
    base: Path,
) -> tuple[str, Path, str]:
    if not isinstance(record, Mapping) or set(record) != {
        "method",
        "phase_ensemble_path",
        "phase_ensemble_sha256",
        "physics_id",
    }:
        raise ValueError("completed CDM stage calibration fields are invalid")
    if record.get("method") != _CALIBRATION_METHOD:
        raise ValueError(
            "completed CDM stage must use resolved_cdm_rate_integration; "
            "fixed or HR5 population delays are not accepted"
        )
    physics_id = _nonempty(record.get("physics_id"), "CDM physics_id")
    reference = _nonempty(record.get("phase_ensemble_path"), "phase_ensemble_path")
    digest = _sha256(record.get("phase_ensemble_sha256"), "phase_ensemble_sha256")
    ensemble_path = _resolve(reference, base)
    _accepted_phase_ensemble(ensemble_path, digest, physics_id)
    return physics_id, ensemble_path, digest


def read_cdm_delay_stage_summary(
    path: str | Path,
    *,
    expected_stage: str,
) -> CDMDelayStageSummary:
    """Read one strict CDM interval record without treating it as a run command."""

    if expected_stage not in CDM_DELAY_STAGES:
        raise ValueError("expected_stage is not a CDM delay stage")
    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM delay stage summary: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "dark_matter_model",
        "stage",
        "delay_myr",
        "elapsed_lower_bound_myr",
        "reason",
        "interval_pc",
        "calibration",
    }
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise ValueError("CDM delay stage summary fields are invalid")
    if record.get("schema_version") != CDM_DELAY_STAGE_SCHEMA_VERSION:
        raise ValueError("CDM delay stage schema_version is unsupported")
    if record.get("dark_matter_model") != "cdm":
        raise ValueError("CDM delay stages require dark_matter_model=cdm")
    if record.get("stage") != expected_stage:
        raise ValueError("CDM delay stage does not match the requested interval")
    status = record.get("status")
    if status not in _STATUSES:
        raise ValueError("CDM delay stage status is unsupported")
    elapsed = _nonnegative(record.get("elapsed_lower_bound_myr"), "elapsed_lower_bound_myr")
    reason_value = record.get("reason")
    if reason_value is not None and (
        not isinstance(reason_value, str) or not reason_value.strip()
    ):
        raise ValueError("CDM delay stage reason must be non-empty text when supplied")
    reason = None if reason_value is None else reason_value.strip()
    if status != "complete":
        if record.get("delay_myr") is not None or record.get("interval_pc") is not None:
            raise ValueError("non-complete CDM stage cannot carry a delay or interval")
        if record.get("calibration") is not None:
            raise ValueError("non-complete CDM stage cannot claim an accepted calibration")
        if status in {"timeout", "censored", "invalid"} and reason is None:
            raise ValueError("non-complete CDM stage requires a reason")
        return CDMDelayStageSummary(
            source_path=source,
            source_sha256=_file_sha256(source),
            stage=expected_stage,
            segment=DelaySegment(
                expected_stage,
                status,
                None,
                elapsed,
                reason=reason,
            ),
            start_separation_pc=None,
            end_separation_pc=None,
            physics_id=None,
            phase_ensemble_path=None,
            phase_ensemble_sha256=None,
        )

    delay = _nonnegative(record.get("delay_myr"), "delay_myr")
    interval = record.get("interval_pc")
    if not isinstance(interval, Mapping) or set(interval) != {
        "start_separation_pc",
        "end_separation_pc",
    }:
        raise ValueError("completed CDM stage interval_pc fields are invalid")
    start = _positive(interval.get("start_separation_pc"), "start_separation_pc")
    end = _positive(interval.get("end_separation_pc"), "end_separation_pc")
    if start <= end:
        raise ValueError("completed CDM stage must evolve from larger to smaller separation")
    physics_id, ensemble_path, ensemble_sha256 = _read_completed_calibration(
        record.get("calibration"),
        base=source.parent,
    )
    source_sha256 = _file_sha256(source)
    return CDMDelayStageSummary(
        source_path=source,
        source_sha256=source_sha256,
        stage=expected_stage,
        segment=DelaySegment(
            expected_stage,
            "complete",
            delay,
            elapsed,
            reason=reason or "accepted CDM resolved-rate integration",
            source_case_id=physics_id,
            source_sha256=source_sha256,
        ),
        start_separation_pc=start,
        end_separation_pc=end,
        physics_id=physics_id,
        phase_ensemble_path=ensemble_path,
        phase_ensemble_sha256=ensemble_sha256,
    )


def peters_gravitational_wave_segment(
    *,
    mass1_msun: float,
    mass2_msun: float,
    start_separation_pc: float,
    eccentricity: float = 0.0,
) -> DelaySegment:
    """Make the final GW-only segment from explicitly supplied orbital data."""

    delay = peters_time_myr(
        mass1_msun,
        mass2_msun,
        start_separation_pc,
        eccentricity,
    )
    return DelaySegment(
        "gravitational_wave",
        "complete",
        delay,
        reason="orbit-averaged Peters gravitational-wave evolution",
        source_case_id="peters_orbit_averaged",
    )


def _invalidated(segment: DelaySegment, reason: str) -> DelaySegment:
    lower_bound = segment.elapsed_lower_bound_myr
    if segment.status == "complete" and segment.delay_myr is not None:
        lower_bound += segment.delay_myr
    return DelaySegment(
        segment.name,
        "invalid",
        None,
        lower_bound,
        reason=reason,
        source_case_id=segment.source_case_id,
        source_sha256=segment.source_sha256,
    )


def compose_cdm_coalescence_time(
    sink_time_myr: float,
    capture_to_hard: CDMDelayStageSummary,
    hard_to_gw: CDMDelayStageSummary,
    gravitational_wave: DelaySegment,
    *,
    gravitational_wave_start_separation_pc: float,
) -> TrueMergeEstimate:
    """Compose separate CDM stages only when their physical seams agree."""

    if gravitational_wave.name != "gravitational_wave" or gravitational_wave.status != "complete":
        raise ValueError("the CDM composer requires a completed gravitational-wave segment")
    gw_start = _positive(
        gravitational_wave_start_separation_pc,
        "gravitational_wave_start_separation_pc",
    )
    outer = capture_to_hard.segment
    inner = hard_to_gw.segment
    gw = gravitational_wave
    if outer.status == "complete" and inner.status == "complete":
        assert capture_to_hard.end_separation_pc is not None
        assert hard_to_gw.start_separation_pc is not None
        if not math.isclose(
            capture_to_hard.end_separation_pc,
            hard_to_gw.start_separation_pc,
            rel_tol=1.0e-10,
            abs_tol=0.0,
        ):
            inner = _invalidated(
                inner,
                "CDM capture-to-hard and hardening intervals do not share one handoff separation",
            )
    if inner.status == "complete":
        assert hard_to_gw.end_separation_pc is not None
        if not math.isclose(
            hard_to_gw.end_separation_pc,
            gw_start,
            rel_tol=1.0e-10,
            abs_tol=0.0,
        ):
            gw = _invalidated(
                gw,
                "CDM hardening and Peters intervals do not share one handoff separation",
            )
    return compose_delay_segments(sink_time_myr, (outer, inner, gw))
