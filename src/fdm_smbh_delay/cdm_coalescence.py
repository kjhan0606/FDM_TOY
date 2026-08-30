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


CDM_DELAY_STAGE_SCHEMA_VERSION = 2
CDM_RATE_TRACK_SCHEMA_VERSION = 2
CDM_DELAY_STAGES = ("capture_to_hard_binary", "hard_binary_to_gw_regime")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_STATUSES = {"complete", "timeout", "censored", "missing", "invalid"}
_CALIBRATION_METHOD = "resolved_cdm_rate_integration"
_RATE_ESTIMATOR_METHOD = "blocked_log_separation_linear_regression_v1"


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
    rate_track_path: Path | None
    rate_track_sha256: str | None

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
            "rate_track": None
            if self.rate_track_path is None
            else {
                "path": str(self.rate_track_path),
                "sha256": self.rate_track_sha256,
            },
        }


@dataclass(frozen=True)
class CDMResolvedRateTrack:
    """Measured same-model logarithmic separation-rate samples for one stage."""

    source_path: Path
    source_sha256: str
    stage: str
    physics_id: str
    separations_pc: tuple[float, ...]
    logarithmic_decay_rates_per_myr: tuple[float, ...]

    @property
    def start_separation_pc(self) -> float:
        return self.separations_pc[0]

    @property
    def end_separation_pc(self) -> float:
        return self.separations_pc[-1]

    @property
    def integrated_delay_myr(self) -> float:
        delay = 0.0
        for left_a, right_a, left_rate, right_rate in zip(
            self.separations_pc[:-1],
            self.separations_pc[1:],
            self.logarithmic_decay_rates_per_myr[:-1],
            self.logarithmic_decay_rates_per_myr[1:],
        ):
            width = math.log(left_a / right_a)
            if math.isclose(left_rate, right_rate, rel_tol=1.0e-12, abs_tol=0.0):
                delay += width / left_rate
            else:
                delay += width * math.log(right_rate / left_rate) / (right_rate - left_rate)
        return delay


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


def _raw_orbit_samples(path: Path, expected_physics_id: str) -> list[tuple[float, float]]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM raw orbit track: {error}") from error
    if not isinstance(record, Mapping):
        raise ValueError("CDM raw orbit track must be a JSON object")
    if (
        record.get("schema_version") != 1
        or record.get("status") != "raw_relative_orbit_track"
        or record.get("dark_matter_model") != "cdm"
        or record.get("physics_id") != expected_physics_id
    ):
        raise ValueError("CDM raw orbit track identity is invalid")
    samples = record.get("samples")
    if not isinstance(samples, list):
        raise ValueError("CDM raw orbit track samples are invalid")
    parsed: list[tuple[float, float]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"CDM raw orbit sample {index} is invalid")
        parsed.append(
            (
                _nonnegative(sample.get("time_myr"), f"raw orbit sample {index} time_myr"),
                _positive(sample.get("separation_pc"), f"raw orbit sample {index} separation_pc"),
            )
        )
    if any(left[0] >= right[0] for left, right in zip(parsed[:-1], parsed[1:])):
        raise ValueError("CDM raw orbit sample times must be strictly increasing")
    return parsed


def _block_rate_points(
    samples: list[tuple[float, float]],
    *,
    samples_per_block: int,
) -> list[dict[str, float]]:
    if samples_per_block < 5:
        raise ValueError("samples_per_block must be at least five")
    block_count = len(samples) // samples_per_block
    if block_count < 3:
        raise ValueError("raw orbit track requires at least three complete regression blocks")
    points: list[dict[str, float]] = []
    for block_index in range(block_count):
        block = samples[
            block_index * samples_per_block : (block_index + 1) * samples_per_block
        ]
        mean_time = sum(item[0] for item in block) / samples_per_block
        logarithms = [math.log(item[1]) for item in block]
        mean_log_separation = sum(logarithms) / samples_per_block
        denominator = sum((time - mean_time) ** 2 for time, _ in block)
        if denominator <= 0.0:
            raise ValueError("raw orbit regression block has no time span")
        slope = sum(
            (time - mean_time) * (log_separation - mean_log_separation)
            for (time, _), log_separation in zip(block, logarithms)
        ) / denominator
        if not math.isfinite(slope) or slope >= 0.0:
            raise ValueError("raw orbit regression block is not secularly decaying")
        points.append(
            {
                "separation_pc": math.exp(mean_log_separation),
                "dln_separation_dt_per_myr": slope,
            }
        )
    if any(
        left["separation_pc"] <= right["separation_pc"]
        for left, right in zip(points[:-1], points[1:])
    ):
        raise ValueError("orbit-averaged separations are not strictly decreasing")
    return points


def derive_cdm_secular_rate_track(
    raw_orbit_track_path: str | Path,
    *,
    stage: str,
    samples_per_block: int,
) -> dict[str, Any]:
    """Make a reproducible candidate rate table from fixed raw-orbit blocks."""

    if stage not in CDM_DELAY_STAGES:
        raise ValueError("stage is not a CDM delay stage")
    source = Path(raw_orbit_track_path).expanduser().resolve()
    try:
        raw_record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM raw orbit track: {error}") from error
    if not isinstance(raw_record, Mapping):
        raise ValueError("CDM raw orbit track must be a JSON object")
    physics_id = _nonempty(raw_record.get("physics_id"), "CDM raw orbit physics_id")
    samples = _raw_orbit_samples(source, physics_id)
    points = _block_rate_points(samples, samples_per_block=samples_per_block)
    return {
        "schema_version": CDM_RATE_TRACK_SCHEMA_VERSION,
        "status": "complete",
        "dark_matter_model": "cdm",
        "stage": stage,
        "physics_id": physics_id,
        "estimator": {
            "method": _RATE_ESTIMATOR_METHOD,
            "raw_orbit_track_path": str(source),
            "raw_orbit_track_sha256": _file_sha256(source),
            "samples_per_block": samples_per_block,
        },
        "rate_points": points,
    }


def read_cdm_resolved_rate_track(
    path: str | Path,
    *,
    expected_stage: str,
    expected_physics_id: str,
) -> CDMResolvedRateTrack:
    """Read a reproducible CDM decay-rate track without extrapolating support."""

    if expected_stage not in CDM_DELAY_STAGES:
        raise ValueError("expected_stage is not a CDM delay stage")
    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM resolved rate track: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "dark_matter_model",
        "stage",
        "physics_id",
        "estimator",
        "rate_points",
    }
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise ValueError("CDM resolved rate-track fields are invalid")
    if (
        record.get("schema_version") != CDM_RATE_TRACK_SCHEMA_VERSION
        or record.get("status") != "complete"
    ):
        raise ValueError("CDM resolved rate track must be a complete schema-v2 record")
    if record.get("dark_matter_model") != "cdm":
        raise ValueError("CDM resolved rate track requires dark_matter_model=cdm")
    if record.get("stage") != expected_stage:
        raise ValueError("CDM resolved rate track stage differs from the delay stage")
    if record.get("physics_id") != expected_physics_id:
        raise ValueError("CDM resolved rate track physics_id differs from the phase ensemble")
    estimator = record.get("estimator")
    if not isinstance(estimator, Mapping) or set(estimator) != {
        "method",
        "raw_orbit_track_path",
        "raw_orbit_track_sha256",
        "samples_per_block",
    } or estimator.get("method") != _RATE_ESTIMATOR_METHOD:
        raise ValueError("CDM resolved rate-track estimator is invalid")
    reference = _nonempty(estimator.get("raw_orbit_track_path"), "raw_orbit_track_path")
    raw_path = _resolve(reference, source.parent)
    expected_raw_sha256 = _sha256(
        estimator.get("raw_orbit_track_sha256"),
        "raw_orbit_track_sha256",
    )
    if _file_sha256(raw_path) != expected_raw_sha256:
        raise ValueError("CDM raw orbit track SHA-256 differs from rate track")
    block_size = estimator.get("samples_per_block")
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError("samples_per_block must be an integer")
    raw_samples = _raw_orbit_samples(raw_path, expected_physics_id)
    expected_points = _block_rate_points(raw_samples, samples_per_block=block_size)
    points = record.get("rate_points")
    if not isinstance(points, list) or len(points) != len(expected_points):
        raise ValueError("CDM resolved rate track points differ from its raw orbit track")
    separations: list[float] = []
    rates: list[float] = []
    for index, (point, expected) in enumerate(zip(points, expected_points)):
        if not isinstance(point, Mapping) or set(point) != {
            "separation_pc",
            "dln_separation_dt_per_myr",
        }:
            raise ValueError(f"CDM resolved rate point {index} fields are invalid")
        separation = _positive(point.get("separation_pc"), f"rate point {index} separation_pc")
        raw_rate = point.get("dln_separation_dt_per_myr")
        if isinstance(raw_rate, bool):
            raise ValueError("CDM resolved rate track must have negative dln_separation_dt_per_myr")
        try:
            signed_rate = float(raw_rate)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "CDM resolved rate track must have negative dln_separation_dt_per_myr"
            ) from error
        if not math.isfinite(signed_rate) or signed_rate >= 0.0:
            raise ValueError("CDM resolved rate track must have negative dln_separation_dt_per_myr")
        if not math.isclose(separation, expected["separation_pc"], rel_tol=1.0e-12):
            raise ValueError("CDM rate separation differs from its raw orbit regression")
        if not math.isclose(signed_rate, expected["dln_separation_dt_per_myr"], rel_tol=1.0e-12):
            raise ValueError("CDM rate differs from its raw orbit regression")
        separations.append(separation)
        rates.append(-signed_rate)
    return CDMResolvedRateTrack(
        source_path=source,
        source_sha256=_file_sha256(source),
        stage=expected_stage,
        physics_id=expected_physics_id,
        separations_pc=tuple(separations),
        logarithmic_decay_rates_per_myr=tuple(rates),
    )


def _read_completed_calibration(
    record: Any,
    *,
    base: Path,
    stage: str,
) -> tuple[str, Path, str, CDMResolvedRateTrack]:
    if not isinstance(record, Mapping) or set(record) != {
        "method",
        "phase_ensemble_path",
        "phase_ensemble_sha256",
        "physics_id",
        "rate_track_path",
        "rate_track_sha256",
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
    rate_reference = _nonempty(record.get("rate_track_path"), "rate_track_path")
    rate_digest = _sha256(record.get("rate_track_sha256"), "rate_track_sha256")
    rate_path = _resolve(rate_reference, base)
    track = read_cdm_resolved_rate_track(
        rate_path,
        expected_stage=stage,
        expected_physics_id=physics_id,
    )
    if track.source_sha256 != rate_digest:
        raise ValueError("CDM resolved rate-track SHA-256 differs from stage summary")
    return physics_id, ensemble_path, digest, track


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
            rate_track_path=None,
            rate_track_sha256=None,
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
    physics_id, ensemble_path, ensemble_sha256, rate_track = _read_completed_calibration(
        record.get("calibration"),
        base=source.parent,
        stage=expected_stage,
    )
    if not math.isclose(
        start,
        rate_track.start_separation_pc,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ) or not math.isclose(
        end,
        rate_track.end_separation_pc,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError("CDM stage interval differs from its resolved rate-track support")
    if not math.isclose(
        delay,
        rate_track.integrated_delay_myr,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError("CDM stage delay differs from its resolved rate-track integral")
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
        rate_track_path=rate_track.source_path,
        rate_track_sha256=rate_track.source_sha256,
    )


def build_cdm_delay_stage_record(
    rate_track_path: str | Path,
    phase_ensemble_path: str | Path,
) -> dict[str, Any]:
    """Integrate one measured CDM rate track into a self-verifying stage record.

    The rate track supplies only its resolved separation support.  This helper
    never extends the first or final interval, interpolates to another physics
    point, or replaces a rejected/missing phase ensemble with a population
    delay.
    """

    phase_path = Path(phase_ensemble_path).expanduser().resolve()
    try:
        phase_record = json.loads(phase_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM phase ensemble: {error}") from error
    if not isinstance(phase_record, Mapping):
        raise ValueError("CDM phase ensemble must be a JSON object")
    physics_id = _nonempty(phase_record.get("physics_id"), "CDM phase ensemble physics_id")
    phase_sha256 = _file_sha256(phase_path)
    _accepted_phase_ensemble(phase_path, phase_sha256, physics_id)

    raw_track_path = Path(rate_track_path).expanduser().resolve()
    try:
        raw_track = json.loads(raw_track_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM resolved rate track: {error}") from error
    if not isinstance(raw_track, Mapping):
        raise ValueError("CDM resolved rate track must be a JSON object")
    stage = raw_track.get("stage")
    if stage not in CDM_DELAY_STAGES:
        raise ValueError("CDM resolved rate track stage is unsupported")
    track = read_cdm_resolved_rate_track(
        raw_track_path,
        expected_stage=stage,
        expected_physics_id=physics_id,
    )
    return {
        "schema_version": CDM_DELAY_STAGE_SCHEMA_VERSION,
        "status": "complete",
        "dark_matter_model": "cdm",
        "stage": track.stage,
        "delay_myr": track.integrated_delay_myr,
        "elapsed_lower_bound_myr": 0.0,
        "reason": "accepted CDM resolved logarithmic-separation-rate integration",
        "interval_pc": {
            "start_separation_pc": track.start_separation_pc,
            "end_separation_pc": track.end_separation_pc,
        },
        "calibration": {
            "method": _CALIBRATION_METHOD,
            "phase_ensemble_path": str(phase_path),
            "phase_ensemble_sha256": phase_sha256,
            "physics_id": physics_id,
            "rate_track_path": str(track.source_path),
            "rate_track_sha256": track.source_sha256,
        },
    }


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
