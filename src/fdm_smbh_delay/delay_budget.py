"""Compose numerical sink events and physical delay segments conservatively."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import brentq


def cosmic_age_myr(*, redshift: float, h0_km_s_mpc: float, omega_m: float) -> float:
    """Cosmic age in a flat matter-plus-Lambda cosmology."""

    if redshift < 0.0:
        raise ValueError("redshift must be non-negative")
    cosmology = FlatLambdaCDM(H0=h0_km_s_mpc, Om0=omega_m, Tcmb0=0.0 * u.K)
    return float(cosmology.age(redshift).to_value(u.Myr))


@dataclass(frozen=True)
class DelaySegment:
    name: str
    status: str
    delay_myr: float | None
    elapsed_lower_bound_myr: float = 0.0
    reason: str | None = None
    source_case_id: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "complete",
            "timeout",
            "censored",
            "missing",
            "invalid",
        }:
            raise ValueError(f"unsupported segment status={self.status!r}")
        if self.status == "complete" and (
            self.delay_myr is None
            or isinstance(self.delay_myr, bool)
            or not math.isfinite(self.delay_myr)
            or self.delay_myr < 0.0
        ):
            raise ValueError("complete segments require a finite non-negative delay")
        if self.status != "complete" and self.delay_myr is not None:
            raise ValueError("non-complete segments cannot carry a completed delay")
        if (
            isinstance(self.elapsed_lower_bound_myr, bool)
            or not math.isfinite(self.elapsed_lower_bound_myr)
            or self.elapsed_lower_bound_myr < 0.0
        ):
            raise ValueError("elapsed lower bound must be finite and non-negative")
        if self.reason is not None and (
            not isinstance(self.reason, str) or not self.reason.strip()
        ):
            raise ValueError("segment reason must be non-empty text when supplied")
        if self.source_case_id is not None and (
            not isinstance(self.source_case_id, str) or not self.source_case_id.strip()
        ):
            raise ValueError("source_case_id must be non-empty text when supplied")
        if self.source_sha256 is not None and (
            not isinstance(self.source_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", self.source_sha256) is None
        ):
            raise ValueError("source_sha256 must be exactly 64 hexadecimal characters")


_DELAY_RECORD_FIELDS = {
    "name",
    "status",
    "delay_myr",
    "elapsed_lower_bound_myr",
    "reason",
    "source_case_id",
    "source_sha256",
}


def read_verified_delay_segment_record(
    path: str | Path, *, expected_name: str
) -> DelaySegment:
    """Read one serialized delay segment without accepting a hand-typed delay.

    Completed records must carry both a case identifier and a SHA-256 digest.
    The digest is an upstream evidence identity; this reader deliberately does
    not infer a physical interval from a bare number supplied on a command
    line.  Non-complete records remain useful as explicit censoring evidence.
    """

    expected_name = expected_name.strip()
    if not expected_name:
        raise ValueError("expected delay segment name must be non-empty")
    source = Path(path).expanduser().resolve()
    try:
        record: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read delay segment record: {error}") from error
    if not isinstance(record, dict) or set(record) != _DELAY_RECORD_FIELDS:
        raise ValueError("delay segment record fields are invalid")
    if record.get("name") != expected_name:
        raise ValueError("delay segment record name differs from requested interval")
    try:
        segment = DelaySegment(**record)
    except (TypeError, ValueError) as error:
        raise ValueError(f"delay segment record is invalid: {error}") from error
    if segment.status == "complete":
        if segment.source_case_id is None or segment.source_sha256 is None:
            raise ValueError(
                "completed delay segment records require source_case_id and source_sha256"
            )
    return segment


@dataclass(frozen=True)
class TrueMergeEstimate:
    status: str
    sink_time_myr: float
    total_delay_myr: float | None
    true_merge_time_myr: float | None
    delay_lower_bound_myr: float
    missing_segments: tuple[str, ...]
    censored_segments: tuple[str, ...] = ()
    segments: tuple[DelaySegment, ...] = ()


def compose_true_merge_time(
    sink_time_myr: float,
    kpc_to_pc: DelaySegment,
    fdm_pc_to_0p01pc: DelaySegment,
    gravitational_wave: DelaySegment,
) -> TrueMergeEstimate:
    """Add three physical intervals without treating missing physics as zero."""

    return compose_delay_segments(
        sink_time_myr,
        (kpc_to_pc, fdm_pc_to_0p01pc, gravitational_wave),
    )


def compose_delay_segments(
    sink_time_myr: float,
    segments: tuple[DelaySegment, ...],
) -> TrueMergeEstimate:
    """Compose named physical intervals without assigning missing time to zero.

    This is deliberately model-neutral.  Callers retain their model-specific
    segment names and must establish the physics of every completed segment
    before asking this function to add it.
    """

    if (
        isinstance(sink_time_myr, bool)
        or not math.isfinite(sink_time_myr)
        or sink_time_myr < 0.0
    ):
        raise ValueError("sink_time_myr must be finite and non-negative")
    if not segments:
        raise ValueError("at least one physical delay segment is required")
    names = tuple(segment.name for segment in segments)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("delay segment names must be non-empty text")
    if len(set(names)) != len(names):
        raise ValueError("delay segment names must be unique")
    missing = tuple(segment.name for segment in segments if segment.status == "missing")
    censored = tuple(
        segment.name
        for segment in segments
        if segment.status in {"timeout", "censored"}
    )
    invalid = any(segment.status == "invalid" for segment in segments)
    lower_bound = sum(
        segment.delay_myr
        if segment.status == "complete" and segment.delay_myr is not None
        else segment.elapsed_lower_bound_myr
        for segment in segments
    )
    if invalid:
        status = "invalid"
    elif censored:
        status = "censored"
    elif missing:
        status = "incomplete"
    else:
        status = "complete"
    if status == "complete":
        total = sum(segment.delay_myr or 0.0 for segment in segments)
        true_time = sink_time_myr + total
    else:
        total = None
        true_time = None
    return TrueMergeEstimate(
        status,
        sink_time_myr,
        total,
        true_time,
        lower_bound,
        missing,
        censored,
        segments,
    )


def redshift_after_delay(
    *, z_sink: float, delay_myr: float, h0_km_s_mpc: float, omega_m: float
) -> float | None:
    """Convert a completed physical delay to redshift in a flat cosmology.

    Returns ``None`` if the inferred coalescence lies after z=0.
    """

    if z_sink < 0.0 or delay_myr < 0.0:
        raise ValueError("z_sink and delay_myr must be non-negative")
    cosmology = FlatLambdaCDM(H0=h0_km_s_mpc, Om0=omega_m, Tcmb0=0.0 * u.K)
    age_sink = cosmic_age_myr(
        redshift=z_sink, h0_km_s_mpc=h0_km_s_mpc, omega_m=omega_m
    )
    target_age = age_sink + delay_myr
    age_now = cosmology.age(0.0).to_value(u.Myr)
    if target_age > age_now:
        return None
    if delay_myr == 0.0:
        return float(z_sink)

    def age_residual(redshift: float) -> float:
        return float(cosmology.age(redshift).to_value(u.Myr) - target_age)

    return float(brentq(age_residual, 0.0, z_sink, xtol=1.0e-12, rtol=1.0e-12))
