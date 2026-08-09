"""Compose numerical sink events and physical delay segments conservatively."""

from __future__ import annotations

from dataclasses import dataclass

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

    def __post_init__(self) -> None:
        if self.status not in {"complete", "timeout", "missing", "invalid"}:
            raise ValueError(f"unsupported segment status={self.status!r}")
        if self.status == "complete" and (self.delay_myr is None or self.delay_myr < 0.0):
            raise ValueError("complete segments require a non-negative delay")
        if self.elapsed_lower_bound_myr < 0.0:
            raise ValueError("elapsed lower bound must be non-negative")


@dataclass(frozen=True)
class TrueMergeEstimate:
    status: str
    sink_time_myr: float
    total_delay_myr: float | None
    true_merge_time_myr: float | None
    delay_lower_bound_myr: float
    missing_segments: tuple[str, ...]


def compose_true_merge_time(
    sink_time_myr: float,
    kpc_to_pc: DelaySegment,
    fdm_pc_to_0p01pc: DelaySegment,
    gravitational_wave: DelaySegment,
) -> TrueMergeEstimate:
    """Add three physical intervals without treating missing physics as zero."""

    if sink_time_myr < 0.0:
        raise ValueError("sink_time_myr must be non-negative")
    segments = (kpc_to_pc, fdm_pc_to_0p01pc, gravitational_wave)
    missing = tuple(segment.name for segment in segments if segment.status == "missing")
    invalid = any(segment.status == "invalid" for segment in segments)
    timed_out = any(segment.status == "timeout" for segment in segments)
    lower_bound = sum(
        segment.delay_myr
        if segment.status == "complete" and segment.delay_myr is not None
        else segment.elapsed_lower_bound_myr
        for segment in segments
    )
    if invalid:
        status = "invalid"
    elif timed_out:
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
    return TrueMergeEstimate(status, sink_time_myr, total, true_time, lower_bound, missing)


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
