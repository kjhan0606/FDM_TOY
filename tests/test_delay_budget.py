from __future__ import annotations

import pytest

from fdm_smbh_delay.delay_budget import (
    DelaySegment,
    compose_true_merge_time,
    cosmic_age_myr,
    redshift_after_delay,
)


def test_complete_true_merge_budget() -> None:
    result = compose_true_merge_time(
        1000.0,
        DelaySegment("kpc_to_pc", "complete", 20.0),
        DelaySegment("fdm", "complete", 30.0),
        DelaySegment("gw", "complete", 4.0),
    )
    assert result.status == "complete"
    assert result.total_delay_myr == pytest.approx(54.0)
    assert result.true_merge_time_myr == pytest.approx(1054.0)


def test_missing_or_timeout_is_never_silently_zero() -> None:
    incomplete = compose_true_merge_time(
        1000.0,
        DelaySegment("kpc_to_pc", "missing", None),
        DelaySegment("fdm", "complete", 30.0),
        DelaySegment("gw", "complete", 4.0),
    )
    assert incomplete.status == "incomplete"
    assert incomplete.true_merge_time_myr is None
    assert incomplete.missing_segments == ("kpc_to_pc",)

    censored = compose_true_merge_time(
        1000.0,
        DelaySegment("kpc_to_pc", "complete", 20.0),
        DelaySegment("fdm", "timeout", None, elapsed_lower_bound_myr=500.0),
        DelaySegment("gw", "complete", 4.0),
    )
    assert censored.status == "censored"
    assert censored.delay_lower_bound_myr == pytest.approx(524.0)
    assert censored.total_delay_myr is None
    assert censored.censored_segments == ("fdm",)


def test_explicit_calibration_censoring_is_not_missing_or_zero() -> None:
    result = compose_true_merge_time(
        1000.0,
        DelaySegment(
            "kpc_to_pc",
            "censored",
            None,
            elapsed_lower_bound_myr=7.0,
            reason="outside accepted zoom support",
        ),
        DelaySegment("fdm", "complete", 30.0),
        DelaySegment("gw", "missing", None),
    )
    assert result.status == "censored"
    assert result.total_delay_myr is None
    assert result.delay_lower_bound_myr == pytest.approx(37.0)
    assert result.censored_segments == ("kpc_to_pc",)
    assert result.missing_segments == ("gw",)
    assert result.segments[0].reason == "outside accepted zoom support"


def test_invalid_segment_still_takes_precedence_over_censoring() -> None:
    result = compose_true_merge_time(
        1000.0,
        DelaySegment("kpc_to_pc", "censored", None),
        DelaySegment("fdm", "invalid", None),
        DelaySegment("gw", "complete", 4.0),
    )
    assert result.status == "invalid"


@pytest.mark.parametrize("delay", [float("nan"), float("inf"), -1.0])
def test_complete_segment_rejects_nonphysical_delay(delay: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        DelaySegment("fdm", "complete", delay)


@pytest.mark.parametrize("status", ["timeout", "censored", "missing", "invalid"])
def test_noncomplete_segment_rejects_completed_delay(status: str) -> None:
    with pytest.raises(ValueError, match="cannot carry a completed delay"):
        DelaySegment("fdm", status, 1.0)


@pytest.mark.parametrize("reason", ["", "   ", 17])
def test_segment_rejects_malformed_reason(reason) -> None:
    with pytest.raises(ValueError, match="reason must be non-empty text"):
        DelaySegment("fdm", "censored", None, reason=reason)


@pytest.mark.parametrize(
    "source_sha256",
    ["", "a" * 63, "a" * 65, "g" * 64, 64],
)
def test_segment_rejects_malformed_source_hash(source_sha256) -> None:
    with pytest.raises(ValueError, match="exactly 64 hexadecimal"):
        DelaySegment(
            "kpc_to_pc",
            "complete",
            1.0,
            source_sha256=source_sha256,
        )


@pytest.mark.parametrize("sink_time", [float("nan"), float("inf"), -1.0, True])
def test_composer_rejects_nonphysical_sink_time(sink_time: float) -> None:
    complete = DelaySegment("complete", "complete", 1.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        compose_true_merge_time(sink_time, complete, complete, complete)


def test_redshift_moves_forward_in_cosmic_time() -> None:
    assert cosmic_age_myr(redshift=1.0, h0_km_s_mpc=67.66, omega_m=0.3111) > 0.0
    z_coal = redshift_after_delay(z_sink=1.0, delay_myr=1000.0, h0_km_s_mpc=67.66, omega_m=0.3111)
    assert z_coal is not None
    assert 0.0 < z_coal < 1.0
    assert redshift_after_delay(
        z_sink=0.1, delay_myr=14000.0, h0_km_s_mpc=67.66, omega_m=0.3111
    ) is None
