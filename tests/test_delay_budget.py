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


def test_redshift_moves_forward_in_cosmic_time() -> None:
    assert cosmic_age_myr(redshift=1.0, h0_km_s_mpc=67.66, omega_m=0.3111) > 0.0
    z_coal = redshift_after_delay(z_sink=1.0, delay_myr=1000.0, h0_km_s_mpc=67.66, omega_m=0.3111)
    assert z_coal is not None
    assert 0.0 < z_coal < 1.0
    assert redshift_after_delay(
        z_sink=0.1, delay_myr=14000.0, h0_km_s_mpc=67.66, omega_m=0.3111
    ) is None
