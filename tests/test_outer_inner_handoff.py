from __future__ import annotations

import pytest

from fdm_smbh_delay.outer_inner_handoff import (
    HandoffRatePoint,
    HandoffSimilarityState,
    validate_outer_inner_handoff,
)


def _points(scale: float = 1.0):
    return tuple(
        HandoffRatePoint(
            separation_pc=separation,
            orbital_power_pc2_myr3=-100.0 * scale,
            orbital_torque_msun_pc2_myr=-20.0 * scale,
            eccentricity=0.2,
        )
        for separation in (10.0, 20.0, 40.0, 80.0)
    )


def _similarity(*, e: float = 0.2):
    return HandoffSimilarityState(
        fdm_particle_mass_ev=1.0e-21,
        soliton_mass_msun=1.0e9,
        core_radius_pc=2.2,
        eta_sp=0.38,
        binary_mass_ratio=0.5,
        binary_to_soliton_mass_ratio=0.1,
        eccentricity=e,
        semimajor_axis_over_core=1.0,
    )


def test_handoff_accepts_positive_width_and_matched_rates() -> None:
    decision = validate_outer_inner_handoff(
        outer_points=_points(),
        inner_points=_points(scale=1.05),
        outer_similarity=_similarity(),
        inner_similarity=_similarity(),
    )
    assert decision.accepted
    assert decision.overlap_low_pc == pytest.approx(10.0)
    assert decision.overlap_high_pc == pytest.approx(80.0)
    assert decision.matched_points == 4
    assert decision.maximum_power_fractional_difference < 0.20


def test_handoff_censors_disjoint_support() -> None:
    inner = tuple(
        HandoffRatePoint(200.0 + index * 10.0, -100.0, -20.0, 0.2)
        for index in range(4)
    )
    decision = validate_outer_inner_handoff(
        outer_points=_points(),
        inner_points=inner,
        outer_similarity=_similarity(),
        inner_similarity=_similarity(),
    )
    assert decision.status == "censored"
    assert "do not overlap" in decision.reasons[0]


def test_handoff_censors_rate_and_similarity_mismatch() -> None:
    decision = validate_outer_inner_handoff(
        outer_points=_points(),
        inner_points=_points(scale=2.0),
        outer_similarity=_similarity(),
        inner_similarity=HandoffSimilarityState(
            fdm_particle_mass_ev=1.0e-21,
            soliton_mass_msun=1.4e9,
            core_radius_pc=2.2,
            eta_sp=0.38,
            binary_mass_ratio=0.5,
            binary_to_soliton_mass_ratio=0.1,
            eccentricity=0.25,
            semimajor_axis_over_core=1.0,
        ),
    )
    assert decision.status == "censored"
    assert any("rate" in reason for reason in decision.reasons)
    assert any("similarity" in reason for reason in decision.reasons)


def test_handoff_rejects_force_sign_change() -> None:
    inner = list(_points())
    inner[2] = HandoffRatePoint(40.0, 1.0, -20.0, 0.2)
    decision = validate_outer_inner_handoff(
        outer_points=_points(),
        inner_points=tuple(inner),
        outer_similarity=_similarity(),
        inner_similarity=_similarity(),
    )
    assert decision.status == "censored"
    assert "orbital power changes sign across handoff" in decision.reasons


def test_handoff_censors_zero_exchange_instead_of_accepting_agreement() -> None:
    zero_points = tuple(
        HandoffRatePoint(separation, 0.0, 0.0, 0.2)
        for separation in (10.0, 20.0, 40.0, 80.0)
    )
    decision = validate_outer_inner_handoff(
        outer_points=zero_points,
        inner_points=zero_points,
        outer_similarity=_similarity(),
        inner_similarity=_similarity(),
    )
    assert decision.status == "censored"
    assert "unresolved" in decision.reasons[-1]
