from __future__ import annotations

import hashlib

import pytest

from fdm_smbh_delay.backreaction import (
    BackreactionEvidence,
    BackreactionTrackPoint,
    assess_live_frozen_backreaction,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _evidence(
    *,
    model: str = "cdm",
    live_checkpoint_id: str = "checkpoint-1",
    frozen_checkpoint_id: str = "checkpoint-1",
    live_energy: float = 1.0e-4,
    frozen_energy: float = 2.0e-4,
    live_resolution: float = 8.0,
    frozen_resolution: float = 8.0,
    live_accounting: str = "live_resolved",
) -> BackreactionEvidence:
    if model == "fdm":
        live_accounting = "live_wave_only"
    return BackreactionEvidence(
        model=model,
        live_checkpoint_id=live_checkpoint_id,
        frozen_checkpoint_id=frozen_checkpoint_id,
        live_source_path="live-track.json",
        live_source_sha256=_sha("live"),
        frozen_source_path="frozen-track.json",
        frozen_source_sha256=_sha("frozen"),
        live_force_accounting=live_accounting,
        frozen_force_accounting="frozen_background",
        maximum_live_relative_energy_error=live_energy,
        maximum_frozen_relative_energy_error=frozen_energy,
        minimum_live_orbital_resolution_cells=live_resolution,
        minimum_frozen_orbital_resolution_cells=frozen_resolution,
    )


def _tracks(
    *,
    frozen_power_scale: float = 1.0,
    frozen_torque_scale: float = 1.0,
    frozen_eccentricity_offset: float = 0.0,
):
    separations = (1.0, 2.0, 4.0, 8.0)
    live = tuple(
        BackreactionTrackPoint(
            separation_pc=radius,
            orbital_power_pc2_myr3=-2.0 / radius,
            orbital_torque_msun_pc2_myr=-1.0 / radius,
            eccentricity=0.2,
        )
        for radius in separations
    )
    frozen = tuple(
        BackreactionTrackPoint(
            separation_pc=radius,
            orbital_power_pc2_myr3=-2.0 * frozen_power_scale / radius,
            orbital_torque_msun_pc2_myr=-1.0 * frozen_torque_scale / radius,
            eccentricity=0.2 + frozen_eccentricity_offset,
        )
        for radius in separations
    )
    return live, frozen


def test_matching_pair_allows_offline_closure() -> None:
    live, frozen = _tracks()
    decision = assess_live_frozen_backreaction(
        model="cdm",
        live_points=live,
        frozen_points=frozen,
        evidence=_evidence(),
    )
    assert decision.offline_acceptable
    assert decision.matched_points == 4
    assert decision.maximum_power_fractional_difference == pytest.approx(0.0)


def test_large_rate_change_requires_runtime_treatment() -> None:
    live, frozen = _tracks(frozen_power_scale=1.5, frozen_torque_scale=1.0)
    decision = assess_live_frozen_backreaction(
        model="sidm",
        live_points=live,
        frozen_points=frozen,
        evidence=_evidence(model="sidm"),
    )
    assert decision.runtime_required
    assert any("power differs" in reason for reason in decision.reasons)


def test_sign_reversal_requires_runtime_treatment() -> None:
    live, frozen = _tracks()
    frozen = tuple(
        BackreactionTrackPoint(
            point.separation_pc,
            -point.orbital_power_pc2_myr3,
            point.orbital_torque_msun_pc2_myr,
            point.eccentricity,
        )
        if point.separation_pc == 2.0
        else point
        for point in frozen
    )
    decision = assess_live_frozen_backreaction(
        model="cdm",
        live_points=live,
        frozen_points=frozen,
        evidence=_evidence(),
    )
    assert decision.runtime_required
    assert any("power changes sign" in reason for reason in decision.reasons)


def test_unresolved_rate_is_censored_not_zero_agreement() -> None:
    live, frozen = _tracks()
    frozen = tuple(
        BackreactionTrackPoint(
            point.separation_pc,
            0.0,
            point.orbital_torque_msun_pc2_myr,
            point.eccentricity,
        )
        if point.separation_pc == 4.0
        else point
        for point in frozen
    )
    decision = assess_live_frozen_backreaction(
        model="fdm",
        live_points=live,
        frozen_points=frozen,
        evidence=_evidence(model="fdm"),
    )
    assert decision.censored
    assert any("unresolved" in reason for reason in decision.reasons)


def test_bad_quality_or_checkpoint_is_censored() -> None:
    live, frozen = _tracks()
    decision = assess_live_frozen_backreaction(
        model="cdm",
        live_points=live,
        frozen_points=frozen,
        evidence=_evidence(
            frozen_checkpoint_id="checkpoint-other",
            live_energy=2.0e-3,
            frozen_resolution=2.0,
        ),
    )
    assert decision.censored
    assert any("checkpoint" in reason for reason in decision.reasons)
    assert any("conservation" in reason for reason in decision.reasons)
    assert any("resolution" in reason for reason in decision.reasons)


def test_fdm_live_track_must_be_wave_only() -> None:
    with pytest.raises(ValueError, match="live_wave_only"):
        BackreactionEvidence(
            model="fdm",
            live_checkpoint_id="checkpoint-1",
            frozen_checkpoint_id="checkpoint-1",
            live_source_path="live",
            live_source_sha256=_sha("live"),
            frozen_source_path="frozen",
            frozen_source_sha256=_sha("frozen"),
            live_force_accounting="analytic_fdm_drag",
            frozen_force_accounting="frozen_background",
            maximum_live_relative_energy_error=0.0,
            maximum_frozen_relative_energy_error=0.0,
            minimum_live_orbital_resolution_cells=4.0,
            minimum_frozen_orbital_resolution_cells=4.0,
        )


def test_no_overlap_is_censored() -> None:
    live, _ = _tracks()
    frozen = tuple(
        BackreactionTrackPoint(
            point.separation_pc + 20.0,
            point.orbital_power_pc2_myr3,
            point.orbital_torque_msun_pc2_myr,
            point.eccentricity,
        )
        for point in live
    )
    decision = assess_live_frozen_backreaction(
        model="cdm",
        live_points=live,
        frozen_points=frozen,
        evidence=_evidence(),
    )
    assert decision.censored
    assert decision.matched_points == 0
