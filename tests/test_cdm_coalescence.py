from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from fdm_smbh_delay.cdm_coalescence import (
    build_cdm_delay_stage_record,
    compose_cdm_coalescence_time,
    derive_cdm_secular_rate_track,
    peters_gravitational_wave_segment,
    read_cdm_delay_stage_summary,
    read_cdm_resolved_rate_track,
)
from fdm_smbh_delay.cdm_true_time_cli import main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, record: dict[str, object]) -> Path:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _accepted_phase_ensemble(tmp_path: Path, *, model: str = "cdm") -> Path:
    return _write_json(
        tmp_path / f"phase_ensemble_{model}.json",
        {
            "schema_version": 1,
            "status": "accepted_model_specific_phase_ensemble",
            "dark_matter_model": model,
            "physics_id": "zoomphys-cdm-test",
            "replicates": [0, 1],
            "members": [],
            "reasons": [],
        },
    )


def _rate_track(
    tmp_path: Path,
    *,
    stage: str,
    delay_myr: float,
    start_pc: float,
    end_pc: float,
) -> Path:
    rate = -math.log(start_pc / end_pc) / delay_myr
    step = delay_myr / 10.0
    raw = _write_json(
        tmp_path / f"{stage}_raw_orbit.json",
        {
            "schema_version": 1,
            "status": "raw_relative_orbit_track",
            "dark_matter_model": "cdm",
            "physics_id": "zoomphys-cdm-test",
            "samples": [
                {
                    "time_myr": index * step,
                    "separation_pc": start_pc
                    * math.exp(rate * (index * step - 2.0 * step)),
                }
                for index in range(15)
            ],
        },
    )
    return _write_json(
        tmp_path / f"{stage}_rate_track.json",
        derive_cdm_secular_rate_track(raw, stage=stage, samples_per_block=5),
    )


def _complete_stage(
    tmp_path: Path,
    *,
    stage: str,
    delay_myr: float,
    start_pc: float,
    end_pc: float,
    phase_path: Path,
    method: str = "resolved_cdm_rate_integration",
) -> Path:
    rate_track = _rate_track(
        tmp_path,
        stage=stage,
        delay_myr=delay_myr,
        start_pc=start_pc,
        end_pc=end_pc,
    )
    return _write_json(
        tmp_path / f"{stage}.json",
        {
            "schema_version": 2,
            "status": "complete",
            "dark_matter_model": "cdm",
            "stage": stage,
            "delay_myr": delay_myr,
            "elapsed_lower_bound_myr": 0.0,
            "reason": "paired-resolution and phase-accepted rate integration",
            "interval_pc": {
                "start_separation_pc": start_pc,
                "end_separation_pc": end_pc,
            },
            "calibration": {
                "method": method,
                "phase_ensemble_path": phase_path.name,
                "phase_ensemble_sha256": _sha256(phase_path),
                "physics_id": "zoomphys-cdm-test",
                "rate_track_path": rate_track.name,
                "rate_track_sha256": _sha256(rate_track),
            },
        },
    )


def _censored_stage(tmp_path: Path, *, stage: str) -> Path:
    return _write_json(
        tmp_path / f"{stage}.json",
        {
            "schema_version": 2,
            "status": "censored",
            "dark_matter_model": "cdm",
            "stage": stage,
            "delay_myr": None,
            "elapsed_lower_bound_myr": 12.0,
            "reason": "outside accepted CDM resolution and phase support",
            "interval_pc": None,
            "calibration": None,
        },
    )


def test_completed_cdm_stages_require_accepted_cdm_phase_evidence(tmp_path: Path) -> None:
    phase = _accepted_phase_ensemble(tmp_path)
    stage = _complete_stage(
        tmp_path,
        stage="capture_to_hard_binary",
        delay_myr=20.0,
        start_pc=1000.0,
        end_pc=1.0,
        phase_path=phase,
    )
    parsed = read_cdm_delay_stage_summary(stage, expected_stage="capture_to_hard_binary")
    assert parsed.segment.status == "complete"
    assert parsed.segment.delay_myr == pytest.approx(20.0)
    assert parsed.physics_id == "zoomphys-cdm-test"
    assert parsed.phase_ensemble_sha256 == _sha256(phase)


def test_stage_builder_integrates_only_the_measured_rate_track_support(
    tmp_path: Path,
) -> None:
    phase = _accepted_phase_ensemble(tmp_path)
    track = _rate_track(
        tmp_path,
        stage="capture_to_hard_binary",
        delay_myr=20.0,
        start_pc=1000.0,
        end_pc=1.0,
    )
    record = build_cdm_delay_stage_record(track, phase)
    assert record["schema_version"] == 2
    assert record["delay_myr"] == pytest.approx(20.0)
    assert record["interval_pc"]["start_separation_pc"] == pytest.approx(1000.0)
    assert record["interval_pc"]["end_separation_pc"] == pytest.approx(1.0)
    summary = _write_json(tmp_path / "built_stage.json", record)
    parsed = read_cdm_delay_stage_summary(summary, expected_stage="capture_to_hard_binary")
    assert parsed.rate_track_sha256 == _sha256(track)

    record["delay_myr"] = 21.0
    tampered = _write_json(tmp_path / "tampered_stage.json", record)
    with pytest.raises(ValueError, match="differs from its resolved rate-track integral"):
        read_cdm_delay_stage_summary(tampered, expected_stage="capture_to_hard_binary")

    rate_record = json.loads(track.read_text(encoding="utf-8"))
    rate_record["rate_points"][0]["dln_separation_dt_per_myr"] *= 1.1
    forged_rate = _write_json(tmp_path / "forged_rate_track.json", rate_record)
    with pytest.raises(ValueError, match="differs from its raw orbit regression"):
        read_cdm_resolved_rate_track(
            forged_rate,
            expected_stage="capture_to_hard_binary",
            expected_physics_id="zoomphys-cdm-test",
        )


def test_fixed_population_delay_and_non_cdm_phase_evidence_are_rejected(tmp_path: Path) -> None:
    phase = _accepted_phase_ensemble(tmp_path)
    fixed = _complete_stage(
        tmp_path,
        stage="capture_to_hard_binary",
        delay_myr=20.0,
        start_pc=1000.0,
        end_pc=1.0,
        phase_path=phase,
        method="hr5_fixed_population_delay",
    )
    with pytest.raises(ValueError, match="fixed or HR5 population delays"):
        read_cdm_delay_stage_summary(fixed, expected_stage="capture_to_hard_binary")

    non_cdm_phase = _accepted_phase_ensemble(tmp_path, model="fdm")
    stage = _complete_stage(
        tmp_path,
        stage="hard_binary_to_gw_regime",
        delay_myr=4.0,
        start_pc=1.0,
        end_pc=0.01,
        phase_path=non_cdm_phase,
    )
    with pytest.raises(ValueError, match="non-CDM phase ensemble"):
        read_cdm_delay_stage_summary(stage, expected_stage="hard_binary_to_gw_regime")


def test_cdm_composition_requires_unbroken_environmental_and_gw_handoffs(
    tmp_path: Path,
) -> None:
    phase = _accepted_phase_ensemble(tmp_path)
    outer = read_cdm_delay_stage_summary(
        _complete_stage(
            tmp_path,
            stage="capture_to_hard_binary",
            delay_myr=20.0,
            start_pc=1000.0,
            end_pc=1.0,
            phase_path=phase,
        ),
        expected_stage="capture_to_hard_binary",
    )
    inner = read_cdm_delay_stage_summary(
        _complete_stage(
            tmp_path,
            stage="hard_binary_to_gw_regime",
            delay_myr=4.0,
            start_pc=1.0,
            end_pc=0.01,
            phase_path=phase,
        ),
        expected_stage="hard_binary_to_gw_regime",
    )
    gw = peters_gravitational_wave_segment(
        mass1_msun=1.0e8,
        mass2_msun=3.0e7,
        start_separation_pc=0.01,
        eccentricity=0.2,
    )
    complete = compose_cdm_coalescence_time(
        1000.0,
        outer,
        inner,
        gw,
        gravitational_wave_start_separation_pc=0.01,
    )
    assert complete.status == "complete"
    assert complete.total_delay_myr is not None
    assert complete.total_delay_myr > 24.0

    mismatched = compose_cdm_coalescence_time(
        1000.0,
        outer,
        inner,
        gw,
        gravitational_wave_start_separation_pc=0.02,
    )
    assert mismatched.status == "invalid"
    assert mismatched.true_merge_time_myr is None
    assert mismatched.segments[-1].reason is not None
    assert "do not share" in mismatched.segments[-1].reason

    disconnected_inner = read_cdm_delay_stage_summary(
        _complete_stage(
            tmp_path,
            stage="hard_binary_to_gw_regime",
            delay_myr=4.0,
            start_pc=2.0,
            end_pc=0.01,
            phase_path=phase,
        ),
        expected_stage="hard_binary_to_gw_regime",
    )
    disconnected = compose_cdm_coalescence_time(
        1000.0,
        outer,
        disconnected_inner,
        gw,
        gravitational_wave_start_separation_pc=0.01,
    )
    assert disconnected.status == "invalid"
    assert disconnected.segments[1].status == "invalid"


def test_cdm_cli_keeps_censored_environmental_stage_distinct_from_missing_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    phase = _accepted_phase_ensemble(tmp_path)
    outer = _censored_stage(tmp_path, stage="capture_to_hard_binary")
    inner = _complete_stage(
        tmp_path,
        stage="hard_binary_to_gw_regime",
        delay_myr=4.0,
        start_pc=1.0,
        end_pc=0.01,
        phase_path=phase,
    )
    assert (
        main(
            [
                "--sink-time",
                "1 Gyr",
                "--capture-to-hard-summary",
                str(outer),
                "--hard-to-gw-summary",
                str(inner),
                "--gw-mass1",
                "1e8 Msun",
                "--gw-mass2",
                "3e7 Msun",
                "--gw-start-separation",
                "0.01 pc",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "censored"
    assert result["estimate"]["true_merge_time_myr"] is None
    assert result["estimate"]["censored_segments"] == ["capture_to_hard_binary"]
    assert result["dark_matter_model"] == "cdm"
