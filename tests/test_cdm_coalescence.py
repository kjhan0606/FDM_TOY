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
from fdm_smbh_delay.model_specific_physics import (
    assess_model_specific_phase_ensemble,
    compare_model_specific_resolution_pair,
    read_resolved_model_physics_result,
)
from fdm_smbh_delay.zoom_calibration import load_zoom_grid

from cdm_orbit_fixture import make_attested_raw_track, unique_rate_path
from test_model_specific_physics import (
    _case as _model_case,
    _physics_input as _model_physics_input,
    _result_record as _model_result_record,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, record: dict[str, object]) -> Path:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _accepted_phase_ensemble(tmp_path: Path, *, model: str = "cdm") -> Path:
    if model == "cdm":
        members = []
        specification_path = Path("configs/cdm_noncompacting_zoom_grid.yaml").resolve()
        grid = load_zoom_grid(specification_path)
        manifest = grid.manifest_sha256
        phase_inputs: dict[int, tuple] = {}
        for replicate in (0, 1):
            candidates = [case for case in grid.cases if case.replicate == replicate]
            assert len(candidates) == 2
            fine_case, coarse_case = sorted(
                candidates,
                key=lambda case: case.numerics.finest_cell_size_pc,
            )
            fine_root = tmp_path / f"phase-inputs-r{replicate}-fine"
            coarse_root = tmp_path / f"phase-inputs-r{replicate}-coarse"
            fine_input, fine_hashes = _model_physics_input(
                fine_root, case=fine_case, manifest_sha256=manifest
            )
            coarse_input, coarse_hashes = _model_physics_input(
                coarse_root, case=coarse_case, manifest_sha256=manifest
            )
            phase_inputs[replicate] = (
                fine_input,
                coarse_input,
                fine_case,
                coarse_case,
                fine_hashes,
                coarse_hashes,
            )
        outer_raw = make_attested_raw_track(
            tmp_path,
            start_pc=1000.0,
            end_pc=1.0,
            model_physics_input=phase_inputs[0][1],
            case_id=phase_inputs[0][3].case_id,
        )
        inner_raw = make_attested_raw_track(
            tmp_path,
            start_pc=1.0,
            end_pc=0.01,
            model_physics_input=phase_inputs[1][0],
            case_id=phase_inputs[1][2].case_id,
        )
        overlap_raw = make_attested_raw_track(
            tmp_path,
            start_pc=2.0,
            end_pc=0.01,
            model_physics_input=phase_inputs[1][1],
            case_id=phase_inputs[1][3].case_id,
        )
        outer_identity = Path(json.loads(outer_raw.read_text(encoding="utf-8"))["runtime_identity"]["path"])
        inner_identity = Path(json.loads(inner_raw.read_text(encoding="utf-8"))["runtime_identity"]["path"])
        overlap_identity = Path(json.loads(overlap_raw.read_text(encoding="utf-8"))["runtime_identity"]["path"])
        for replicate in (0, 1):
            fine_input, coarse_input, fine_case, coarse_case, fine_hashes, coarse_hashes = phase_inputs[replicate]
            fine_root = tmp_path / f"phase-inputs-r{replicate}-fine"
            coarse_root = tmp_path / f"phase-inputs-r{replicate}-coarse"
            fine_path = fine_root / "fine_result.json"
            coarse_path = coarse_root / "coarse_result.json"
            _write_json(
                fine_path,
                _model_result_record(
                    case=fine_case,
                    manifest_sha256=manifest,
                    physics_input=fine_input,
                    artifacts=fine_hashes["cdm"],
                    runtime_identity_path=(
                        inner_identity if replicate == 1 else None
                    ),
                ),
            )
            _write_json(
                coarse_path,
                _model_result_record(
                    case=coarse_case,
                    manifest_sha256=manifest,
                    physics_input=coarse_input,
                    artifacts=coarse_hashes["cdm"],
                    runtime_identity_path=(
                        outer_identity if replicate == 0 else overlap_identity
                    ),
                ),
            )
            members.append(
                compare_model_specific_resolution_pair(
                    read_resolved_model_physics_result(
                        fine_path,
                        case=fine_case,
                        zoom_manifest_sha256=manifest,
                    ),
                    read_resolved_model_physics_result(
                        coarse_path,
                        case=coarse_case,
                        zoom_manifest_sha256=manifest,
                    ),
                )
            )
        phase = assess_model_specific_phase_ensemble(tuple(members))
        return _write_json(
            tmp_path / "phase_ensemble_cdm.json",
            phase.as_dict(zoom_specification_path=specification_path),
        )
    return _write_json(
        tmp_path / f"phase_ensemble_{model}.json",
        {
            "schema_version": 2,
            "status": "accepted_model_specific_phase_ensemble",
            "interpretation": "test non-CDM phase",
            "dark_matter_model": model,
            "physics_id": "zoomphys-88aec2c614c5bbb0",
            "zoom_manifest_sha256": "a" * 64,
            "zoom_specification_path": str(
                Path("configs/cdm_noncompacting_zoom_grid.yaml").resolve()
            ),
            "zoom_specification_sha256": _sha256(
                Path("configs/cdm_noncompacting_zoom_grid.yaml").resolve()
            ),
            "capture_event": {
                "event_uid": "capture-7-9",
                "event_sha256": "a" * 64,
            },
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
    capture_event_uid: str = "capture-7-9",
    capture_event_sha256: str = "a" * 64,
    raw_path: Path | None = None,
) -> Path:
    if raw_path is None:
        for candidate in sorted(tmp_path.glob("**/raw_relative_orbit_track.json")):
            try:
                candidate_record = json.loads(candidate.read_text(encoding="utf-8"))
                candidate_track = derive_cdm_secular_rate_track(
                    candidate,
                    stage=stage,
                    samples_per_block=5,
                )
                candidate_points = candidate_track.get("rate_points", [])
                if (
                    candidate_track.get("status") == "complete"
                    and isinstance(candidate_points, list)
                    and candidate_points
                    and math.isclose(
                        float(candidate_points[0]["separation_pc"]),
                        start_pc,
                        rel_tol=1.0e-12,
                    )
                    and math.isclose(
                        float(candidate_points[-1]["separation_pc"]),
                        end_pc,
                        rel_tol=1.0e-12,
                    )
                    and candidate_record.get("capture_event_uid") == capture_event_uid
                ):
                    raw_path = candidate
                    break
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
    raw = raw_path or make_attested_raw_track(
        tmp_path,
        start_pc=start_pc,
        end_pc=end_pc,
        capture_event_uid=capture_event_uid,
    )
    return _write_json(
        unique_rate_path(tmp_path, stage),
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
    capture_event_uid: str = "capture-7-9",
    capture_event_sha256: str = "a" * 64,
) -> Path:
    rate_track = _rate_track(
        tmp_path,
        stage=stage,
        delay_myr=delay_myr,
        start_pc=start_pc,
        end_pc=end_pc,
        capture_event_uid=capture_event_uid,
        capture_event_sha256=capture_event_sha256,
    )
    rate_record = json.loads(rate_track.read_text(encoding="utf-8"))
    resolved_track = read_cdm_resolved_rate_track(
        rate_track,
        expected_stage=stage,
        expected_physics_id="zoomphys-88aec2c614c5bbb0",
    )
    return _write_json(
        tmp_path / f"{stage}.json",
        {
            "schema_version": 3,
            "status": "complete",
            "dark_matter_model": "cdm",
            "stage": stage,
            "delay_myr": resolved_track.integrated_delay_myr,
            "delay_uncertainty_myr": resolved_track.integrated_delay_uncertainty_myr,
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
                "physics_id": "zoomphys-88aec2c614c5bbb0",
                "capture_event_uid": rate_record["capture_event_uid"],
                "capture_event_sha256": rate_record["capture_event_sha256"],
                "rate_track_path": rate_track.name,
                "rate_track_sha256": _sha256(rate_track),
            },
        },
    )


def _censored_stage(tmp_path: Path, *, stage: str) -> Path:
    return _write_json(
        tmp_path / f"{stage}.json",
        {
            "schema_version": 3,
            "status": "censored",
            "dark_matter_model": "cdm",
            "stage": stage,
            "delay_myr": None,
            "delay_uncertainty_myr": None,
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
    assert parsed.segment.delay_myr == pytest.approx(0.001, rel=1.0e-6)
    assert parsed.physics_id == "zoomphys-88aec2c614c5bbb0"
    assert parsed.phase_ensemble_sha256 == _sha256(phase)


def test_completed_cdm_stage_reassesses_phase_members_and_capture_identity(
    tmp_path: Path,
) -> None:
    phase = _accepted_phase_ensemble(tmp_path)
    phase_record = json.loads(phase.read_text(encoding="utf-8"))
    phase_record["members"] = []
    tampered_phase = _write_json(tmp_path / "tampered_phase.json", phase_record)
    stage = _complete_stage(
        tmp_path,
        stage="capture_to_hard_binary",
        delay_myr=20.0,
        start_pc=1000.0,
        end_pc=1.0,
        phase_path=tampered_phase,
    )
    with pytest.raises(ValueError, match="at least two members"):
        read_cdm_delay_stage_summary(stage, expected_stage="capture_to_hard_binary")


def test_cdm_stage_rejects_a_rate_track_from_an_unlisted_runtime_identity(
    tmp_path: Path,
) -> None:
    phase = _accepted_phase_ensemble(tmp_path)
    foreign_raw = make_attested_raw_track(
        tmp_path,
        start_pc=1000.0,
        end_pc=1.0,
    )
    foreign_rate = _write_json(
        tmp_path / "foreign_rate_track.json",
        derive_cdm_secular_rate_track(
            foreign_raw,
            stage="capture_to_hard_binary",
            samples_per_block=5,
        ),
    )
    with pytest.raises(ValueError, match="not one of the accepted phase members"):
        build_cdm_delay_stage_record(foreign_rate, phase)


def test_cdm_phase_anchor_requires_rate_ledger_output_in_runtime_identity(
    tmp_path: Path,
) -> None:
    phase = _accepted_phase_ensemble(tmp_path)
    foreign_raw = make_attested_raw_track(tmp_path, start_pc=1000.0, end_pc=1.0)
    foreign_raw_record = json.loads(foreign_raw.read_text(encoding="utf-8"))
    foreign_identity = foreign_raw_record["runtime_identity"]
    foreign_rate = _write_json(
        tmp_path / "foreign_anchor_rate_track.json",
        derive_cdm_secular_rate_track(
            foreign_raw,
            stage="capture_to_hard_binary",
            samples_per_block=5,
        ),
    )

    phase_record = json.loads(phase.read_text(encoding="utf-8"))
    anchor = phase_record["members"][0]["comparison"]
    result_path = Path(anchor["source"]["path"])
    result_record = json.loads(result_path.read_text(encoding="utf-8"))
    result_record["runtime_identity"] = {
        "path": foreign_identity["path"],
        "sha256": foreign_identity["sha256"],
    }
    _write_json(result_path, result_record)
    anchor["source"]["sha256"] = _sha256(result_path)
    anchor["runtime_identity"] = {
        "path": str(Path(foreign_identity["path"]).resolve()),
        "sha256": foreign_identity["sha256"],
    }
    tampered_phase = _write_json(tmp_path / "foreign_anchor_phase.json", phase_record)

    with pytest.raises(ValueError, match="outside its runtime output set"):
        build_cdm_delay_stage_record(foreign_rate, tampered_phase)


def test_cdm_phase_rejects_a_case_not_in_the_zoom_specification(
    tmp_path: Path,
) -> None:
    phase = _accepted_phase_ensemble(tmp_path)
    phase_record = json.loads(phase.read_text(encoding="utf-8"))
    forged_case = _model_case("cdm", finest_cell_size_pc=0.5).as_dict()
    descriptor = phase_record["members"][0]["reference"]
    descriptor["case"] = forged_case
    descriptor["case_id"] = forged_case["case_id"]
    tampered_phase = _write_json(tmp_path / "foreign_case_phase.json", phase_record)
    rate_track = _rate_track(
        tmp_path,
        stage="capture_to_hard_binary",
        delay_myr=20.0,
        start_pc=1000.0,
        end_pc=1.0,
    )
    with pytest.raises(ValueError, match="not an exact member"):
        build_cdm_delay_stage_record(rate_track, tampered_phase)


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
    assert record["schema_version"] == 3
    parsed_track = read_cdm_resolved_rate_track(
        track,
        expected_stage="capture_to_hard_binary",
        expected_physics_id="zoomphys-88aec2c614c5bbb0",
    )
    assert record["delay_myr"] == pytest.approx(parsed_track.integrated_delay_myr)
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
            expected_physics_id="zoomphys-88aec2c614c5bbb0",
        )


def test_cdm_rate_track_records_orbit_coverage_and_delay_uncertainty(
    tmp_path: Path,
) -> None:
    track_path = _rate_track(
        tmp_path,
        stage="capture_to_hard_binary",
        delay_myr=20.0,
        start_pc=1000.0,
        end_pc=1.0,
    )
    record = json.loads(track_path.read_text(encoding="utf-8"))
    assert record["schema_version"] == 3
    assert record["status"] == "complete"
    assert record["integrated_delay_uncertainty_myr"] >= 0.0
    assert all(
        point["orbit_count"] >= 1.0 and 0.0 <= point["r_squared"] <= 1.0
        for point in record["rate_points"]
    )
    parsed = read_cdm_resolved_rate_track(
        track_path,
        expected_stage="capture_to_hard_binary",
        expected_physics_id="zoomphys-88aec2c614c5bbb0",
    )
    assert parsed.capture_event_uid == "capture-7-9"
    assert parsed.integrated_delay_uncertainty_myr == pytest.approx(
        record["integrated_delay_uncertainty_myr"]
    )


def test_cdm_rate_derivation_rejects_hand_edited_attested_samples(
    tmp_path: Path,
) -> None:
    raw_path = make_attested_raw_track(
        tmp_path,
        start_pc=1000.0,
        end_pc=1.0,
    )
    record = json.loads(raw_path.read_text(encoding="utf-8"))
    record["samples"][0]["separation_pc"] *= 1.01
    tampered = _write_json(tmp_path / "tampered_raw_track.json", record)
    with pytest.raises(ValueError, match="differs from its current lagRamses extraction"):
        derive_cdm_secular_rate_track(
            tampered,
            stage="capture_to_hard_binary",
            samples_per_block=5,
        )


def test_cdm_rate_derivation_requires_at_least_one_orbit_per_block(
    tmp_path: Path,
) -> None:
    raw_path = make_attested_raw_track(tmp_path)
    with pytest.raises(ValueError, match="at least one complete orbit"):
        derive_cdm_secular_rate_track(
            raw_path,
            stage="capture_to_hard_binary",
            samples_per_block=5,
            minimum_orbits_per_block=0.5,
        )


def test_nondecaying_cdm_rate_track_is_explicitly_censored(tmp_path: Path) -> None:
    raw_path = make_attested_raw_track(
        tmp_path,
        start_pc=1.0,
        end_pc=1.0,
        stalled=True,
    )
    record = derive_cdm_secular_rate_track(
        raw_path,
        stage="capture_to_hard_binary",
        samples_per_block=5,
    )
    assert record["status"] == "censored"
    assert "nonnegative_secular_slope" in record["reason"]
    phase = _accepted_phase_ensemble(tmp_path)
    rate_path = _write_json(tmp_path / "stalled_rate.json", record)
    stage = _write_json(
        tmp_path / "stalled_stage.json",
        build_cdm_delay_stage_record(rate_path, phase),
    )
    assert read_cdm_delay_stage_summary(
        stage, expected_stage="capture_to_hard_binary"
    ).segment.status == "censored"


def test_cdm_rate_track_censors_insufficient_orbit_coverage(tmp_path: Path) -> None:
    raw_path = make_attested_raw_track(
        tmp_path,
        start_pc=100.0,
        end_pc=10.0,
        density_msun_pc3=1.0,
    )
    record = derive_cdm_secular_rate_track(
        raw_path,
        stage="capture_to_hard_binary",
        samples_per_block=5,
    )
    assert record["status"] == "censored"
    assert record["reason"] == "insufficient_complete_orbit_blocks_for_secular_rate"


def test_cdm_rate_track_censors_legacy_samples_without_physical_period(
    tmp_path: Path,
) -> None:
    raw_path = make_attested_raw_track(
        tmp_path,
        start_pc=10.0,
        end_pc=1.0,
        include_unit_d=False,
    )
    record = derive_cdm_secular_rate_track(
        raw_path,
        stage="capture_to_hard_binary",
        samples_per_block=5,
    )
    assert record["status"] == "censored"
    assert record["reason"] == "missing_orbital_period_for_secular_rate"


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
    assert complete.total_delay_myr > gw.delay_myr
    assert complete.total_delay_myr == pytest.approx(
        outer.segment.delay_myr + inner.segment.delay_myr + gw.delay_myr
    )

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

    different_capture_dir = tmp_path / "different-capture"
    different_capture_dir.mkdir()
    different_phase = different_capture_dir / phase.name
    different_phase.write_bytes(phase.read_bytes())
    different_capture_stage = _complete_stage(
        different_capture_dir,
        stage="hard_binary_to_gw_regime",
        delay_myr=4.0,
        start_pc=1.0,
        end_pc=0.01,
        phase_path=different_phase,
        capture_event_uid="capture-other",
        capture_event_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="capture event"):
        read_cdm_delay_stage_summary(
            different_capture_stage,
            expected_stage="hard_binary_to_gw_regime",
        )


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
