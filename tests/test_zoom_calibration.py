from __future__ import annotations

from dataclasses import replace
import json
import os

import numpy as np
import pytest

from capture_protocol_fixture import write_committed_capture
from test_capture_ledger import _binary_rows
from fdm_smbh_delay.capture_ledger import CaptureLedgerError, read_capture_ledger
from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.zoom_calibration import (
    KpcDelayCalibrationTable,
    ZoomPhysicsPoint,
    accepted_kpc_delay_row,
    apply_kpc_delay_calibration,
    bind_zoom_result_to_capture_ledger,
    build_zoom_grid,
    compare_zoom_resolution_pair,
    read_zoom_result,
)


def _committed_zoom_capture(tmp_path):
    rows = _binary_rows()
    rows[0]["boxlen"] = 20000.0
    rows[0]["merge_radius_code"] = 6000.0
    speed = np.sqrt(G_INTERNAL * 2.0e8 * 0.5 / 5000.0)
    rows[1]["position_code"] = [2500.0, 0.0, 0.0]
    rows[2]["position_code"] = [-2500.0, 0.0, 0.0]
    rows[1]["velocity_code"] = [0.0, 0.5 * speed, 0.0]
    rows[2]["velocity_code"] = [0.0, -0.5 * speed, 0.0]
    rows[3]["legacy_pair_bound"] = False
    path = tmp_path / "capture.jsonl"
    uid = write_committed_capture(path, rows)
    event = read_capture_ledger(path).events[0]
    assert event.event_uid == uid
    assert event.post_compaction_verified and event.native_conservation_verified
    assert event.binary_orbital_state is not None
    assert event.binary_orbital_state.eccentricity == pytest.approx(0.5)
    return path


def _bound_result(path, case, ledger_path):
    return bind_zoom_result_to_capture_ledger(
        read_zoom_result(path, case), ledger_path,
    )


def _specification() -> dict:
    return {
        "schema_version": 1,
        "replicates": 1,
        "baseline": {
            "host_stellar_mass_msun": 1.0e11,
            "host_scale_radius_pc": 1000.0,
            "host_inner_slope": 1.0,
            "binary_total_mass_msun": 2.0e8,
            "mass_ratio_q": 1.0,
            "gas_fraction": 0.2,
            "gas_rotation_fraction": 0.5,
            "initial_orbit_eccentricity": 0.5,
            "initial_separation_pc": 5000.0,
            "nuclear_envelope_to_secondary_bh_mass": 10.0,
            "dark_matter_model": "fdm",
            "fdm_particle_mass_ev": 1.0e-21,
            "fdm_core_radius_pc": 200.0,
            "fdm_soliton_mass_msun": 1.0e10,
        },
        "variations": [{"mass_ratio_q": 0.3}],
        "numerics": [
            {
                "levelmax": 17,
                "finest_cell_size_pc": 2.0,
                "collisionless_particle_mass_msun": 1.0e4,
                "minimum_softening_pc": 2.0,
            },
            {
                "levelmax": 18,
                "finest_cell_size_pc": 1.0,
                "collisionless_particle_mass_msun": 1.25e3,
                "minimum_softening_pc": 1.0,
            },
        ],
    }


def _write_result(path, case, *, delay_scale: float = 1.0, cells: float = 8.0):
    stages = {
        "numerical_capture": {
            "status": "complete",
            "elapsed_since_capture_myr": 0.0,
            "separation_pc": 5000.0,
        },
        "common_nucleus": {
            "status": "complete",
            "elapsed_since_capture_myr": 10.0 * delay_scale,
            "separation_pc": 100.0,
        },
        "bound_binary": {
            "status": "complete",
            "elapsed_since_capture_myr": 15.0 * delay_scale,
            "separation_pc": 10.0,
        },
        "hard_binary": {
            "status": "complete",
            "elapsed_since_capture_myr": 20.0 * delay_scale,
            "separation_pc": 1.0,
        },
    }
    record = {
        "schema_version": 1,
        "case_id": case.case_id,
        "case": case.as_dict(),
        "capture_event_uid": "10-1-7-9-2",
        "stages": stages,
        "analytic_kpc_to_hard_delay_myr": 25.0,
        "integration_time_myr": 30.0,
        "diagnostics": {
            "maximum_relative_energy_error": 1.0e-5,
            "maximum_relative_angular_momentum_error": 2.0e-5,
            "minimum_transition_radius_cells": cells,
        },
    }
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")


def test_grid_pairs_every_physics_point_at_two_resolutions() -> None:
    grid = build_zoom_grid(_specification())
    assert len(grid.cases) == 4
    assert len(grid.manifest_sha256) == 64
    for physics_id in {case.physics.physics_id for case in grid.cases}:
        cases = [case for case in grid.cases if case.physics.physics_id == physics_id]
        assert [case.numerics.levelmax for case in cases] == [17, 18]


def test_grid_rejects_duplicate_physical_points() -> None:
    spec = _specification()
    spec["variations"] = [{"mass_ratio_q": 1.0}]
    with pytest.raises(ValueError, match="duplicated"):
        build_zoom_grid(spec)


def test_pure_fdm_point_can_explicitly_omit_stellar_baryons() -> None:
    mapping = dict(_specification()["baseline"])
    mapping.update(host_stellar_mass_msun=0.0, gas_fraction=0.0, gas_rotation_fraction=0.0)
    point = ZoomPhysicsPoint(**mapping)
    assert point.host_stellar_mass_msun == 0.0
    assert point.gas_fraction == 0.0


def test_resolution_pair_builds_exact_point_delay_row(tmp_path) -> None:
    grid = build_zoom_grid(_specification())
    coarse_case, fine_case = grid.cases[:2]
    coarse_path = tmp_path / "coarse.json"
    fine_path = tmp_path / "fine.json"
    _write_result(coarse_path, coarse_case, delay_scale=1.1)
    _write_result(fine_path, fine_case, delay_scale=1.0)
    ledger_path = _committed_zoom_capture(tmp_path)
    coarse = _bound_result(coarse_path, coarse_case, ledger_path)
    fine = _bound_result(fine_path, fine_case, ledger_path)
    convergence = compare_zoom_resolution_pair(fine, coarse)
    assert convergence.status == "accepted"
    assert convergence.maximum_stage_delay_systematic_fraction == pytest.approx(0.1)
    row = accepted_kpc_delay_row(convergence)
    assert row.multiplicative_delay_correction == pytest.approx(0.8)
    assert row.reference_result_sha256 == fine.source_sha256
    assert row.comparison_result_sha256 == coarse.source_sha256
    assert row.source_sha256 not in {fine.source_sha256, coarse.source_sha256}
    assert row.capture_event_sha256 == fine.capture_event_sha256
    table = KpcDelayCalibrationTable((row,))
    assert table.lookup(fine_case.physics) == row
    unmeasured = replace(fine_case.physics, mass_ratio_q=0.7)
    with pytest.raises(ValueError, match="extrapolation is prohibited"):
        table.lookup(unmeasured)

    calibrated = apply_kpc_delay_calibration(
        table,
        physics=fine_case.physics,
        analytic_baseline_delay_myr=50.0,
    )
    assert calibrated.status == "complete"
    assert calibrated.delay_myr == pytest.approx(40.0)
    assert calibrated.source_case_id == row.source_case_id
    assert calibrated.source_sha256 == row.source_sha256
    assert len(calibrated.source_sha256) == 64

    outside_support = table.calibrated_delay_segment(unmeasured, 50.0)
    assert outside_support.status == "censored"
    assert outside_support.delay_myr is None
    assert "extrapolation is prohibited" in (outside_support.reason or "")

    malformed = replace(row, source_sha256="not-a-sha256")
    with pytest.raises(ValueError, match="provenance is incomplete"):
        KpcDelayCalibrationTable((malformed,))
    with pytest.raises(ValueError, match="provenance is incomplete"):
        KpcDelayCalibrationTable((replace(row, comparison_result_sha256="0" * 64),))
    with pytest.raises(ValueError, match="provenance is incomplete"):
        KpcDelayCalibrationTable((replace(row, capture_event_sha256="0" * 64),))

    revised_coarse = json.loads(coarse_path.read_text())
    revised_coarse["stages"]["hard_binary"]["elapsed_since_capture_myr"] = 21.0
    coarse_path.write_text(json.dumps(revised_coarse), encoding="utf-8")
    changed_convergence = compare_zoom_resolution_pair(
        fine, _bound_result(coarse_path, coarse_case, ledger_path)
    )
    assert changed_convergence.status == "accepted"
    assert accepted_kpc_delay_row(changed_convergence).source_sha256 != row.source_sha256


def test_resolution_pair_rejects_different_capture_events(tmp_path) -> None:
    coarse_case, fine_case = build_zoom_grid(_specification()).cases[:2]
    coarse_path = tmp_path / "coarse.json"
    fine_path = tmp_path / "fine.json"
    _write_result(coarse_path, coarse_case)
    _write_result(fine_path, fine_case)
    coarse_record = json.loads(coarse_path.read_text())
    coarse_record["capture_event_uid"] = "another-capture"
    coarse_path.write_text(json.dumps(coarse_record))
    with pytest.raises(ValueError, match="different capture events"):
        compare_zoom_resolution_pair(
            read_zoom_result(fine_path, fine_case),
            read_zoom_result(coarse_path, coarse_case),
        )


def test_zoom_pair_requires_committed_capture_binding(tmp_path) -> None:
    coarse_case, fine_case = build_zoom_grid(_specification()).cases[:2]
    coarse_path = tmp_path / "coarse.json"
    fine_path = tmp_path / "fine.json"
    _write_result(coarse_path, coarse_case)
    _write_result(fine_path, fine_case)
    with pytest.raises(ValueError, match="verified ledger capture"):
        compare_zoom_resolution_pair(
            read_zoom_result(fine_path, fine_case),
            read_zoom_result(coarse_path, coarse_case),
        )
    ledger_path = _committed_zoom_capture(tmp_path)
    altered = json.loads(fine_path.read_text())
    altered["capture_event_uid"] = "10-1-7-10-2"
    fine_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the active ledger lineage"):
        _bound_result(fine_path, fine_case, ledger_path)


def test_zoom_capture_binding_rejects_changed_physical_state(tmp_path) -> None:
    case = build_zoom_grid(_specification()).cases[0]
    result_path = tmp_path / "result.json"
    _write_result(result_path, case)
    ledger_path = _committed_zoom_capture(tmp_path)
    different_mass = replace(case.physics, binary_total_mass_msun=2.1e8)
    altered_case = replace(case, physics=different_mass)
    altered = json.loads(result_path.read_text())
    altered["case_id"] = altered_case.case_id
    altered["case"] = altered_case.as_dict()
    result_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with the committed ledger"):
        _bound_result(result_path, altered_case, ledger_path)


def test_zoom_capture_binding_rejects_changed_initial_eccentricity(tmp_path) -> None:
    case = build_zoom_grid(_specification()).cases[0]
    result_path = tmp_path / "result.json"
    _write_result(result_path, case)
    ledger_path = _committed_zoom_capture(tmp_path)
    altered_case = replace(
        case, physics=replace(case.physics, initial_orbit_eccentricity=0.3),
    )
    altered = json.loads(result_path.read_text())
    altered["case_id"] = altered_case.case_id
    altered["case"] = altered_case.as_dict()
    result_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with the committed ledger"):
        _bound_result(result_path, altered_case, ledger_path)


def test_zoom_capture_binding_rejects_legacy_bare_event(tmp_path) -> None:
    case = build_zoom_grid(_specification()).cases[0]
    result_path = tmp_path / "result.json"
    _write_result(result_path, case)
    bare = tmp_path / "bare.jsonl"
    bare.write_text(
        "".join(json.dumps(row) + "\n" for row in _binary_rows()),
        encoding="utf-8",
    )
    with pytest.raises(CaptureLedgerError, match="legacy bare events"):
        _bound_result(result_path, case, bare)


def test_zoom_binding_rejects_later_superseding_restart(tmp_path) -> None:
    coarse_case, fine_case = build_zoom_grid(_specification()).cases[:2]
    coarse_path = tmp_path / "coarse.json"
    fine_path = tmp_path / "fine.json"
    _write_result(coarse_path, coarse_case)
    _write_result(fine_path, fine_case)
    ledger_path = _committed_zoom_capture(tmp_path)
    fine = _bound_result(fine_path, fine_case, ledger_path)
    coarse = _bound_result(coarse_path, coarse_case, ledger_path)
    convergence = compare_zoom_resolution_pair(fine, coarse)
    assert convergence.status == "accepted"

    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    checkpoint = {
        "schema_version": 1, "record_type": "checkpoint",
        "output_number": 1, "nstep_coarse": 10,
    }
    restart = {
        "schema_version": 1, "record_type": "attempt_begin",
        "restart_output": 1, "resume_step": 10,
    }
    ledger_path.write_text(
        "".join(json.dumps(row) + "\n" for row in
                [rows[0], checkpoint, *rows[1:], restart]),
        encoding="utf-8",
    )
    assert read_capture_ledger(ledger_path).events == ()
    with pytest.raises(ValueError, match="ledger changed after zoom binding"):
        compare_zoom_resolution_pair(fine, coarse)
    with pytest.raises(ValueError, match="ledger changed after zoom binding"):
        accepted_kpc_delay_row(convergence)
    with pytest.raises(ValueError, match="absent from the active ledger lineage"):
        _bound_result(fine_path, fine_case, ledger_path)


def test_zoom_binding_rejects_same_size_ledger_rewrite_with_restored_mtime(
    tmp_path,
) -> None:
    coarse_case, fine_case = build_zoom_grid(_specification()).cases[:2]
    coarse_path = tmp_path / "coarse.json"
    fine_path = tmp_path / "fine.json"
    _write_result(coarse_path, coarse_case)
    _write_result(fine_path, fine_case)
    ledger_path = _committed_zoom_capture(tmp_path)
    fine = _bound_result(fine_path, fine_case, ledger_path)
    coarse = _bound_result(coarse_path, coarse_case, ledger_path)
    convergence = compare_zoom_resolution_pair(fine, coarse)
    assert convergence.status == "accepted"

    original_stat = ledger_path.stat()
    original_text = ledger_path.read_text(encoding="utf-8")
    assert '"spin_magnitude": 0.5' in original_text
    modified_text = original_text.replace(
        '"spin_magnitude": 0.5', '"spin_magnitude": 0.6', 1,
    )
    assert len(modified_text) == len(original_text)
    ledger_path.write_text(modified_text, encoding="utf-8")
    os.utime(
        ledger_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert ledger_path.stat().st_size == original_stat.st_size
    assert ledger_path.stat().st_mtime_ns == original_stat.st_mtime_ns
    with pytest.raises(ValueError, match="ledger changed after zoom binding"):
        compare_zoom_resolution_pair(fine, coarse)
    with pytest.raises(ValueError, match="ledger changed after zoom binding"):
        accepted_kpc_delay_row(convergence)


def test_zoom_binding_rejects_result_changed_after_read(tmp_path) -> None:
    coarse_case, fine_case = build_zoom_grid(_specification()).cases[:2]
    coarse_path = tmp_path / "coarse.json"
    fine_path = tmp_path / "fine.json"
    _write_result(coarse_path, coarse_case)
    _write_result(fine_path, fine_case)
    ledger_path = _committed_zoom_capture(tmp_path)
    fine = _bound_result(fine_path, fine_case, ledger_path)
    coarse = _bound_result(coarse_path, coarse_case, ledger_path)
    modified = json.loads(coarse_path.read_text())
    modified["stages"]["hard_binary"]["elapsed_since_capture_myr"] = 21.0
    coarse_path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="zoom result changed after capture binding"):
        compare_zoom_resolution_pair(fine, coarse)


@pytest.mark.parametrize("coarse_baseline", [None, 30.0])
def test_resolution_pair_rejects_missing_or_changed_analytic_baseline(
    tmp_path, coarse_baseline
) -> None:
    coarse_case, fine_case = build_zoom_grid(_specification()).cases[:2]
    coarse_path = tmp_path / "coarse.json"
    fine_path = tmp_path / "fine.json"
    _write_result(coarse_path, coarse_case)
    _write_result(fine_path, fine_case)
    record = json.loads(coarse_path.read_text())
    record["analytic_kpc_to_hard_delay_myr"] = coarse_baseline
    coarse_path.write_text(json.dumps(record), encoding="utf-8")
    ledger_path = _committed_zoom_capture(tmp_path)
    comparison = compare_zoom_resolution_pair(
        _bound_result(fine_path, fine_case, ledger_path),
        _bound_result(coarse_path, coarse_case, ledger_path),
    )
    assert comparison.status == "rejected"
    assert any("baseline" in reason for reason in comparison.reasons)
    with pytest.raises(ValueError, match="only accepted zoom convergence"):
        accepted_kpc_delay_row(comparison)


@pytest.mark.parametrize("uid", [7, "", "   "])
def test_zoom_result_rejects_nontext_or_blank_capture_uid(tmp_path, uid) -> None:
    case = build_zoom_grid(_specification()).cases[0]
    path = tmp_path / "result.json"
    _write_result(path, case)
    record = json.loads(path.read_text())
    record["capture_event_uid"] = uid
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="provenance does not match"):
        read_zoom_result(path, case)


@pytest.mark.parametrize("baseline", [0.0, -1.0, float("nan"), float("inf"), True])
def test_kpc_delay_consumer_rejects_invalid_baseline(tmp_path, baseline) -> None:
    grid = build_zoom_grid(_specification())
    coarse_case, fine_case = grid.cases[:2]
    coarse_path = tmp_path / "coarse.json"
    fine_path = tmp_path / "fine.json"
    _write_result(coarse_path, coarse_case, delay_scale=1.1)
    _write_result(fine_path, fine_case, delay_scale=1.0)
    ledger_path = _committed_zoom_capture(tmp_path)
    row = accepted_kpc_delay_row(
        compare_zoom_resolution_pair(
            _bound_result(fine_path, fine_case, ledger_path),
            _bound_result(coarse_path, coarse_case, ledger_path),
        )
    )
    table = KpcDelayCalibrationTable((row,))
    with pytest.raises(ValueError, match="finite and positive"):
        table.calibrated_delay_segment(fine_case.physics, baseline)


def test_underresolved_transition_rejects_zoom_pair(tmp_path) -> None:
    grid = build_zoom_grid(_specification())
    coarse_case, fine_case = grid.cases[:2]
    coarse_path = tmp_path / "coarse.json"
    fine_path = tmp_path / "fine.json"
    _write_result(coarse_path, coarse_case, cells=3.0)
    _write_result(fine_path, fine_case, cells=8.0)
    ledger_path = _committed_zoom_capture(tmp_path)
    convergence = compare_zoom_resolution_pair(
        _bound_result(fine_path, fine_case, ledger_path),
        _bound_result(coarse_path, coarse_case, ledger_path),
    )
    assert convergence.status == "rejected"
    assert any("underresolved" in reason for reason in convergence.reasons)


def test_stage_time_reversal_is_rejected(tmp_path) -> None:
    case = build_zoom_grid(_specification()).cases[0]
    path = tmp_path / "result.json"
    _write_result(path, case)
    record = json.loads(path.read_text())
    record["stages"]["bound_binary"]["elapsed_since_capture_myr"] = 5.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="not physically ordered"):
        read_zoom_result(path, case)


def test_zoom_stage_cannot_exceed_integrated_time(tmp_path) -> None:
    case = build_zoom_grid(_specification()).cases[0]
    path = tmp_path / "result.json"
    _write_result(path, case)
    record = json.loads(path.read_text())
    record["integration_time_myr"] = 19.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds the recorded integration time"):
        read_zoom_result(path, case)


def test_zoom_capture_separation_must_match_manifest_initial_state(tmp_path) -> None:
    case = build_zoom_grid(_specification()).cases[0]
    path = tmp_path / "result.json"
    _write_result(path, case)
    record = json.loads(path.read_text())
    record["stages"]["numerical_capture"]["separation_pc"] = 4900.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the manifest initial state"):
        read_zoom_result(path, case)
