from __future__ import annotations

from dataclasses import replace
import json

import pytest

from fdm_smbh_delay.zoom_calibration import (
    KpcDelayCalibrationTable,
    ZoomPhysicsPoint,
    accepted_kpc_delay_row,
    apply_kpc_delay_calibration,
    build_zoom_grid,
    compare_zoom_resolution_pair,
    read_zoom_result,
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
        "capture_event_uid": "capture-1",
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
    coarse = read_zoom_result(coarse_path, coarse_case)
    fine = read_zoom_result(fine_path, fine_case)
    convergence = compare_zoom_resolution_pair(fine, coarse)
    assert convergence.status == "accepted"
    assert convergence.maximum_stage_delay_systematic_fraction == pytest.approx(0.1)
    row = accepted_kpc_delay_row(convergence)
    assert row.multiplicative_delay_correction == pytest.approx(0.8)
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


@pytest.mark.parametrize("baseline", [0.0, -1.0, float("nan"), float("inf"), True])
def test_kpc_delay_consumer_rejects_invalid_baseline(tmp_path, baseline) -> None:
    grid = build_zoom_grid(_specification())
    coarse_case, fine_case = grid.cases[:2]
    coarse_path = tmp_path / "coarse.json"
    fine_path = tmp_path / "fine.json"
    _write_result(coarse_path, coarse_case, delay_scale=1.1)
    _write_result(fine_path, fine_case, delay_scale=1.0)
    row = accepted_kpc_delay_row(
        compare_zoom_resolution_pair(
            read_zoom_result(fine_path, fine_case),
            read_zoom_result(coarse_path, coarse_case),
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
    convergence = compare_zoom_resolution_pair(
        read_zoom_result(fine_path, fine_case),
        read_zoom_result(coarse_path, coarse_case),
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
