from __future__ import annotations

from pathlib import Path

import pytest

from scripts.plan_qe_followup_resources import plan_resources


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,run_id,effective_grid_cells\n"
        "case_a,case_a_n256,256\n"
        "case_a,case_a_n512,512\n"
        "case_a,case_a_n768,768\n"
    )
    cases = tmp_path / "cases.csv"
    cases.write_text(
        "case_id,kepler_period_myr,target_duration_myr\n"
        "case_a,0.002,0.024\n"
    )
    return manifest, cases


def test_followup_plan_uses_finest_pair_and_fixed_cell_size_box_control(
    tmp_path: Path,
) -> None:
    manifest, cases = _inputs(tmp_path)
    plan = plan_resources(manifest, cases)
    row = plan["cases"][0]
    assert plan["status"] == "qe_followup_resource_design_only_no_run_authorized"
    assert row["nominal_kepler_orbits_in_existing_target"] == pytest.approx(12)
    assert row["minimum_orbits_for_full_bin_coverage"] == 64
    assert row["minimum_duration_myr_at_initial_kepler_period"] == pytest.approx(
        0.128
    )
    assert row["minimum_duration_over_existing_target"] == pytest.approx(64 / 12)
    controls = row["same_cell_size_box_controls"]
    assert [item["original_resolution"] for item in controls] == [512, 768]
    assert [item["same_cell_size_control_resolution"] for item in controls] == [
        1024, 1536
    ]
    assert [item["uniform_grid_memory_estimate_gib"] for item in controls] == [
        128.0, 432.0
    ]
    assert all(item["estimate_exceeds_reference_gpu"] for item in controls)


def test_missing_or_invalid_case_cannot_be_planned(tmp_path: Path) -> None:
    manifest, cases = _inputs(tmp_path)
    cases.write_text("case_id,kepler_period_myr,target_duration_myr\n")
    with pytest.raises(ValueError, match="absent"):
        plan_resources(manifest, cases)
    cases.write_text(
        "case_id,kepler_period_myr,target_duration_myr\ncase_a,nan,0.024\n"
    )
    with pytest.raises(ValueError, match="invalid duration"):
        plan_resources(manifest, cases)


def test_invalid_resource_request_is_rejected(tmp_path: Path) -> None:
    manifest, cases = _inputs(tmp_path)
    with pytest.raises(ValueError, match="invalid"):
        plan_resources(manifest, cases, box_factor=1)
