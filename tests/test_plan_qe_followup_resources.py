from __future__ import annotations

from pathlib import Path

import pytest

from scripts import plan_qe_followup_resources as planner
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


@pytest.mark.parametrize(
    ("pilot_status", "expected"),
    [
        ("insufficient_initially_resolved_orbits", "initial_resolution_redesign_required"),
        ("no_common_resolved_separation", "common_separation_support_redesign_required"),
        ("common_resolved_separation_binned", "common_support_only_new_registered_sampling_required"),
    ],
)
def test_followup_plan_distinguishes_pilot_support_from_orbit_budget(
    tmp_path: Path, monkeypatch, pilot_status: str, expected: str,
) -> None:
    manifest, cases = _inputs(tmp_path)
    assessment = tmp_path / "assessment.json"
    assessment.write_text("{}")
    calls = []

    def audited(path, *, separation_bins, minimum_orbits_per_bin):
        calls.append((path, separation_bins, minimum_orbits_per_bin))
        return {
            "assessment": str(path), "assessment_sha256": "a" * 64,
            "cases": [{"case_id": "case_a",
                       "peak_device_memory_bytes_fine_coarse": [16 * 1024**3, 2 * 1024**3],
                       "diagnostic": {
                "status": pilot_status, "resolved_complete_orbits": [12, 11],
                "common_minimum_separation_pc": 0.4,
                "common_maximum_separation_pc": 0.5,
                "orbit_count_eligible_bins": 0,
            }}],
        }

    monkeypatch.setattr(planner, "audit_assessment", audited)
    plan = plan_resources(manifest, cases, assessment=assessment)
    assert calls == [(assessment, 8, 8)]
    assert plan["verified_pilot_assessment"]["sha256"] == "a" * 64
    assert plan["cases"][0]["pilot_support"]["status"] == expected
    assert plan["cases"][0]["minimum_orbits_for_full_bin_coverage"] == 64
    assert plan["cases"][0]["pilot_support"]["fine_measured_peak_allocated_gib"] == 16.0
    assert plan["cases"][0]["pilot_support"][
        "cubic_extrapolated_doubled_box_allocated_gib"
    ] == 128.0
    assert plan["cases"][0]["pilot_support"][
        "cubic_projection_exceeds_reference_gpu"
    ] is True


def test_followup_plan_rejects_assessment_case_mismatch(tmp_path: Path, monkeypatch) -> None:
    manifest, cases = _inputs(tmp_path)
    monkeypatch.setattr(planner, "audit_assessment", lambda *args, **kwargs: {
        "assessment": "pilot.json", "assessment_sha256": "a" * 64,
        "cases": [{"case_id": "different", "diagnostic": {
            "status": "common_resolved_separation_binned",
            "resolved_complete_orbits": [8, 8],
        }}],
    })
    with pytest.raises(ValueError, match="lacks manifest case"):
        plan_resources(manifest, cases, assessment=tmp_path / "pilot.json")
