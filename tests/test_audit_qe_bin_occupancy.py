from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from scripts import audit_qe_bin_occupancy as audit

pair_occupancy = audit.pair_occupancy


def _run(separations: list[float], *, cutoff: float | None = None) -> dict:
    orbit = np.zeros(
        len(separations),
        dtype=[
            ("mean_separation_pc", float),
            ("mean_separation_over_cell_size", float),
            ("end_time_myr", float),
        ],
    )
    orbit["mean_separation_pc"] = separations
    orbit["mean_separation_over_cell_size"] = 3.0
    orbit["end_time_myr"] = np.arange(1, len(separations) + 1)
    return {
        "orbit_series": orbit,
        "conservation": {
            "initial_spatially_resolved_duration_myr": (
                float(len(separations)) if cutoff is None else cutoff
            )
        },
    }


def test_occupancy_uses_contiguous_instantaneously_resolved_orbits() -> None:
    fine = _run([0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48])
    coarse = _run([0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48], cutoff=8)
    result = pair_occupancy(
        fine, coarse, separation_bins=1, minimum_orbits_per_bin=8
    )
    assert result["resolved_complete_orbits"] == [9, 8]
    assert result["bins"][0]["complete_orbits"] == [8, 8]
    assert result["orbit_count_eligible_bins"] == 1
    assert result["purpose"] == "resource_design_only_no_calibration_admission"


def test_no_common_resolved_separation_is_not_binned() -> None:
    result = pair_occupancy(
        _run([0.1, 0.11]), _run([0.2, 0.21]), separation_bins=8
    )
    assert result["status"] == "no_common_resolved_separation"
    assert result["bins"] == []


def test_insufficient_resolved_orbits_is_distinct_from_no_overlap() -> None:
    result = pair_occupancy(
        _run([0.1, 0.11], cutoff=1), _run([0.1, 0.11]), separation_bins=8
    )
    assert result["status"] == "insufficient_initially_resolved_orbits"
    assert result["resolved_complete_orbits"] == [1, 2]


def test_eight_bin_budget_is_only_a_necessary_full_coverage_bound() -> None:
    positions = list(np.linspace(0.4, 0.5, 12))
    result = pair_occupancy(_run(positions), _run(positions))
    assert result["minimum_total_orbits_for_full_bin_coverage"] == 64
    assert result["resolved_complete_orbits"] == [12, 12]
    assert result["orbit_count_eligible_bins"] == 0


def test_invalid_bin_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid"):
        pair_occupancy(_run([0.1, 0.2]), _run([0.1, 0.2]), separation_bins=0)


def test_one_bin_probe_cannot_publish_calibration_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = []

    def summarize(_runs, *, separation_bins):
        assert separation_bins == 1
        return {"matched_separation": {"bins": [{"bin": 0}]}}

    def build(source):
        paths.append(source.convergence_summary)
        assert source.profile_id == "boey2025"
        assert source.convergence_summary.is_file()
        return SimpleNamespace(accepted_rows=(object(),), rejected_bins=())

    monkeypatch.setattr(audit, "summarize_convergence", summarize)
    monkeypatch.setattr(audit, "build_source_rows", build)
    result = audit.exploratory_single_bin_gates([{}, {}], profile_id="boey2025")
    assert result["accepted_bin_count"] == 1
    assert result["production_calibration_row_admitted"] is False
    assert not paths[0].exists()


def test_one_bin_probe_with_no_bin_skips_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "summarize_convergence",
        lambda runs, *, separation_bins: {"matched_separation": None},
    )
    monkeypatch.setattr(
        audit,
        "build_source_rows",
        lambda source: pytest.fail("builder must not be called"),
    )
    assert audit.exploratory_single_bin_gates([{}, {}], profile_id="boey2025") == {
        "status": "no_exploratory_bin",
        "accepted_bin_count": 0,
    }
