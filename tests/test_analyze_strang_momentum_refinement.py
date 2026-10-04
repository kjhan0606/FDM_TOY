"""Source-bound nonzero-limit checks for the Strang momentum audit."""

import json

import pytest

from scripts import analyze_strang_momentum_refinement as analysis


def test_momentum_refinement_fails_nonzero_limit_and_changed_source(
    tmp_path, monkeypatch,
) -> None:
    root = tmp_path / "traces"
    root.mkdir()
    runs = []
    for label, factor, stop in analysis._LEVELS:
        run = root / label
        run.mkdir()
        runs.append({
            "run": str(run),
            "time_step_factor": factor,
            "slurm_job_id": "123",
            "source_commit": "commit",
            "source_fingerprints": [["source.py", "digest"]],
            "launch_hashes": ["wave", "body"],
            "conservation_summary_sha256": "conservation",
            "conservation_timeseries_sha256": "series",
            "provenance_manifest_sha256": "manifest",
        })
        total_x = -0.7 + 0.4 * factor**2
        record = {
            "status": "periodic_tsc_strang_momentum_diagnostic_not_a_calibration_release",
            "run": str(run), "factor": factor, "last_wave_step": stop,
            "slurm_job_id": "123", "source_commit": "commit",
            "source_fingerprints": [["source.py", "digest"]],
            "initial_wave_sha256": "wave", "initial_body_sha256": "body",
            "conservation_summary_sha256": "conservation",
            "conservation_timeseries_sha256": "series",
            "provenance_manifest_sha256": "manifest",
            "analysis_source_sha256": "hash", "calibration_eligible": False,
            "momentum_change_resolved_above_rounding_floor": True,
            "wave_momentum_initial_code": [0, 0, 0],
            "wave_momentum_final_code": [30, 0, 0],
            "body_momentum_initial_code": [0, 0, 0],
            "body_momentum_final_code": [total_x - 30, 0, 0],
            "total_momentum_change_code": [total_x, 0, 0],
        }
        (root / f"momentum_{label}_v2.json").write_text(json.dumps(record))
    summary_path = tmp_path / "trace.json"
    summary_path.write_text(json.dumps({
        "status": "periodic_tsc_strang_short_prefix_diagnostic_not_a_calibration_release",
        "trace_root": str(root), "calibration_eligible": False,
        "runs": runs,
    }))
    monkeypatch.setattr(analysis, "_sha256", lambda path: "hash")
    result = analysis.summarize(root, summary_path)
    assert result["momentum_diagnostic_passed"] is False
    assert result["momentum_limit_over_exchange"] == pytest.approx(0.7 / 30.7)
    assert result["total_momentum_observed_pair_orders"] == pytest.approx([2, 2])
    path = root / "momentum_f0125_v2.json"
    changed = json.loads(path.read_text())
    changed["source_commit"] = "other"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="provenance differs"):
        analysis.summarize(root, summary_path)
