from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import reassess_qe_extension as reassessment


def _manifest(path: Path) -> None:
    path.write_text(
        "case_id,run_id,effective_grid_cells\n"
        "case_a,case_a_n128,128\n"
        "case_a,case_a_n256,256\n"
        "case_a,case_a_n512,512\n"
    )


def test_finest_adjacent_pair_skips_obsolete_coarse_level(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest)
    assert reassessment.finest_adjacent_pairs(manifest) == [
        ("case_a", (512, "case_a_n512"), (256, "case_a_n256"))
    ]


def test_missing_run_is_rejected_before_output_creation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest)
    output = tmp_path / "assessment"
    with pytest.raises(ValueError, match="lacks adapter metadata"):
        reassessment.reassess(
            manifest, tmp_path / "torch", output, profile_id="boey2025"
        )
    assert not output.exists()


def test_wrong_manifest_identity_is_rejected_before_output_creation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest)
    torch_root = tmp_path / "torch"
    for run_id in ("case_a_n256", "case_a_n512"):
        run = torch_root / run_id
        run.mkdir(parents=True)
        (run / "fdm_adapter_metadata.json").write_text(
            json.dumps({"case_id": "another_case", "resolution": int(run_id[-3:])})
        )
        (run / "torch_run_summary.json").write_text('{"status":"complete"}')
        (run / "wave_response_summary.json").write_text('{"status":"diagnosed"}')
        for name in reassessment.RUN_INPUTS:
            path = run / name
            if not path.exists():
                path.write_text("fixture\n")
    output = tmp_path / "assessment"
    with pytest.raises(ValueError, match="identity disagrees"):
        reassessment.reassess(
            manifest, torch_root, output, profile_id="boey2025"
        )
    assert not output.exists()


def test_numerical_candidate_is_not_promoted_without_box_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest)
    torch_root = tmp_path / "torch"
    for run_id in ("case_a_n256", "case_a_n512"):
        run = torch_root / run_id
        run.mkdir(parents=True)
        (run / "fdm_adapter_metadata.json").write_text(
            json.dumps({"case_id": "case_a", "resolution": int(run_id[-3:])})
        )
        (run / "torch_run_summary.json").write_text('{"status":"complete"}')
        (run / "wave_response_summary.json").write_text('{"status":"diagnosed"}')
        for name in reassessment.RUN_INPUTS:
            path = run / name
            if not path.exists():
                path.write_text("fixture\n")
    monkeypatch.setattr(
        reassessment, "load_convergence_run", lambda label, path: {"label": label}
    )
    monkeypatch.setattr(
        reassessment,
        "summarize_convergence",
        lambda runs: {
            "matched_separation": {
                "retained_bins": 1,
                "selection_status": "matched_separation_bins_evaluated",
                "bins": [{"bin": 0}],
            }
        },
    )
    monkeypatch.setattr(
        reassessment,
        "build_source_rows",
        lambda source: SimpleNamespace(accepted_rows=(object(),), rejected_bins=()),
    )
    output = tmp_path / "assessment"
    record = reassessment.reassess(
        manifest, torch_root, output, profile_id="boey2025"
    )
    saved = json.loads((output / "assessment.json").read_text())
    assert record == saved
    assert saved["candidate_case_count"] == 1
    assert saved["production_calibration_row_admitted"] is False
    assert saved["cases"][0]["status"] == "candidate_needs_doubled_box_control"
    with pytest.raises(FileExistsError):
        reassessment.reassess(
            manifest, torch_root, output, profile_id="boey2025"
        )


def test_failed_comparison_does_not_publish_partial_assessment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest)
    torch_root = tmp_path / "torch"
    for run_id in ("case_a_n256", "case_a_n512"):
        run = torch_root / run_id
        run.mkdir(parents=True)
        (run / "fdm_adapter_metadata.json").write_text(
            json.dumps({"case_id": "case_a", "resolution": int(run_id[-3:])})
        )
        (run / "torch_run_summary.json").write_text('{"status":"complete"}')
        (run / "wave_response_summary.json").write_text('{"status":"diagnosed"}')
        for name in reassessment.RUN_INPUTS:
            path = run / name
            if not path.exists():
                path.write_text("fixture\n")
    monkeypatch.setattr(
        reassessment, "load_convergence_run", lambda label, path: {"label": label}
    )

    def fail(_runs):
        raise ValueError("invalid comparison")

    monkeypatch.setattr(reassessment, "summarize_convergence", fail)
    output = tmp_path / "assessment"
    with pytest.raises(ValueError, match="invalid comparison"):
        reassessment.reassess(
            manifest, torch_root, output, profile_id="boey2025"
        )
    assert not output.exists()


def test_no_shared_resolved_interval_is_censored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest)
    torch_root = tmp_path / "torch"
    for run_id in ("case_a_n256", "case_a_n512"):
        run = torch_root / run_id
        run.mkdir(parents=True)
        (run / "fdm_adapter_metadata.json").write_text(
            json.dumps({"case_id": "case_a", "resolution": int(run_id[-3:])})
        )
        (run / "torch_run_summary.json").write_text('{"status":"complete"}')
        (run / "wave_response_summary.json").write_text('{"status":"diagnosed"}')
        for name in reassessment.RUN_INPUTS:
            path = run / name
            if not path.exists():
                path.write_text("fixture\n")
    monkeypatch.setattr(
        reassessment, "load_convergence_run", lambda label, path: {"label": label}
    )

    def no_overlap(_runs):
        raise ValueError("the calculations have no shared resolved interval")

    monkeypatch.setattr(reassessment, "summarize_convergence", no_overlap)
    output = tmp_path / "assessment"
    record = reassessment.reassess(
        manifest, torch_root, output, profile_id="boey2025"
    )
    assert record["cases"][0]["status"] == "no_common_interval_censored"
    assert record["candidate_case_count"] == 0
    assert (output / "assessment.json").is_file()


def test_no_common_resolved_separation_is_censored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest)
    torch_root = tmp_path / "torch"
    for run_id in ("case_a_n256", "case_a_n512"):
        run = torch_root / run_id
        run.mkdir(parents=True)
        (run / "fdm_adapter_metadata.json").write_text(
            json.dumps({"case_id": "case_a", "resolution": int(run_id[-3:])})
        )
        (run / "torch_run_summary.json").write_text('{"status":"complete"}')
        (run / "wave_response_summary.json").write_text('{"status":"diagnosed"}')
        for name in reassessment.RUN_INPUTS:
            path = run / name
            if not path.exists():
                path.write_text("fixture\n")
    monkeypatch.setattr(
        reassessment, "load_convergence_run", lambda label, path: {"label": label}
    )
    monkeypatch.setattr(
        reassessment,
        "summarize_convergence",
        lambda runs: {"matched_separation": None},
    )
    output = tmp_path / "assessment"
    record = reassessment.reassess(
        manifest, torch_root, output, profile_id="boey2025"
    )
    assert record["cases"][0]["status"] == "no_common_resolved_separation_censored"
    assert record["cases"][0]["matched_bins"] == 0
    assert record["candidate_case_count"] == 0
    assert (output / "assessment.json").is_file()
