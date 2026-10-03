from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.reassess_qe_extension import RUN_INPUTS
from scripts.verify_qe_reassessment import verify_assessment


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,run_id,effective_grid_cells\n"
        "qe_test,qe_test_n512,512\nqe_test,qe_test_n256,256\n"
    )
    root = tmp_path / "torch"
    row = {"case_id": "qe_test", "accepted_bins": 0,
           "production_calibration_row_admitted": False,
           "matched_bins": 0,
           "selection_status": "matched_separation_bins_evaluated",
           "status": "no_matched_bins_censored", "rejected_bins": []}
    for label, resolution in (("fine", 512), ("coarse", 256)):
        run = root / f"qe_test_n{resolution}"
        run.mkdir(parents=True)
        for name in RUN_INPUTS:
            contents = {"status": "complete"} if name == "torch_run_summary.json" else (
                {"status": "diagnosed"} if name == "wave_response_summary.json" else {}
            )
            if name == "fdm_adapter_metadata.json":
                contents = {"case_id": "qe_test", "resolution": resolution}
            (run / name).write_text(json.dumps(contents) if name.endswith(".json") else "time,value\n0,0\n")
        row[f"{label}_run"] = str(run.resolve())
        row[f"{label}_input_sha256"] = {name: _sha(run / name) for name in RUN_INPUTS}
    output = tmp_path / "assessment"
    output.mkdir()
    convergence = output / "qe_test_n512_n256.json"
    convergence.write_text(json.dumps({
        "status": "common_resolved_interval_compared",
        "matched_separation": {"retained_bins": 0,
                               "selection_status": "matched_separation_bins_evaluated",
                               "bins": []},
    }))
    row["convergence_file"] = str(convergence.resolve())
    row["convergence_sha256"] = _sha(convergence)
    assessment = output / "assessment.json"
    assessment.write_text(json.dumps({
        "schema_version": 1, "status": "qe_finest_adjacent_pairs_reassessed",
        "manifest": str(manifest.resolve()), "manifest_sha256": _sha(manifest),
        "torch_root": str(root.resolve()), "profile_id": "test",
        "case_count": 1, "candidate_case_count": 0,
        "production_calibration_row_admitted": False, "cases": [row],
    }))
    return assessment


def test_verified_zero_bin_assessment_is_censored(tmp_path: Path) -> None:
    assessment = _fixture(tmp_path)
    result = verify_assessment(assessment)
    assert result["status"] == "verified_no_accepted_bins_censored"
    assert result["case_count"] == 1
    assert result["accepted_bins"] == 0
    assert result["production_calibration_row_admitted"] is False


def test_tampered_run_input_is_rejected(tmp_path: Path) -> None:
    assessment = _fixture(tmp_path)
    (tmp_path / "torch" / "qe_test_n512" / "conservation_summary.json").write_text('{"modified":true}')
    with pytest.raises(ValueError, match="input checksum mismatch"):
        verify_assessment(assessment)


def test_tampered_convergence_is_rejected(tmp_path: Path) -> None:
    assessment = _fixture(tmp_path)
    (assessment.parent / "qe_test_n512_n256.json").write_text('{}')
    with pytest.raises(ValueError, match="convergence checksum mismatch"):
        verify_assessment(assessment)


def test_false_acceptance_count_is_rejected(tmp_path: Path) -> None:
    assessment = _fixture(tmp_path)
    data = json.loads(assessment.read_text())
    data["cases"][0]["accepted_bins"] = 1
    assessment.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="censoring decision is inconsistent"):
        verify_assessment(assessment)
