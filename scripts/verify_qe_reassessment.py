#!/usr/bin/env python3
"""Verify provenance and censoring of an existing q/e reassessment.

This is a bounded, read-only audit of named diagnostics, not a replacement for
the doubled-box comparison required before a production calibration release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.reassess_qe_extension import RUN_INPUTS, finest_adjacent_pairs
from fdm_smbh_delay.subgrid_table_builder import CalibrationSource, build_source_rows


MAX_INPUT_BYTES = 16 * 1024 * 1024


def _small_bytes(path: Path) -> bytes:
    if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"q/e audit input absent or exceeds size limit: {path}")
    with path.open("rb") as stream:
        payload = stream.read(MAX_INPUT_BYTES + 1)
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError(f"q/e audit input exceeds size limit: {path}")
    return payload


def _hash(path: Path) -> str:
    return hashlib.sha256(_small_bytes(path)).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(_small_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"q/e audit JSON is not an object: {path}")
    return value


def verify_assessment(path: Path) -> dict:
    """Check the manifest, named run inputs, comparisons, and saved decisions."""

    path = path.expanduser().resolve()
    assessment = _json(path)
    if (assessment.get("schema_version") != 1
            or assessment.get("status") != "qe_finest_adjacent_pairs_reassessed"
            or assessment.get("production_calibration_row_admitted") is not False
            or not assessment.get("profile_id")):
        raise ValueError("q/e reassessment status or release claim is invalid")
    manifest = Path(assessment["manifest"]).resolve()
    root = Path(assessment["torch_root"]).resolve()
    if _hash(manifest) != assessment.get("manifest_sha256"):
        raise ValueError("q/e manifest checksum mismatch")
    pairs = finest_adjacent_pairs(manifest)
    cases = assessment.get("cases")
    if not isinstance(cases, list) or len(cases) != len(pairs) or assessment.get("case_count") != len(pairs):
        raise ValueError("q/e reassessment case count mismatch")
    candidates = 0
    accepted_total = 0
    for row, (case_id, fine, coarse) in zip(cases, pairs, strict=True):
        if row.get("case_id") != case_id or row.get("production_calibration_row_admitted") is not False:
            raise ValueError(f"q/e case identity or release claim is invalid: {case_id}")
        for label, (resolution, run_id) in (("fine", fine), ("coarse", coarse)):
            run = (root / run_id).resolve()
            if run.parent != root or row.get(f"{label}_run") != str(run):
                raise ValueError(f"q/e run identity mismatch: {run_id}")
            hashes = row.get(f"{label}_input_sha256")
            if not isinstance(hashes, dict) or set(hashes) != set(RUN_INPUTS):
                raise ValueError(f"q/e run checksum set is incomplete: {run_id}")
            metadata = _json(run / "fdm_adapter_metadata.json")
            if metadata.get("case_id") != case_id or metadata.get("resolution") != resolution:
                raise ValueError(f"q/e run metadata identity mismatch: {run_id}")
            for name in RUN_INPUTS:
                if _hash(run / name) != hashes[name]:
                    raise ValueError(f"q/e run input checksum mismatch: {run_id}/{name}")
            for name, expected in (("torch_run_summary.json", "complete"),
                                   ("wave_response_summary.json", "diagnosed")):
                if _json(run / name).get("status") != expected:
                    raise ValueError(f"q/e run is incomplete: {run_id}/{name}")
        accepted = row.get("accepted_bins")
        if type(accepted) is not int or accepted < 0:
            raise ValueError(f"q/e accepted-bin count is invalid: {case_id}")
        accepted_total += accepted
        if row.get("status") == "no_common_interval_censored":
            if accepted or "convergence_file" in row or not row.get("reason"):
                raise ValueError(f"q/e no-overlap censoring is invalid: {case_id}")
            continue
        convergence = Path(row["convergence_file"]).resolve()
        if convergence.parent != path.parent or _hash(convergence) != row.get("convergence_sha256"):
            raise ValueError(f"q/e convergence checksum mismatch: {case_id}")
        comparison = _json(convergence)
        if comparison.get("status") != "common_resolved_interval_compared":
            raise ValueError(f"q/e convergence is incomplete: {case_id}")
        matched = comparison.get("matched_separation")
        count = 0 if matched is None else matched.get("retained_bins")
        selection = ("no_common_resolved_separation_support" if matched is None
                     else matched.get("selection_status"))
        if type(count) is not int or count < 0 or row.get("matched_bins") != count or row.get("selection_status") != selection:
            raise ValueError(f"q/e matched-bin record is inconsistent: {case_id}")
        if matched is not None and (not isinstance(matched.get("bins"), list) or len(matched["bins"]) != count):
            raise ValueError(f"q/e matched-bin list is inconsistent: {case_id}")
        if matched is None:
            status = "no_common_resolved_separation_censored"
        elif count == 0:
            status = "no_matched_bins_censored"
        else:
            result = build_source_rows(CalibrationSource(assessment["profile_id"], convergence))
            if accepted != len(result.accepted_rows) or row.get("rejected_bins") != list(result.rejected_bins):
                raise ValueError(f"q/e numerical gate record is inconsistent: {case_id}")
            status = ("candidate_needs_doubled_box_control" if accepted
                      else "matched_bins_rejected_censored")
        if row.get("status") != status or (count == 0 and accepted != 0):
            raise ValueError(f"q/e censoring decision is inconsistent: {case_id}")
        candidates += bool(accepted)
    if assessment.get("candidate_case_count") != candidates:
        raise ValueError("q/e candidate-case count mismatch")
    return {
        "status": "verified_candidates_require_doubled_box_control" if candidates else "verified_no_accepted_bins_censored",
        "assessment_sha256": _hash(path),
        "case_count": len(cases),
        "candidate_case_count": candidates,
        "accepted_bins": accepted_total,
        "production_calibration_row_admitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("assessment", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_assessment(args.assessment), sort_keys=True))


if __name__ == "__main__":
    main()
