#!/usr/bin/env python3
"""Reassess the finest adjacent resolution pair of each completed q-e case.

This evaluates existing diagnostics only. A matched-bin candidate is not a
production calibration release without the separate doubled-box control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from fdm_smbh_delay.convergence import load_convergence_run, summarize_convergence
from fdm_smbh_delay.subgrid_table_builder import CalibrationSource, build_source_rows


RUN_INPUTS = (
    "fdm_adapter_metadata.json",
    "config.uldm",
    "torch_run_summary.json",
    "wave_response_summary.json",
    "conservation_summary.json",
    "orbit_averaged_exchange_summary.json",
    "conservation_timeseries.csv",
    "orbit_averaged_exchange.csv",
    "wave_response_timeseries.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finest_adjacent_pairs(manifest: Path) -> list[tuple[str, tuple[int, str], tuple[int, str]]]:
    """Return case ID, finest run, and its next-coarser manifest run."""

    grouped: dict[str, list[tuple[int, str]]] = {}
    seen_run_ids: set[str] = set()
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                case_id = row["case_id"]
                run_id = row["run_id"]
                resolution = int(row["effective_grid_cells"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("q-e manifest lacks required run columns") from error
            if not case_id or not run_id or resolution < 1 or run_id in seen_run_ids:
                raise ValueError("q-e manifest contains an invalid run identity")
            seen_run_ids.add(run_id)
            grouped.setdefault(case_id, []).append((resolution, run_id))
    if not grouped:
        raise ValueError("q-e manifest has no cases")
    pairs = []
    for case_id, runs in sorted(grouped.items()):
        ordered = sorted(runs, reverse=True)
        if len(ordered) < 2 or len({level for level, _ in ordered}) != len(ordered):
            raise ValueError(f"{case_id} lacks two distinct resolution levels")
        pairs.append((case_id, ordered[0], ordered[1]))
    return pairs


def _require_complete(run: Path, *, case_id: str, resolution: int) -> dict[str, str]:
    metadata_path = run / "fdm_adapter_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"q-e run lacks adapter metadata: {run}")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("case_id") != case_id
        or metadata.get("resolution") != resolution
    ):
        raise ValueError(f"q-e run identity disagrees with manifest: {run}")
    for name, status in (
        ("torch_run_summary.json", "complete"),
        ("wave_response_summary.json", "diagnosed"),
    ):
        path = run / name
        if not path.is_file():
            raise ValueError(f"q-e run lacks completed {name}: {run}")
        summary = json.loads(path.read_text())
        if summary.get("status") != status:
            raise ValueError(f"q-e run lacks completed {name}: {run}")
        if "run" in summary and Path(summary["run"]).resolve() != run.resolve():
            raise ValueError(f"q-e run summary points to another directory: {path}")
    missing = [name for name in RUN_INPUTS if not (run / name).is_file()]
    if missing:
        raise ValueError(f"q-e run lacks required diagnostic inputs {missing}: {run}")
    return {name: _sha256(run / name) for name in RUN_INPUTS}


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reassess(
    manifest: Path,
    torch_root: Path,
    output_dir: Path,
    *,
    profile_id: str,
    source_revision: str | None = None,
) -> dict:
    """Publish only a complete assessment; structural input errors abort it."""

    manifest_sha256 = _sha256(manifest)
    pairs = finest_adjacent_pairs(manifest)
    if not profile_id:
        raise ValueError("calibration profile ID is required")
    input_hashes: dict[str, dict[str, str]] = {}
    for case_id, fine, coarse in pairs:
        input_hashes[fine[1]] = _require_complete(
            torch_root / fine[1], case_id=case_id, resolution=fine[0]
        )
        input_hashes[coarse[1]] = _require_complete(
            torch_root / coarse[1], case_id=case_id, resolution=coarse[0]
        )
    if output_dir.exists():
        raise FileExistsError(f"q-e assessment already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.partial-", dir=output_dir.parent
    ) as staging_name:
        return _finish_reassessment(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            pairs=pairs,
            input_hashes=input_hashes,
            torch_root=torch_root,
            output_dir=output_dir,
            staging=Path(staging_name),
            profile_id=profile_id,
            source_revision=source_revision,
        )


def _finish_reassessment(
    *,
    manifest: Path,
    manifest_sha256: str,
    pairs: list[tuple[str, tuple[int, str], tuple[int, str]]],
    input_hashes: dict[str, dict[str, str]],
    torch_root: Path,
    output_dir: Path,
    staging: Path,
    profile_id: str,
    source_revision: str | None,
) -> dict:
    """Build inside an owned temporary directory and publish atomically."""

    cases = []
    for case_id, fine, coarse in pairs:
        fine_run = torch_root / fine[1]
        coarse_run = torch_root / coarse[1]
        loaded = [
            load_convergence_run(f"n{fine[0]}", fine_run),
            load_convergence_run(f"n{coarse[0]}", coarse_run),
        ]
        try:
            summary = summarize_convergence(loaded)
        except ValueError as error:
            if "no shared resolved interval" not in str(error):
                raise
            cases.append({
                "case_id": case_id,
                "fine_run": str(fine_run.resolve()),
                "coarse_run": str(coarse_run.resolve()),
                "fine_input_sha256": input_hashes[fine[1]],
                "coarse_input_sha256": input_hashes[coarse[1]],
                "accepted_bins": 0,
                "production_calibration_row_admitted": False,
                "status": "no_common_interval_censored",
                "reason": str(error),
            })
            continue
        name = f"{case_id}_n{fine[0]}_n{coarse[0]}.json"
        path = staging / name
        _write_json(path, summary)
        matched = summary["matched_separation"]
        row = {
            "case_id": case_id,
            "fine_run": str(fine_run.resolve()),
            "coarse_run": str(coarse_run.resolve()),
            "fine_input_sha256": input_hashes[fine[1]],
            "coarse_input_sha256": input_hashes[coarse[1]],
            "convergence_file": str((output_dir / name).resolve()),
            "convergence_sha256": _sha256(path),
            "matched_bins": 0 if matched is None else int(matched["retained_bins"]),
            "selection_status": (
                "no_common_resolved_separation_support"
                if matched is None else matched["selection_status"]
            ),
            "accepted_bins": 0,
            "rejected_bins": [],
            "production_calibration_row_admitted": False,
        }
        if matched is None:
            row["status"] = "no_common_resolved_separation_censored"
            cases.append(row)
            continue
        if matched["bins"]:
            assessed = build_source_rows(
                CalibrationSource(profile_id=profile_id, convergence_summary=path)
            )
            row["accepted_bins"] = len(assessed.accepted_rows)
            row["rejected_bins"] = list(assessed.rejected_bins)
        if row["accepted_bins"]:
            row["status"] = "candidate_needs_doubled_box_control"
        elif matched["bins"]:
            row["status"] = "matched_bins_rejected_censored"
        else:
            row["status"] = "no_matched_bins_censored"
        cases.append(row)
    if _sha256(manifest) != manifest_sha256:
        raise ValueError("q-e manifest changed during reassessment")
    for case_id, fine, coarse in pairs:
        for resolution, run_id in (fine, coarse):
            current = _require_complete(
                torch_root / run_id, case_id=case_id, resolution=resolution
            )
            if current != input_hashes[run_id]:
                raise ValueError(f"q-e run inputs changed during reassessment: {run_id}")
    assessment = {
        "schema_version": 1,
        "status": "qe_finest_adjacent_pairs_reassessed",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "torch_root": str(torch_root.resolve()),
        "profile_id": profile_id,
        "case_count": len(cases),
        "candidate_case_count": sum(bool(row["accepted_bins"]) for row in cases),
        "production_calibration_row_admitted": False,
        "cases": cases,
    }
    if source_revision is not None:
        assessment["source_revision"] = source_revision
    _write_json(staging / "assessment.json", assessment)
    if output_dir.exists():
        raise FileExistsError(f"q-e assessment appeared during computation: {output_dir}")
    os.rename(staging, output_dir)
    directory_fd = os.open(output_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--torch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("q-e reassessment requires a clean source checkout")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    result = reassess(
        args.manifest.expanduser().resolve(),
        args.torch_root.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        profile_id=args.profile_id,
        source_revision=revision,
    )
    print(json.dumps({
        "status": result["status"],
        "case_count": result["case_count"],
        "candidate_case_count": result["candidate_case_count"],
        "assessment": str(args.output_dir.expanduser().resolve() / "assessment.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
