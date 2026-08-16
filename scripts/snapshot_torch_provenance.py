#!/usr/bin/env python3
"""Freeze the exact uncommitted Torch solver used by existing calculations."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


_UNCOMMITTED_SOLVER_PATHS = (
    Path("scripts/run_torch_wave_case.py"),
    Path("src/fdm_smbh_delay/torch_wave.py"),
)
_COMMITTED_DEPENDENCY_PATHS = (Path("src/fdm_smbh_delay/pyul.py"),)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _git_blob(project: Path, revision: str, path: Path) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{path.as_posix()}"],
        cwd=project,
        check=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def _publish_immutable(path: Path, content: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"refusing to replace a different provenance file: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _load_existing_snapshot(
    *,
    destination: Path,
    resolved_run: Path,
    metadata_path: Path,
    revision: str,
) -> dict[str, Any] | None:
    """Validate and reuse launch-time source without consulting the worktree."""

    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        return None
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        existing.get("status") != "source_snapshot"
        or existing.get("run") != str(resolved_run)
        or existing.get("adapter_revision") != revision
    ):
        raise ValueError(f"existing provenance manifest identity is invalid: {manifest_path}")
    expected_inputs = {
        "fdm_adapter_metadata_sha256": _sha256(metadata_path.read_bytes()),
        "config_sha256": _sha256((resolved_run / "config.uldm").read_bytes()),
    }
    recorded_inputs = existing.get("input_records")
    if not isinstance(recorded_inputs, dict) or any(
        recorded_inputs.get(key) != value for key, value in expected_inputs.items()
    ):
        raise ValueError(
            f"existing provenance manifest input hashes are stale: {manifest_path}"
        )
    records = existing.get("source_files")
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"existing provenance manifest has no source records: {manifest_path}"
        )
    for record in records:
        relative = Path(str(record.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"existing provenance source path is invalid: {manifest_path}"
            )
        frozen = destination / "source" / relative
        try:
            content = frozen.read_bytes()
        except OSError as error:
            raise FileExistsError(
                f"refusing to replace a missing provenance file: {frozen}"
            ) from error
        if (
            len(content) != int(record.get("size_bytes", -1))
            or _sha256(content) != record.get("sha256")
        ):
            raise FileExistsError(
                f"refusing to replace a different provenance file: {frozen}"
            )
    return existing


def _snapshot_run(project: Path, run: Path) -> dict[str, Any]:
    resolved = run.expanduser().resolve()
    metadata_path = resolved / "fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("backend") not in {"pytorch_cpu", "pytorch_cuda"}:
        raise ValueError(f"not a Torch live-wave calculation: {resolved}")
    revision = str(metadata["adapter_revision"])
    metadata_mtime_ns = metadata_path.stat().st_mtime_ns
    destination = resolved / "torch_solver_provenance"
    existing = _load_existing_snapshot(
        destination=destination,
        resolved_run=resolved,
        metadata_path=metadata_path,
        revision=revision,
    )
    if existing is not None:
        return existing

    records = []
    for relative in _UNCOMMITTED_SOLVER_PATHS:
        source = project / relative
        source_stat = source.stat()
        if source_stat.st_mtime_ns > metadata_mtime_ns:
            raise ValueError(
                f"{source} is newer than {metadata_path}; its launch-time "
                "contents can no longer be established from timestamps"
            )
        content = source.read_bytes()
        target = destination / "source" / relative
        _publish_immutable(target, content)
        records.append(
            {
                "path": relative.as_posix(),
                "origin": "launch_worktree",
                "sha256": _sha256(content),
                "size_bytes": len(content),
                "source_mtime_ns": source_stat.st_mtime_ns,
                "source_predates_run_metadata": True,
            }
        )

    for relative in _COMMITTED_DEPENDENCY_PATHS:
        content = _git_blob(project, revision, relative)
        current = (project / relative).read_bytes()
        if current != content:
            raise ValueError(
                f"current numerical dependency differs from the preserved "
                f"launch revision: {relative}"
            )
        target = destination / "source" / relative
        _publish_immutable(target, content)
        records.append(
            {
                "path": relative.as_posix(),
                "origin": f"git_blob:{revision}",
                "sha256": _sha256(content),
                "size_bytes": len(content),
            }
        )

    manifest = {
        "status": "source_snapshot",
        "run": str(resolved),
        "adapter_revision": revision,
        "adapter_worktree_clean_at_reference_launch": metadata.get(
            "adapter_worktree_clean"
        ),
        "snapshot_basis": (
            "the uncommitted Torch runner and numerical operators have file "
            "modification times earlier than this run's metadata; committed "
            "unit-conversion code is read from the recorded adapter revision"
        ),
        "source_files": records,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "numba": _package_version("numba"),
            "torch": metadata.get("torch_version", _package_version("torch")),
            "torch_cuda": metadata.get("cuda_version"),
            "device": metadata.get("device"),
            "device_name": metadata.get("device_name"),
        },
        "input_records": {
            "fdm_adapter_metadata_sha256": _sha256(metadata_path.read_bytes()),
            "config_sha256": _sha256((resolved / "config.uldm").read_bytes()),
            "reference_initial_state": metadata.get("reference_initial_state"),
            "pyul_revision": metadata.get("pyul_revision"),
        },
    }
    manifest_content = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    manifest_path = destination / "manifest.json"
    _publish_immutable(manifest_path, manifest_content)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    snapshots = [_snapshot_run(project, run) for run in args.runs]
    print(
        json.dumps(
            {
                "status": "snapshotted",
                "runs": [snapshot["run"] for snapshot in snapshots],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
