#!/usr/bin/env python3
"""Create a clean, source-bound manifest for a lagRamses writer build.

This command inspects a checkout but never compiles or executes it.  The
manifest is consumed by ``record_fdm_writer_runtime_attestation.py`` and
keeps the selected output writer, FDM writer, and generated Makefile inputs
bound to one full Git revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


_ROLES = (
    ("output_amr", Path("patch/lagRamses/output_amr.kjhan.f90")),
    ("output_fdm", Path("patch/lagRamses/output_fdm.f90")),
    ("bin_makefile", Path("bin/Makefile")),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot inspect Git checkout: {error}") from error
    return result.stdout.strip()


def _write_atomic(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    repository = args.repository.expanduser().resolve()
    if not repository.is_dir():
        raise SystemExit(f"repository is not a directory: {repository}")
    dirty = bool(_git(repository, "status", "--porcelain=v1"))
    if dirty:
        raise SystemExit("refusing to attest a dirty lagRamses checkout")
    commit = _git(repository, "rev-parse", "HEAD").lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise SystemExit("checkout did not report a full Git commit SHA-1")
    files: list[dict[str, str]] = []
    for role, relative in _ROLES:
        path = (repository / relative).resolve()
        if not path.is_file():
            raise SystemExit(f"required build source is missing: {path}")
        files.append({"role": role, "path": str(path), "sha256": _sha256(path)})
    record = {
        "schema_version": 1,
        "status": "lagramses_build_source_manifest",
        "repository": str(repository),
        "git_commit": commit,
        "git_dirty": False,
        "files": files,
    }
    _write_atomic(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
