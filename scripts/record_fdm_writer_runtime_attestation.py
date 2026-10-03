#!/usr/bin/env python3
"""Record artifacts from an operator-completed FDM writer integration test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.pure_fdm_outer_submission import (
    build_fdm_writer_runtime_attestation,
)


def _write_json_atomic(path: Path, record: dict[str, object]) -> None:
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
    parser.add_argument("source", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("fdm_sidecar", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--build-manifest",
        required=True,
        type=Path,
        help=(
            "clean-build source manifest containing output_amr, output_fdm, "
            "and bin/Makefile hashes"
        ),
    )
    parser.add_argument(
        "--operator-confirmed",
        action="store_true",
        help=(
            "confirm that the compiled writer integration test was already run; "
            "this command itself never executes the binary"
        ),
    )
    args = parser.parse_args()
    try:
        record = build_fdm_writer_runtime_attestation(
            args.source,
            args.executable,
            args.fdm_sidecar,
            build_manifest=args.build_manifest,
            operator_confirmed=args.operator_confirmed,
        )
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        print(json.dumps({"status": "runtime_attestation_not_recorded", "reason": str(error)}))
        return 2
    destination = args.output.expanduser().resolve()
    _write_json_atomic(destination, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
