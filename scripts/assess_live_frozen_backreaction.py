#!/usr/bin/env python3
"""Assess a SHA-bound live/frozen track pair without running a simulation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.backreaction import (
    assess_live_frozen_backreaction,
    read_backreaction_manifest,
)


def _write_json_atomic(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = read_backreaction_manifest(manifest_path)
    decision = assess_live_frozen_backreaction(
        model=manifest.model,
        live_points=manifest.live_points,
        frozen_points=manifest.frozen_points,
        evidence=manifest.evidence,
        config=manifest.config,
    )
    output = decision.as_dict()
    output["input_manifest"] = {
        "path": str(manifest.path),
        "sha256": manifest.sha256,
    }
    _write_json_atomic(args.output.expanduser().resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if decision.offline_acceptable else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
