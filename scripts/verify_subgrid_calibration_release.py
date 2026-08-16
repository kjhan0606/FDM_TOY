#!/usr/bin/env python3
"""Verify a production subgrid release and write a durable audit record."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from fdm_smbh_delay.subgrid_calibration import (
    SubgridCalibrationTable,
    find_mass_interpolation_witness,
    verify_subgrid_runtime,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("--required-profile", action="append", default=[])
    parser.add_argument(
        "--mass-interpolation-profile",
        action="append",
        default=[],
    )
    parser.add_argument("--expected-source-count", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    release = arguments.release.expanduser().resolve()
    summary_path = release.with_suffix(".summary.json")
    table = SubgridCalibrationTable.from_release(release)
    summary = json.loads(summary_path.read_text())
    profiles = sorted({row.profile_id for row in table.rows})
    required_profiles = sorted(set(arguments.required_profile))
    if required_profiles and profiles != required_profiles:
        raise SystemExit(
            "accepted release profiles do not match the required profiles"
        )
    if (
        arguments.expected_source_count is not None
        and len(summary["sources"]) != arguments.expected_source_count
    ):
        raise SystemExit("subgrid release source count is invalid")
    if sum(source["accepted_bins"] for source in summary["sources"]) != len(
        table.rows
    ):
        raise SystemExit("subgrid release source counts do not close")

    mass_interpolation_witnesses = {}
    for profile_id in sorted(set(arguments.mass_interpolation_profile)):
        masses = sorted(
            {
                row.binary_to_soliton_mass
                for row in table.rows
                if row.profile_id == profile_id
            }
        )
        if len(masses) < 2:
            raise SystemExit(
                f"accepted {profile_id} data do not span two mass planes"
            )
        witness = find_mass_interpolation_witness(
            table,
            profile_id=profile_id,
        )
        if witness is None:
            raise SystemExit(
                f"accepted {profile_id} mass planes have no usable "
                "separation overlap"
            )
        mass_interpolation_witnesses[profile_id] = asdict(witness)

    runtime = verify_subgrid_runtime(table)
    output = (
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else release.with_suffix(".verification.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "subgrid_calibration_release_verified",
        "schema_version": 1,
        "release_input_sha256": summary["release_input_sha256"],
        "table": {
            "file": str(release),
            "sha256": _sha256(release),
            "rows": len(table.rows),
        },
        "summary": {
            "file": str(summary_path),
            "sha256": _sha256(summary_path),
        },
        "profiles": profiles,
        "sources": len(summary["sources"]),
        "mass_interpolation_witnesses": mass_interpolation_witnesses,
        "runtime": asdict(runtime),
    }
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
