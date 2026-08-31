#!/usr/bin/env python3
"""Assess a SHA-bound live/frozen track pair without running a simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from fdm_smbh_delay.backreaction import (
    BackreactionEvidence,
    BackreactionGateConfig,
    BackreactionTrackPoint,
    assess_live_frozen_backreaction,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be a JSON object")
    return record


def _read_track(path: Path) -> tuple[BackreactionTrackPoint, ...]:
    record = _read_json(path, "backreaction track")
    if set(record) != {"schema_version", "status", "track"}:
        raise ValueError("backreaction track fields are invalid")
    if record["schema_version"] != 1 or record["status"] != "measured_track":
        raise ValueError("backreaction track is not a measured_track schema-v1 record")
    raw_points = record["track"]
    if not isinstance(raw_points, list):
        raise ValueError("backreaction track must contain a list")
    points: list[BackreactionTrackPoint] = []
    expected = {
        "separation_pc",
        "orbital_power_pc2_myr3",
        "orbital_torque_msun_pc2_myr",
        "eccentricity",
    }
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ValueError(f"backreaction track point {index} fields are invalid")
        try:
            points.append(BackreactionTrackPoint(**raw))
        except (TypeError, ValueError) as error:
            raise ValueError(f"backreaction track point {index} is invalid: {error}") from error
    return tuple(points)


def _side(
    record: Any,
    *,
    label: str,
    base: Path,
) -> tuple[dict[str, Any], tuple[BackreactionTrackPoint, ...]]:
    expected = {
        "checkpoint_id",
        "source",
        "force_accounting",
        "maximum_relative_energy_error",
        "minimum_orbital_resolution_cells",
    }
    if not isinstance(record, Mapping) or set(record) != expected:
        raise ValueError(f"{label} evidence fields are invalid")
    source = record.get("source")
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
        raise ValueError(f"{label} source fields are invalid")
    source_path = Path(_nonempty(source.get("path"), f"{label} source path")).expanduser()
    source_path = (source_path if source_path.is_absolute() else base / source_path).resolve()
    declared_sha = _nonempty(source.get("sha256"), f"{label} source SHA-256").lower()
    if len(declared_sha) != 64 or any(ch not in "0123456789abcdef" for ch in declared_sha):
        raise ValueError(f"{label} source SHA-256 is invalid")
    try:
        actual_sha = _sha256(source_path)
    except OSError as error:
        raise ValueError(f"cannot read {label} source: {error}") from error
    if actual_sha != declared_sha:
        raise ValueError(f"{label} source SHA-256 differs")
    parsed = dict(record)
    parsed["checkpoint_id"] = _nonempty(parsed["checkpoint_id"], f"{label} checkpoint_id")
    parsed["source_path"] = str(source_path)
    parsed["source_sha256"] = actual_sha
    parsed["force_accounting"] = _nonempty(
        parsed["force_accounting"], f"{label} force_accounting"
    )
    points = _read_track(source_path)
    return parsed, points


def _gate_config(record: Any) -> BackreactionGateConfig:
    if record is None:
        return BackreactionGateConfig()
    if not isinstance(record, Mapping):
        raise ValueError("backreaction gates must be an object")
    allowed = {
        "minimum_overlap_factor",
        "minimum_overlap_points",
        "maximum_rate_fractional_difference",
        "maximum_eccentricity_difference",
        "maximum_relative_energy_error",
        "minimum_orbital_resolution_cells",
        "maximum_log_separation_match",
        "rate_floor_fraction",
    }
    if set(record) - allowed:
        raise ValueError("backreaction gates contain unsupported fields")
    config = BackreactionGateConfig(**dict(record))
    baseline = BackreactionGateConfig()
    if (
        config.minimum_overlap_factor < baseline.minimum_overlap_factor
        or config.minimum_overlap_points < baseline.minimum_overlap_points
        or config.maximum_rate_fractional_difference > baseline.maximum_rate_fractional_difference
        or config.maximum_eccentricity_difference > baseline.maximum_eccentricity_difference
        or config.maximum_relative_energy_error > baseline.maximum_relative_energy_error
        or config.minimum_orbital_resolution_cells < baseline.minimum_orbital_resolution_cells
        or config.maximum_log_separation_match > baseline.maximum_log_separation_match
        or config.rate_floor_fraction < baseline.rate_floor_fraction
    ):
        raise ValueError("backreaction gates may only be stricter than the project defaults")
    return config


def _write_json_atomic(path: Path, record: dict[str, Any]) -> None:
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
    manifest = _read_json(manifest_path, "backreaction manifest")
    expected = {"schema_version", "model", "live", "frozen"}
    optional = {"gates"}
    if set(manifest) - expected - optional or not expected.issubset(manifest):
        raise ValueError("backreaction manifest fields are invalid")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported backreaction manifest schema")
    model = _nonempty(manifest.get("model"), "backreaction model")
    live_record, live_points = _side(
        manifest.get("live"), label="live", base=manifest_path.parent
    )
    frozen_record, frozen_points = _side(
        manifest.get("frozen"), label="frozen", base=manifest_path.parent
    )
    evidence = BackreactionEvidence(
        model=model,
        live_checkpoint_id=live_record["checkpoint_id"],
        frozen_checkpoint_id=frozen_record["checkpoint_id"],
        live_source_path=live_record["source_path"],
        live_source_sha256=live_record["source_sha256"],
        frozen_source_path=frozen_record["source_path"],
        frozen_source_sha256=frozen_record["source_sha256"],
        live_force_accounting=live_record["force_accounting"],
        frozen_force_accounting=frozen_record["force_accounting"],
        maximum_live_relative_energy_error=live_record["maximum_relative_energy_error"],
        maximum_frozen_relative_energy_error=frozen_record["maximum_relative_energy_error"],
        minimum_live_orbital_resolution_cells=live_record["minimum_orbital_resolution_cells"],
        minimum_frozen_orbital_resolution_cells=frozen_record["minimum_orbital_resolution_cells"],
    )
    decision = assess_live_frozen_backreaction(
        model=model,
        live_points=live_points,
        frozen_points=frozen_points,
        evidence=evidence,
        config=_gate_config(manifest.get("gates")),
    )
    output = decision.as_dict()
    output["input_manifest"] = {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
    }
    _write_json_atomic(args.output.expanduser().resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if decision.offline_acceptable else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
