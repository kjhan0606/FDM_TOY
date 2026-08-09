"""Reproducible result serialization."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import CaseConfig
from .orbit import IntegrationResult


def input_hash(raw_config: dict[str, Any]) -> str:
    canonical = json.dumps(raw_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def source_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("astropy", "numpy", "PyYAML", "scipy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_result(
    result: IntegrationResult,
    config: CaseConfig,
    output_directory: str | Path,
    *,
    write_timeseries: bool = True,
) -> Path:
    """Write summary, exact config, and optionally the sampled CSV history."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(result.summary)
    summary["input_hash"] = input_hash(config.raw)
    summary["provenance"] = {
        "source_commit": source_revision(),
        "dependencies": dependency_versions(),
        "solver": "scipy.integrate.solve_ivp:DOP853",
        "rtol": config.integration.rtol,
        "atol": config.integration.atol,
        "max_step_myr": config.integration.max_step_myr,
        "internal_units": {"length": "pc", "time": "Myr", "mass": "Msun"},
        "solver_message": result.solver_message,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (output / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.raw, stream, sort_keys=False, allow_unicode=True)
    if write_timeseries:
        names = list(result.timeseries)
        rows = zip(*(result.timeseries[name] for name in names), strict=True)
        with (output / "timeseries.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(names)
            writer.writerows(rows)
    return output
