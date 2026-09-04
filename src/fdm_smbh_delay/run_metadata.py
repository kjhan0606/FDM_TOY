"""Compatibility accessors for live-wave run metadata."""

from __future__ import annotations

import json
import math
from pathlib import Path


def saved_interval_count(metadata: dict, config: dict) -> int:
    """Return the diagnostic interval count from new or legacy run records."""

    if "save_number" in metadata:
        count = int(metadata["save_number"])
    else:
        try:
            count = int(config["Save Options"]["Number"])
        except (KeyError, TypeError) as error:
            raise ValueError(
                "run metadata and configuration omit the saved interval count"
            ) from error
    if count < 1:
        raise ValueError("saved interval count must be positive")
    return count


def validate_torch_calibration_completion(
    run: Path,
    *,
    expected_case_id: str,
    expected_resolution: int,
    expected_duration_myr: float,
    expected_saved_intervals: int,
    expected_saved_3d_states: int,
    expected_rk4_substeps: int,
    expected_checkpoint_interval: int,
    expected_time_step_factor: float = 1.0,
    expected_run_id: str | None = None,
) -> tuple[dict, dict]:
    """Validate the complete numerical contract of a Torch calibration run."""

    resolved = run.expanduser().resolve()
    try:
        summary = json.loads(
            (resolved / "torch_run_summary.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (resolved / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Torch calibration records are unreadable: {resolved}") from error
    if not expected_case_id:
        raise ValueError("expected calibration case ID is required")
    expected_positive_integers = (
        expected_resolution,
        expected_saved_intervals,
        expected_saved_3d_states,
        expected_rk4_substeps,
        expected_checkpoint_interval,
    )
    if (
        not math.isfinite(expected_duration_myr)
        or expected_duration_myr <= 0.0
        or not math.isfinite(expected_time_step_factor)
        or not 0.0 < expected_time_step_factor <= 1.0
        or any(value < 1 for value in expected_positive_integers)
    ):
        raise ValueError("expected Torch calibration contract is invalid")
    if expected_run_id is None:
        expected_run_id = f"{expected_case_id}_n{expected_resolution}"
    if not expected_run_id:
        raise ValueError("expected Torch calibration run ID is required")
    if summary.get("status") != "complete":
        raise ValueError("Torch calibration evolution is not complete")
    exact_summary = {
        "saved_intervals": expected_saved_intervals,
    }
    exact_metadata = {
        "backend": "pytorch_cuda",
        "case_id": expected_case_id,
        "run_id": expected_run_id,
        "resolution": expected_resolution,
        "save_number": expected_saved_intervals,
        "saved_3d_states": expected_saved_3d_states,
        "nbody_rk4_substeps_per_wave_step": expected_rk4_substeps,
        "checkpoint_every_saved_intervals": expected_checkpoint_interval,
        "time_step_factor": expected_time_step_factor,
        "analytic_fdm_drag": False,
        "live_wave_force_on_smbhs": True,
        "smbh_force_on_live_wave": True,
    }
    for key, expected in exact_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"Torch calibration summary has invalid {key}")
    for key, expected in exact_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Torch calibration metadata has invalid {key}")
    for record, label in ((summary, "summary"), (metadata, "metadata")):
        try:
            duration = float(record["duration_myr"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Torch calibration {label} has invalid duration_myr"
            ) from error
        if not math.isclose(
            duration, expected_duration_myr, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"Torch calibration {label} has invalid duration_myr"
            )
    try:
        summary_steps = int(summary["actual_wave_steps"])
        metadata_steps = int(metadata["actual_wave_steps"])
        elapsed = float(summary["elapsed_seconds"])
        peak_memory = int(summary["peak_device_memory_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Torch calibration completion metrics are invalid") from error
    if (
        summary_steps < 1
        or summary_steps != metadata_steps
        or not math.isfinite(elapsed)
        or elapsed <= 0.0
        or peak_memory <= 0
        or not metadata.get("device_name")
    ):
        raise ValueError("Torch calibration completion metrics are invalid")
    return summary, metadata
