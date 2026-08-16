from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdm_smbh_delay.run_metadata import (
    saved_interval_count,
    validate_torch_calibration_completion,
)


def test_saved_interval_count_prefers_metadata() -> None:
    assert saved_interval_count(
        {"save_number": 512}, {"Save Options": {"Number": 32}}
    ) == 512


def test_saved_interval_count_supports_legacy_config() -> None:
    assert saved_interval_count({}, {"Save Options": {"Number": 32}}) == 32
    with pytest.raises(ValueError, match="omit"):
        saved_interval_count({}, {})
    with pytest.raises(ValueError, match="positive"):
        saved_interval_count({"save_number": 0}, {})


def _write_complete_torch_run(path: Path) -> None:
    path.mkdir()
    (path / "torch_run_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "duration_myr": 0.8,
                "saved_intervals": 2048,
                "actual_wave_steps": 73728,
                "elapsed_seconds": 2800.0,
                "peak_device_memory_bytes": 16_000_000_000,
            }
        )
    )
    (path / "fdm_adapter_metadata.json").write_text(
        json.dumps(
            {
                "backend": "pytorch_cuda",
                "case_id": "boey_each02pct",
                "run_id": "boey_each02pct_n384",
                "resolution": 384,
                "duration_myr": 0.8,
                "save_number": 2048,
                "saved_3d_states": 17,
                "nbody_rk4_substeps_per_wave_step": 9,
                "checkpoint_every_saved_intervals": 32,
                "time_step_factor": 1.0,
                "analytic_fdm_drag": False,
                "live_wave_force_on_smbhs": True,
                "smbh_force_on_live_wave": True,
                "actual_wave_steps": 73728,
                "device_name": "NVIDIA A100-SXM4-80GB",
            }
        )
    )


def _validate_complete_torch_run(path: Path) -> tuple[dict, dict]:
    return validate_torch_calibration_completion(
        path,
        expected_case_id="boey_each02pct",
        expected_resolution=384,
        expected_duration_myr=0.8,
        expected_saved_intervals=2048,
        expected_saved_3d_states=17,
        expected_rk4_substeps=9,
        expected_checkpoint_interval=32,
    )


def test_complete_torch_calibration_contract(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_complete_torch_run(run)
    summary, metadata = _validate_complete_torch_run(run)
    assert summary["actual_wave_steps"] == metadata["actual_wave_steps"]


@pytest.mark.parametrize(
    "record_name,key,value,match",
    [
        ("torch_run_summary.json", "saved_intervals", 2047, "saved_intervals"),
        ("fdm_adapter_metadata.json", "duration_myr", 0.7, "duration_myr"),
        ("fdm_adapter_metadata.json", "saved_3d_states", 16, "saved_3d_states"),
        ("fdm_adapter_metadata.json", "analytic_fdm_drag", True, "analytic_fdm_drag"),
        (
            "fdm_adapter_metadata.json",
            "live_wave_force_on_smbhs",
            False,
            "live_wave_force_on_smbhs",
        ),
    ],
)
def test_torch_calibration_contract_rejects_drift(
    tmp_path: Path,
    record_name: str,
    key: str,
    value: object,
    match: str,
) -> None:
    run = tmp_path / "run"
    _write_complete_torch_run(run)
    record_path = run / record_name
    record = json.loads(record_path.read_text())
    record[key] = value
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=match):
        _validate_complete_torch_run(run)
