import json
from pathlib import Path

import numpy as np
import pytest

from fdm_smbh_delay.convergence import load_convergence_run, summarize_convergence


def _write_run(path: Path, *, scale: float, time_step_factor: float) -> None:
    path.mkdir()
    (path / "fdm_adapter_metadata.json").write_text(
        json.dumps(
            {
                "resolution": 128,
                "cell_size_pc": 0.25,
                "time_step_factor": time_step_factor,
                "nbody_rk4_substeps_per_wave_step": 9,
            }
        )
    )
    (path / "config.uldm").write_text(
        json.dumps({"Temporal Step Factor": time_step_factor, "RK Steps": 36})
    )
    (path / "conservation_summary.json").write_text(
        json.dumps(
            {
                "initial_spatially_resolved_duration_myr": 1.5,
                "max_total_energy_drift_over_energy_transfer": 0.002,
            }
        )
    )
    (path / "orbit_averaged_exchange_summary.json").write_text(
        json.dumps(
            {
                "initial_resolved_window_block_bootstrap": {
                    "window_orbits": 4,
                    "orbital_power": {"estimate": -2.0 * scale},
                    "orbital_torque": {"estimate": -1.0 * scale},
                }
            }
        )
    )
    time = np.array([0.0, 1.0, 2.0])
    columns = {
        "time_myr": time,
        "separation_pc": 1.0 - 0.1 * scale * time,
        "energy_error_over_transfer": 0.001 * time,
        "binary_orbital_energy": -2.0 * scale * time,
        "binary_angular_momentum_msun_pc2_myr": -scale * time,
        "wave_intrinsic_energy": 3.0 * scale * time,
        "wave_bh_interaction_grid": -scale * time,
        "bh_com_kinetic_energy": np.zeros_like(time),
        "combined_energy": 0.01 * scale * time,
    }
    np.savetxt(
        path / "conservation_timeseries.csv",
        np.column_stack(tuple(columns.values())),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )


def test_common_interval_comparison_uses_resolved_duration(tmp_path: Path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.1, time_step_factor=0.5)
    result = summarize_convergence(
        (
            load_convergence_run("first", first_path),
            load_convergence_run("second", second_path),
        )
    )
    assert result["common_interval_end_myr"] == pytest.approx(1.5)
    second = result["runs"][1]
    assert second["common_interval"]["mean_binary_orbital_energy_rate"] == pytest.approx(-2.2)
    assert second["difference_from_reference"][
        "mean_binary_orbital_energy_rate_fractional_difference"
    ] == pytest.approx(-0.1)
    assert second["difference_from_reference"][
        "separation_difference_over_reference_initial"
    ] == pytest.approx(-0.015)


def test_common_interval_requires_unique_labels(tmp_path: Path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.0, time_step_factor=0.5)
    with pytest.raises(ValueError, match="unique"):
        summarize_convergence(
            (
                load_convergence_run("same", first_path),
                load_convergence_run("same", second_path),
            )
        )
