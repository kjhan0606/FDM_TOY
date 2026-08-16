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
                "initial_resolved_energy_drift_over_transfer": 0.002,
                "initial_resolved_energy_conservation_passed": True,
                "max_total_energy_drift_over_energy_transfer": 0.02,
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
    orbit_columns = {
        "start_time_myr": np.array([0.0, 0.5, 1.0, 1.5]),
        "end_time_myr": np.array([0.5, 1.0, 1.5, 2.0]),
        "orbital_period_myr": np.full(4, 0.5),
        "mean_separation_pc": np.array([0.98, 0.93, 0.88, 0.83]),
        "mean_separation_over_cell_size": np.array([3.92, 3.72, 3.52, 3.32]),
        "orbital_power": np.full(4, -2.0 * scale),
        "orbital_torque": np.full(4, -scale),
        "wave_intrinsic_energy_rate": np.full(4, 3.0 * scale),
        "wave_bh_interaction_energy_rate": np.full(4, -scale),
        "bh_com_kinetic_energy_rate": np.zeros(4),
        "combined_energy_residual_rate": np.full(4, 0.01 * scale),
    }
    np.savetxt(
        path / "orbit_averaged_exchange.csv",
        np.column_stack(tuple(orbit_columns.values())),
        delimiter=",",
        header=",".join(orbit_columns),
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
        ),
        separation_bins=2,
        minimum_orbits_per_separation_bin=2,
    )
    assert result["common_interval_start_myr"] == pytest.approx(0.0)
    assert result["common_interval_end_myr"] == pytest.approx(1.5)
    second = result["runs"][1]
    assert second["initial_resolved_energy_drift_over_transfer"] == pytest.approx(
        0.002
    )
    assert second["initial_resolved_energy_conservation_passed"] is True
    assert second["maximum_energy_error_over_transfer"] == pytest.approx(0.02)
    assert second["common_interval"]["mean_binary_orbital_energy_rate"] == pytest.approx(-2.2)
    assert second["difference_from_reference"][
        "mean_binary_orbital_energy_rate_fractional_difference"
    ] == pytest.approx(-0.1)
    assert second["difference_from_reference"][
        "separation_difference_over_reference_initial"
    ] == pytest.approx(-0.015)
    assert result["common_orbit_window_start_myr"] == pytest.approx(0.0)
    assert second["common_orbit_window"]["rates"]["orbital_power"][
        "estimate"
    ] == pytest.approx(-2.2)
    assert second["common_orbit_window"][
        "fractional_rate_difference_from_reference"
    ]["orbital_power"] == pytest.approx(-0.1)
    assert second["common_orbit_window"]["rates"]["wave_total_energy_rate"][
        "estimate"
    ] == pytest.approx(2.2)
    matched = result["matched_separation"]
    assert matched["requested_bins"] == 2
    assert matched["minimum_complete_orbits_per_run_per_bin"] == 2
    assert matched["retained_bins"] == 2
    for separation_bin in matched["bins"]:
        second_at_matched_separation = separation_bin["runs"][1]
        assert second_at_matched_separation[
            "fractional_rate_difference_from_reference"
        ]["orbital_power"] == pytest.approx(-0.1)
    aggregate = matched[
        "aggregate_fractional_rate_differences_from_reference"
    ][1]["rate_differences"]["orbital_power"]
    assert aggregate["bins"] == 2
    assert aggregate[
        "median_absolute_fractional_difference"
    ] == pytest.approx(0.1)
    assert aggregate[
        "maximum_absolute_fractional_difference"
    ] == pytest.approx(0.1)


def test_matched_separation_requires_positive_bin_count(tmp_path: Path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.0, time_step_factor=0.5)
    with pytest.raises(ValueError, match="bin count"):
        summarize_convergence(
            (
                load_convergence_run("first", first_path),
                load_convergence_run("second", second_path),
            ),
            separation_bins=0,
            minimum_orbits_per_separation_bin=2,
        )


def test_matched_separation_requires_two_orbits_per_bin(tmp_path: Path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.0, time_step_factor=0.5)
    with pytest.raises(ValueError, match="at least two"):
        summarize_convergence(
            (
                load_convergence_run("first", first_path),
                load_convergence_run("second", second_path),
            ),
            separation_bins=2,
            minimum_orbits_per_separation_bin=1,
        )


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


def test_metadata_values_do_not_require_legacy_config_keys(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.0, time_step_factor=0.5)
    for path in (first_path, second_path):
        (path / "config.uldm").write_text("{}")

    result = summarize_convergence(
        (
            load_convergence_run("first", first_path),
            load_convergence_run("second", second_path),
        )
    )

    assert result["runs"][0]["time_step_factor"] == pytest.approx(1.0)
    assert result["runs"][1]["nbody_rk4_substeps_per_wave_step"] == 9


def test_load_requires_energy_error_column(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, scale=1.0, time_step_factor=1.0)
    table = np.genfromtxt(
        run / "conservation_timeseries.csv", delimiter=",", names=True
    )
    names = [
        name for name in table.dtype.names or ()
        if name != "energy_error_over_transfer"
    ]
    np.savetxt(
        run / "conservation_timeseries.csv",
        np.column_stack([table[name] for name in names]),
        delimiter=",",
        header=",".join(names),
        comments="",
    )

    with pytest.raises(ValueError, match="energy_error_over_transfer"):
        load_convergence_run("run", run)


def test_common_orbit_window_reports_actual_cycle_coverage(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.0, time_step_factor=0.5)
    orbit_path = second_path / "orbit_averaged_exchange.csv"
    orbit = np.genfromtxt(orbit_path, delimiter=",", names=True)
    orbit["start_time_myr"] += 0.1
    orbit["end_time_myr"] += 0.1
    np.savetxt(
        orbit_path,
        np.column_stack([orbit[name] for name in orbit.dtype.names or ()]),
        delimiter=",",
        header=",".join(orbit.dtype.names or ()),
        comments="",
    )

    result = summarize_convergence(
        (
            load_convergence_run("first", first_path),
            load_convergence_run("second", second_path),
        )
    )

    first, second = result["runs"]
    assert result["common_orbit_window_start_myr"] == pytest.approx(0.1)
    assert first["common_orbit_window"]["start_time_myr"] == pytest.approx(0.5)
    assert first["common_orbit_window"]["end_time_myr"] == pytest.approx(1.5)
    assert second["common_orbit_window"]["start_time_myr"] == pytest.approx(0.1)
    assert second["common_orbit_window"]["end_time_myr"] == pytest.approx(1.1)


def test_common_interval_uses_latest_input_start_time(tmp_path: Path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.1, time_step_factor=0.5)
    table_path = second_path / "conservation_timeseries.csv"
    table = np.genfromtxt(table_path, delimiter=",", names=True)
    np.savetxt(
        table_path,
        np.column_stack([table[name][1:] for name in table.dtype.names or ()]),
        delimiter=",",
        header=",".join(table.dtype.names or ()),
        comments="",
    )

    result = summarize_convergence(
        (
            load_convergence_run("first", first_path),
            load_convergence_run("second", second_path),
        )
    )

    assert result["common_interval_start_myr"] == pytest.approx(1.0)
    assert result["common_interval_end_myr"] == pytest.approx(1.5)
    assert result["runs"][0]["common_interval"][
        "mean_binary_orbital_energy_rate"
    ] == pytest.approx(-2.0)
    assert result["runs"][1]["common_interval"][
        "mean_binary_orbital_energy_rate"
    ] == pytest.approx(-2.2)
