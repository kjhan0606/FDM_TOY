import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fdm_smbh_delay.convergence import (
    _bootstrap_orbit_coordinate_ratio,
    _bootstrap_orbit_rates,
    load_convergence_run,
    summarize_convergence,
    validate_orbit_artifact_provenance,
)
from fdm_smbh_delay.qe_box_control import verify_fixed_comparison_summary
from fdm_smbh_delay.qe_followup_design import build_qe_followup_design
from scripts.summarize_pyul_convergence import _parse_edges


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_identity(paths: list[Path]) -> dict:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(_digest(path).encode())
        digest.update(b"\n")
    return {"count": len(paths), "ordered_name_and_content_sha256": digest.hexdigest()}


def test_orbit_artifact_provenance_binds_run_upstream_and_csv(tmp_path: Path) -> None:
    metadata = {"run_id": "qe_run_a", "case_id": "qe_case", "resolution": 256}
    (tmp_path / "fdm_adapter_metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "config.uldm").write_text('{"Duration":1}\n')
    (tmp_path / "conservation_summary.json").write_text('{"status":"diagnosed"}\n')
    (tmp_path / "conservation_timeseries.csv").write_text("time_myr\n0\n")
    (tmp_path / "orbit_averaged_exchange.csv").write_text("cycle\n0\n")
    state_dir = tmp_path / "Outputs" / "NBody"
    state_dir.mkdir(parents=True)
    state_path = state_dir / "NTM_#0.npy"
    np.save(state_path, np.zeros((2, 6)))
    summary = {"artifact_provenance": {
        "schema_version": 1,
        "run_identity": metadata.copy(),
        "upstream": {
            "fdm_adapter_metadata_sha256": _digest(tmp_path / "fdm_adapter_metadata.json"),
            "config_sha256": _digest(tmp_path / "config.uldm"),
            "conservation_summary_sha256": _digest(tmp_path / "conservation_summary.json"),
            "conservation_timeseries_sha256": _digest(tmp_path / "conservation_timeseries.csv"),
        },
        "orbit_csv_sha256": _digest(tmp_path / "orbit_averaged_exchange.csv"),
        "nbody_state_set": _state_identity([state_path]),
    }}
    assert validate_orbit_artifact_provenance(tmp_path, metadata, summary)["status"] == (
        "verified_orbit_artifact_provenance_v1"
    )
    copied_metadata = {**metadata, "run_id": "qe_run_b"}
    with pytest.raises(ValueError, match="run identity differs"):
        validate_orbit_artifact_provenance(tmp_path, copied_metadata, summary)
    (tmp_path / "fdm_adapter_metadata.json").write_text(json.dumps(copied_metadata))
    copied_binding = json.loads(json.dumps(summary))
    copied_binding["artifact_provenance"]["run_identity"] = copied_metadata
    with pytest.raises(ValueError, match="upstream digest differs"):
        validate_orbit_artifact_provenance(tmp_path, copied_metadata, copied_binding)
    (tmp_path / "fdm_adapter_metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "conservation_timeseries.csv").write_text("time_myr\n0\n1\n")
    with pytest.raises(ValueError, match="upstream digest differs"):
        validate_orbit_artifact_provenance(tmp_path, metadata, summary)
    (tmp_path / "conservation_timeseries.csv").write_text("time_myr\n0\n")
    (tmp_path / "orbit_averaged_exchange.csv").write_text("cycle\n1\n")
    with pytest.raises(ValueError, match="CSV digest differs"):
        validate_orbit_artifact_provenance(tmp_path, metadata, summary)
    (tmp_path / "orbit_averaged_exchange.csv").write_text("cycle\n0\n")
    np.save(state_dir / "NTM_#1.npy", np.full((2, 6), 2.0))
    assert validate_orbit_artifact_provenance(
        tmp_path, metadata, summary
    )["status"] == "verified_orbit_artifact_provenance_v1"
    np.save(state_path, np.ones((2, 6)))
    with pytest.raises(ValueError, match="N-body state set differs"):
        validate_orbit_artifact_provenance(tmp_path, metadata, summary)


def test_orbit_artifact_provenance_marks_unbound_archive_unverified(
    tmp_path: Path,
) -> None:
    status = validate_orbit_artifact_provenance(
        tmp_path, {"run_id": "old"}, {"status": "orbit_averaged"}
    )
    assert status["status"] == "legacy_unverified_orbit_artifacts"


def _write_run(
    path: Path, *, scale: float, time_step_factor: float,
    include_osculating_axis: bool = True,
) -> None:
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
                "initial_resolved_energy_drift_over_transfer": 0.001,
                "maximum_initial_resolved_energy_error_over_transfer": 0.001,
                "time_of_maximum_initial_resolved_energy_error_myr": 1.5,
                "final_energy_error_over_transfer": 0.001,
                "maximum_energy_error_over_transfer_tolerance": 0.01,
                "initial_resolved_energy_conservation_passed": True,
                "energy_transfer_conservation_passed": True,
                "max_total_energy_drift_over_energy_transfer": 0.001,
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
    time = np.array([0.0, 1.5, 2.0])
    separation = 1.0 - 0.1 * scale * time
    separation[-1] = 0.4
    columns = {
        "time_myr": time,
        "separation_pc": separation,
        "energy_error_over_transfer": np.array([0.0, 0.001, 0.001]),
        "binary_orbital_energy": -2.0 * scale * time,
        "binary_angular_momentum_msun_pc2_myr": -scale * time,
        "wave_intrinsic_energy": 3.0 * scale * time,
        "wave_bh_interaction_grid": -scale * time,
        "bh_com_kinetic_energy": np.zeros_like(time),
        "combined_energy": 0.003 * scale * time,
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
        "mean_eccentricity_osculating": np.full(4, 0.2),
        "orbital_power": np.full(4, -2.0 * scale),
        "orbital_torque": np.full(4, -scale),
        "wave_intrinsic_energy_rate": np.full(4, 3.0 * scale),
        "wave_bh_interaction_energy_rate": np.full(4, -scale),
        "bh_com_kinetic_energy_rate": np.zeros(4),
        "combined_energy_residual_rate": np.full(4, 0.01 * scale),
    }
    if include_osculating_axis:
        orbit_columns["mean_semimajor_axis_osculating_pc"] = np.array(
            [1.0, 0.95, 0.90, 0.85]
        )
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
        0.001
    )
    assert second[
        "maximum_initial_resolved_energy_error_over_transfer"
    ] == pytest.approx(0.001)
    assert second["initial_resolved_energy_conservation_passed"] is True
    assert second["maximum_energy_error_over_transfer"] == pytest.approx(0.001)
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
    assert matched["retained_bins"] == 1
    assert matched["selection_status"] == "matched_separation_bins_evaluated"
    assert matched["resolved_complete_orbits_by_run"] == [
        {
            "label": "first",
            "complete_orbits": 3,
            "initial_instantaneous_resolved_duration_myr": 1.5,
        },
        {
            "label": "second",
            "complete_orbits": 3,
            "initial_instantaneous_resolved_duration_myr": 1.5,
        },
    ]
    for separation_bin in matched["bins"]:
        second_at_matched_separation = separation_bin["runs"][1]
        assert second_at_matched_separation[
            "fractional_rate_difference_from_reference"
        ]["orbital_power"] == pytest.approx(-0.1)
        assert second_at_matched_separation[
            "mean_eccentricity_osculating"
        ] == pytest.approx(0.2)
        assert second_at_matched_separation[
            "minimum_orbit_mean_eccentricity"
        ] == pytest.approx(0.2)
        assert second_at_matched_separation[
            "maximum_orbit_mean_eccentricity"
        ] == pytest.approx(0.2)
        assert second_at_matched_separation[
            "minimum_orbit_mean_semimajor_axis_pc"
        ] <= second_at_matched_separation[
            "mean_semimajor_axis_osculating_pc"
        ] <= second_at_matched_separation[
            "maximum_orbit_mean_semimajor_axis_pc"
        ]
        selected = (
            (np.array([0.98, 0.93, 0.88]) >= separation_bin["lower_separation_pc"])
            & (np.array([0.98, 0.93, 0.88]) <= separation_bin["upper_separation_pc"])
        )
        assert second_at_matched_separation[
            "mean_semimajor_axis_osculating_pc"
        ] == pytest.approx(np.mean(np.array([1.0, 0.95, 0.90])[selected]))
    aggregate = matched[
        "aggregate_fractional_rate_differences_from_reference"
    ][1]["rate_differences"]["orbital_power"]
    assert aggregate["bins"] == 1
    assert aggregate[
        "median_absolute_fractional_difference"
    ] == pytest.approx(0.1)
    assert aggregate[
        "maximum_absolute_fractional_difference"
    ] == pytest.approx(0.1)


def test_loader_rejects_legacy_summary_without_resolved_peak(tmp_path: Path) -> None:
    run = tmp_path / "legacy"
    _write_run(run, scale=1.0, time_step_factor=1.0)
    path = run / "conservation_summary.json"
    summary = json.loads(path.read_text())
    del summary["maximum_initial_resolved_energy_error_over_transfer"]
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="lacks peak-resolved evidence"):
        load_convergence_run("legacy", run)


def test_loader_rejects_passing_final_with_failing_resolved_peak(
    tmp_path: Path,
) -> None:
    run = tmp_path / "false_positive"
    _write_run(run, scale=1.0, time_step_factor=1.0)
    path = run / "conservation_summary.json"
    summary = json.loads(path.read_text())
    summary["maximum_initial_resolved_energy_error_over_transfer"] = 0.012
    summary["initial_resolved_energy_drift_over_transfer"] = 0.012
    summary["final_energy_error_over_transfer"] = 0.0003
    summary["initial_resolved_energy_conservation_passed"] = True
    summary["energy_transfer_conservation_passed"] = True
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="pass flag disagrees with resolved peak"):
        load_convergence_run("false_positive", run)


def test_loader_rejects_understated_resolved_cutoff(tmp_path: Path) -> None:
    run = tmp_path / "understated"
    _write_run(run, scale=1.0, time_step_factor=1.0)
    path = run / "conservation_summary.json"
    summary = json.loads(path.read_text())
    summary["initial_spatially_resolved_duration_myr"] = 0.0
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="resolved duration disagrees"):
        load_convergence_run("understated", run)


def test_loader_recomputes_normalized_history_from_raw_energies(
    tmp_path: Path,
) -> None:
    run = tmp_path / "tampered_history"
    _write_run(run, scale=1.0, time_step_factor=1.0)
    path = run / "conservation_timeseries.csv"
    table = np.genfromtxt(path, delimiter=",", names=True)
    table["energy_error_over_transfer"][1] = 0.0
    np.savetxt(
        path,
        np.column_stack([table[name] for name in table.dtype.names or ()]),
        delimiter=",", header=",".join(table.dtype.names or ()), comments="",
    )
    with pytest.raises(ValueError, match="history disagrees with raw energies"):
        load_convergence_run("tampered_history", run)


def test_loader_rejects_one_resolved_sample(tmp_path: Path) -> None:
    run = tmp_path / "one_sample"
    _write_run(run, scale=1.0, time_step_factor=1.0)
    path = run / "conservation_timeseries.csv"
    table = np.genfromtxt(path, delimiter=",", names=True)
    table["separation_pc"][1] = 0.4
    np.savetxt(
        path,
        np.column_stack([table[name] for name in table.dtype.names or ()]),
        delimiter=",", header=",".join(table.dtype.names or ()), comments="",
    )
    with pytest.raises(ValueError, match="fewer than two states"):
        load_convergence_run("one_sample", run)


def test_matched_separation_excludes_orbits_ending_after_instantaneous_cutoff(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.0, time_step_factor=0.5)
    result = summarize_convergence(
        (
            load_convergence_run("first", first_path),
            load_convergence_run("second", second_path),
        ),
        separation_bins=1,
        minimum_orbits_per_separation_bin=4,
    )

    matched = result["matched_separation"]
    assert matched["retained_bins"] == 0
    assert matched["bins"] == []
    assert matched["selection_status"] == "matched_separation_bins_evaluated"
    assert matched["resolved_complete_orbits_by_run"][1] == {
        "label": "second",
        "complete_orbits": 3,
        "initial_instantaneous_resolved_duration_myr": 1.5,
    }


def test_qe_comparison_requires_measured_osculating_axis(tmp_path: Path) -> None:
    run = tmp_path / "qe_run"
    _write_run(
        run, scale=1.0, time_step_factor=1.0,
        include_osculating_axis=False,
    )
    metadata_path = run / "fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["case_id"] = "qe_synthetic"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mean_semimajor_axis_osculating_pc"):
        load_convergence_run("qe", run)


def test_eight_orbit_bootstrap_keeps_two_independent_blocks() -> None:
    fields = [
        ("orbital_period_myr", float),
        *(
            (field, float)
            for field in (
                "orbital_power",
                "orbital_torque",
                "wave_intrinsic_energy_rate",
                "wave_bh_interaction_energy_rate",
                "bh_com_kinetic_energy_rate",
                "combined_energy_residual_rate",
            )
        ),
    ]
    orbit = np.zeros(8, dtype=fields)
    orbit["orbital_period_myr"] = 1.0
    varying_rate = np.arange(1.0, 9.0)
    for field in orbit.dtype.names or ():
        if field != "orbital_period_myr":
            orbit[field] = varying_rate

    rates = _bootstrap_orbit_rates(orbit, np.arange(8))

    for interval in rates.values():
        assert interval["bootstrap_block_length_orbits"] == 4
        assert interval["minimum_independent_blocks"] == 2
        assert interval["lower_95"] < interval["upper_95"]


def test_coordinate_ratio_bootstrap_resamples_joint_orbit_states() -> None:
    orbit = np.zeros(
        8,
        dtype=[
            ("orbital_period_myr", float),
            ("mean_separation_pc", float),
            ("mean_semimajor_axis_osculating_pc", float),
            ("mean_eccentricity_osculating", float),
        ],
    )
    axis = np.linspace(0.4, 0.6, 8)
    orbit["orbital_period_myr"] = np.linspace(0.8, 1.2, 8)
    orbit["mean_semimajor_axis_osculating_pc"] = axis
    orbit["mean_separation_pc"] = 0.9 * axis
    orbit["mean_eccentricity_osculating"] = 0.2
    result = _bootstrap_orbit_coordinate_ratio(orbit, np.arange(8))
    expected = 0.9 / (1.0 + 0.5 * 0.2**2)
    assert result["estimate"] == pytest.approx(expected)
    assert result["lower_95"] == pytest.approx(expected)
    assert result["upper_95"] == pytest.approx(expected)
    assert result["bootstrap_block_length_orbits"] == 4
    assert result["minimum_independent_blocks"] == 2


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


def test_fixed_separation_edges_preserve_identical_physical_bins(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.0, time_step_factor=0.5)
    loaded = (
        load_convergence_run("first", first_path),
        load_convergence_run("second", second_path),
    )
    result = summarize_convergence(
        loaded, separation_bins=1, minimum_orbits_per_separation_bin=2,
        separation_bin_edges_pc=(0.88, 0.98),
    )
    matched = result["matched_separation"]
    assert result["requested_separation_bin_edges_pc"] == [0.88, 0.98]
    assert matched["bin_edge_policy"] == "fixed_physical_edges"
    assert matched["separation_bin_edges_pc"] == [0.88, 0.98]
    assert matched["retained_bins"] == 1
    assert matched["bins"][0]["lower_separation_pc"] == pytest.approx(0.88)
    assert matched["bins"][0]["upper_separation_pc"] == pytest.approx(0.98)


def test_fixed_bin_outside_full_common_support_is_censored(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.0, time_step_factor=0.5)
    result = summarize_convergence(
        (load_convergence_run("first", first_path),
         load_convergence_run("second", second_path)),
        separation_bins=1, minimum_orbits_per_separation_bin=2,
        separation_bin_edges_pc=(0.87, 0.98),
    )
    assert result["matched_separation"]["retained_bins"] == 0
    assert result["matched_separation"]["bins"] == []


def test_fixed_edge_cli_parser_rejects_non_numeric_edges() -> None:
    assert _parse_edges("0.1,0.2") == (0.1, 0.2)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_edges("0.1,bad")


def test_fixed_comparison_is_recomputed_from_raw_diagnostics(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.0, time_step_factor=0.5)
    for run in (first_path, second_path):
        (run / "wave_response_timeseries.csv").write_text(
            "time_myr,measured_half_density_radius_pc\n0,1\n1,1\n"
        )
        (run / "torch_run_summary.json").write_text(
            json.dumps({"status": "complete", "run": str(run)})
        )
        (run / "wave_response_summary.json").write_text(
            json.dumps({"status": "diagnosed", "run": str(run)})
        )
    summary = summarize_convergence(
        (load_convergence_run("first", first_path),
         load_convergence_run("second", second_path)),
        separation_bins=1, minimum_orbits_per_separation_bin=2,
        separation_bin_edges_pc=(0.88, 0.98),
    )
    path = tmp_path / "fixed.json"
    path.write_text(json.dumps(summary))
    verified = verify_fixed_comparison_summary(path)
    assert len(verified["raw_inputs"]) == 2
    assert len(verified["comparison_sha256"]) == 64
    config_path = first_path / "config.uldm"
    config = json.loads(config_path.read_text())
    config["Temporal Step Factor"] = 1.1
    config_path.write_text(json.dumps(config))
    # The metadata field takes precedence, so alter that too.
    metadata_path = first_path / "fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["time_step_factor"] = 1.1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not reproducible from raw diagnostics"):
        verify_fixed_comparison_summary(path)


def test_registered_fixed_comparison_recomputes_and_rechecks_design(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases.csv"
    cases.write_text(
        "case_id,mass_ratio_q,eccentricity,initial_separation_pc,core_radius_pc,"
        "kepler_period_myr\nqe_test,1.0,0.0,0.9,2.2,0.001\n"
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,run_id,effective_grid_cells,box_size_pc,finest_cell_size_pc,"
        "plummer_radius_pc\n"
        "qe_test,qe_test_n128,128,32,0.25,0.125\n"
        "qe_test,qe_test_n256,256,32,0.125,0.0625\n"
    )
    design = build_qe_followup_design(
        case_id="qe_test", physical_cases=cases, run_manifest=manifest,
        coarse_resolution=128, fine_resolution=256,
        separation_bin_edges_pc=(0.88, 0.98), duration_myr=0.02,
    )
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design))
    design_file_sha256 = hashlib.sha256(design_path.read_bytes()).hexdigest()
    runs = []
    for role, resolution, cell_size in (
        ("fine", 256, 0.125), ("coarse", 128, 0.25)
    ):
        run = tmp_path / role
        _write_run(run, scale=1.0, time_step_factor=1.0)
        metadata_path = run / "fdm_adapter_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update({
            "case_id": "qe_test", "resolution": resolution,
            "cell_size_pc": cell_size, "box_size_pc": 32.0,
            "duration_myr": 0.02,
            "qe_design_binding": {
                "status": "qe_prospective_design_bound_not_a_calibration_release",
                "design_sha256": design["design_sha256"],
                "file_sha256": design_file_sha256, "role": role,
            },
        })
        metadata_path.write_text(json.dumps(metadata))
        if role == "fine":
            conservation_path = run / "conservation_summary.json"
            conservation = json.loads(conservation_path.read_text())
            conservation["initial_spatially_resolved_duration_myr"] = 2.0
            conservation_path.write_text(json.dumps(conservation))
        (run / "wave_response_timeseries.csv").write_text(
            "time_myr,measured_half_density_radius_pc\n0,1\n1,1\n"
        )
        (run / "torch_run_summary.json").write_text(
            json.dumps({"status": "complete", "run": str(run)})
        )
        (run / "wave_response_summary.json").write_text(
            json.dumps({"status": "diagnosed", "run": str(run)})
        )
        runs.append(load_convergence_run(f"n{resolution}", run))
    summary = summarize_convergence(
        runs, separation_bins=1, minimum_orbits_per_separation_bin=8,
        separation_bin_edges_pc=(0.88, 0.98),
    )
    binding = {
        "path": str(design_path), "physical_cases_path": str(cases),
        "run_manifest_path": str(manifest),
        "design_sha256": design["design_sha256"],
        "file_sha256": design_file_sha256,
        "comparison_kind": "resolution_pair",
        "status": "registered_design_bound_not_a_calibration_release",
    }
    summary["qe_design_binding"] = binding
    path = tmp_path / "registered-comparison.json"
    path.write_text(json.dumps(summary))
    assert len(verify_fixed_comparison_summary(path)["raw_inputs"]) == 2
    summary["qe_design_binding"]["design_sha256"] = "0" * 64
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="differs from its fixed design"):
        verify_fixed_comparison_summary(path)


@pytest.mark.parametrize("edges", [(0.98, 0.88), (0.88, 0.88),
                                     (0.88, float("nan")), (0.88, 0.98, 1.0)])
def test_fixed_separation_edges_reject_invalid_design(
    tmp_path: Path, edges: tuple[float, ...],
) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_run(first_path, scale=1.0, time_step_factor=1.0)
    _write_run(second_path, scale=1.0, time_step_factor=0.5)
    with pytest.raises(ValueError, match="fixed separation-bin edges"):
        summarize_convergence(
            (load_convergence_run("first", first_path),
             load_convergence_run("second", second_path)),
            separation_bins=1, minimum_orbits_per_separation_bin=2,
            separation_bin_edges_pc=edges,
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
    with pytest.raises(ValueError, match="history disagrees with raw energies"):
        load_convergence_run("second", second_path)
