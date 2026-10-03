from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fdm_smbh_delay.exchange_scaling import exchange_scales
from fdm_smbh_delay.subgrid_calibration import (
    SubgridCalibrationTable,
    advance_calibrated_exchange,
    physical_subgrid_rates,
)
from fdm_smbh_delay.subgrid_table_builder import (
    CalibrationSource,
    build_source_rows,
    write_calibration_table,
)
from fdm_smbh_delay.qe_box_control import (
    assess_qe_box_control as _assess_qe_box_control,
)
from fdm_smbh_delay import qe_box_control as qe_box_module


def assess_qe_box_control(
    resolution_pair: CalibrationSource, doubled_box: CalibrationSource,
) -> dict:
    """Synthetic comparison fixtures lack raw trajectory diagnostics."""

    return _assess_qe_box_control(
        resolution_pair, doubled_box, verify_raw=False
    )


def _write_run(
    path: Path,
    *,
    resolution: int,
    half_density_radius_pc: float,
) -> None:
    path.mkdir()
    (path / "fdm_adapter_metadata.json").write_text(
        json.dumps(
            {
                "case_id": "boey_each02pct",
                "particle_mass_ev": 1.0e-21,
                "core_radius_reference_pc": 2.0,
                "box_size_pc": 40.0,
                "resolution": resolution,
                "plummer_radius_pc": 0.1,
                "mass_ratio_q": 1.0,
                "initial_eccentricity": 0.0,
            }
        )
    )
    (path / "config.uldm").write_text(
        json.dumps(
            {
                "Matter Particles": {
                    "Condition": [[2.0e7], [2.0e7]]
                },
                "ULDM Solitons": {"Condition": [[1.0e9]]},
            }
        )
    )
    np.savetxt(
        path / "wave_response_timeseries.csv",
        np.column_stack(
            (
                np.asarray([0.0, 0.5, 1.0]),
                np.full(3, half_density_radius_pc),
            )
        ),
        delimiter=",",
        header="time_myr,measured_half_density_radius_pc",
        comments="",
    )


def _matched_bin(
    index: int,
    *,
    lower: float,
    upper: float,
    difference: float,
    scales,
) -> dict:
    axis_mean = 0.52 * (lower + upper)
    reference_rates = {
        "orbital_power": {"estimate": -scales.orbital_power_msun_pc2_myr3},
        "orbital_torque": {
            "estimate": -2.0 * scales.orbital_torque_msun_pc2_myr2
        },
        "wave_total_energy_rate": {
            "estimate": scales.orbital_power_msun_pc2_myr3
        },
    }
    comparison_rates = {
        field: {"estimate": values["estimate"] * (1.0 + difference)}
        for field, values in reference_rates.items()
    }
    return {
        "bin": index,
        "lower_separation_pc": lower,
        "upper_separation_pc": upper,
        "runs": [
            {
                "label": "n512",
                "complete_orbits": 20,
                "mean_separation_pc": 0.5 * (lower + upper),
                "mean_eccentricity_osculating": 0.23,
                "minimum_orbit_mean_eccentricity": 0.22,
                "maximum_orbit_mean_eccentricity": 0.24,
                "mean_semimajor_axis_osculating_pc": axis_mean,
                "minimum_orbit_mean_semimajor_axis_pc": axis_mean * 0.98,
                "maximum_orbit_mean_semimajor_axis_pc": axis_mean * 1.02,
                "minimum_time_myr": 0.0,
                "maximum_time_myr": 0.5,
                "rates": reference_rates,
                "fractional_rate_difference_from_reference": {
                    field: 0.0 for field in reference_rates
                },
            },
            {
                "label": "n384",
                "complete_orbits": 18,
                "mean_separation_pc": 0.5 * (lower + upper),
                "mean_eccentricity_osculating": 0.23,
                "minimum_orbit_mean_eccentricity": 0.22,
                "maximum_orbit_mean_eccentricity": 0.24,
                "mean_semimajor_axis_osculating_pc": axis_mean,
                "minimum_orbit_mean_semimajor_axis_pc": axis_mean * 0.98,
                "maximum_orbit_mean_semimajor_axis_pc": axis_mean * 1.02,
                "minimum_time_myr": 0.0,
                "maximum_time_myr": 0.5,
                "rates": comparison_rates,
                "fractional_rate_difference_from_reference": {
                    field: difference for field in reference_rates
                },
            },
        ],
    }


def _write_summary(tmp_path: Path) -> Path:
    n512 = tmp_path / "n512"
    n384 = tmp_path / "n384"
    _write_run(n512, resolution=512, half_density_radius_pc=1.0)
    _write_run(n384, resolution=384, half_density_radius_pc=1.0)
    scales = exchange_scales(
        mass1_msun=2.0e7,
        mass2_msun=2.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=2.0,
    )
    path = tmp_path / "convergence.json"
    path.write_text(
        json.dumps(
            {
                "status": "common_resolved_interval_compared",
                "reference_label": "n512",
                "runs": [
                    {
                        "label": "n512",
                        "run": str(n512),
                        "initial_resolved_energy_drift_over_transfer": 0.002,
                        "maximum_energy_error_over_transfer": 0.2,
                    },
                    {
                        "label": "n384",
                        "run": str(n384),
                        "initial_resolved_energy_drift_over_transfer": 0.003,
                        "maximum_energy_error_over_transfer": 0.3,
                    },
                ],
                "matched_separation": {
                    "minimum_complete_orbits_per_run_per_bin": 8,
                    "bins": [
                        _matched_bin(
                            0,
                            lower=0.4,
                            upper=0.8,
                            difference=0.10,
                            scales=scales,
                        ),
                        _matched_bin(
                            1,
                            lower=0.8,
                            upper=1.2,
                            difference=0.25,
                            scales=scales,
                        ),
                    ],
                },
            }
        )
    )
    return path


def test_builder_accepts_only_bins_below_the_systematic_limit(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    result = build_source_rows(
        CalibrationSource("boey2025", path),
        maximum_spatial_systematic_fraction=0.20,
    )
    assert len(result.accepted_rows) == 1
    assert len(result.rejected_bins) == 1
    row = result.accepted_rows[0]
    assert row.profile_id == "boey2025"
    assert row.binary_to_soliton_mass == pytest.approx(0.04)
    assert row.dimensionless_orbital_power == pytest.approx(-1.0)
    assert row.dimensionless_orbital_torque == pytest.approx(-2.0)
    assert row.dimensionless_wave_total_energy_rate == pytest.approx(1.0)
    assert row.orbital_power_spatial_systematic_fraction == pytest.approx(0.10)
    assert row.absolute_mean_eccentricity_mismatch == pytest.approx(0.0)
    assert row.reference_minimum_half_density_radius_over_cell_size > 2.0
    assert row.comparison_minimum_half_density_radius_over_cell_size > 2.0
    assert "orbital_power exceeds the spatial systematic limit" in (
        result.rejected_bins[0]["reasons"]
    )


def test_builder_rejects_an_underresolved_core(tmp_path: Path) -> None:
    path = _write_summary(tmp_path)
    n384 = tmp_path / "n384"
    np.savetxt(
        n384 / "wave_response_timeseries.csv",
        np.column_stack((np.asarray([0.0, 0.5, 1.0]), np.full(3, 0.1))),
        delimiter=",",
        header="time_myr,measured_half_density_radius_pc",
        comments="",
    )
    result = build_source_rows(CalibrationSource("boey2025", path))
    assert not result.accepted_rows
    assert all(
        "comparison half-density radius is underresolved" in row["reasons"]
        for row in result.rejected_bins
    )


def test_builder_rejects_initial_resolved_hamiltonian_error(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    summary = json.loads(path.read_text())
    summary["runs"][1][
        "initial_resolved_energy_drift_over_transfer"
    ] = 0.011
    path.write_text(json.dumps(summary))

    result = build_source_rows(CalibrationSource("boey2025", path))

    assert not result.accepted_rows
    assert all(
        "n384 exceeds the initial-resolved Hamiltonian error limit"
        in row["reasons"]
        for row in result.rejected_bins
    )


def test_builder_requires_initial_resolved_hamiltonian_error(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    summary = json.loads(path.read_text())
    del summary["runs"][1]["initial_resolved_energy_drift_over_transfer"]
    path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="initial-resolved Hamiltonian error"):
        build_source_rows(CalibrationSource("boey2025", path))


def test_builder_requires_wave_response_for_both_resolutions(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    (tmp_path / "n384" / "wave_response_timeseries.csv").unlink()
    with pytest.raises(FileNotFoundError, match="requires sparse wave response"):
        build_source_rows(CalibrationSource("boey2025", path))


def test_builder_rejects_relaxed_production_gates(tmp_path: Path) -> None:
    path = _write_summary(tmp_path)
    with pytest.raises(ValueError, match="acceptance limits are invalid"):
        build_source_rows(
            CalibrationSource("boey2025", path),
            maximum_spatial_systematic_fraction=0.21,
        )

    with pytest.raises(ValueError, match="acceptance limits are invalid"):
        build_source_rows(
            CalibrationSource("boey2025", path),
            maximum_eccentricity_mismatch=0.021,
        )


def test_builder_rejects_mismatched_mean_eccentricity(tmp_path: Path) -> None:
    path = _write_summary(tmp_path)
    summary = json.loads(path.read_text())
    for separation_bin in summary["matched_separation"]["bins"]:
        separation_bin["runs"][1]["mean_eccentricity_osculating"] = 0.251
    path.write_text(json.dumps(summary))

    result = build_source_rows(CalibrationSource("boey2025", path))

    assert not result.accepted_rows
    assert all(
        "mean eccentricity mismatch exceeds the acceptance limit"
        in row["reasons"]
        for row in result.rejected_bins
    )
    assert all(
        row["absolute_mean_eccentricity_mismatch"] == pytest.approx(0.021)
        for row in result.rejected_bins
    )


def test_builder_rejects_a_softened_binary_separation(tmp_path: Path) -> None:
    path = _write_summary(tmp_path)
    metadata_path = tmp_path / "n384" / "fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["plummer_radius_pc"] = 0.25
    metadata_path.write_text(json.dumps(metadata))
    result = build_source_rows(CalibrationSource("boey2025", path))
    assert not result.accepted_rows
    assert any(
        "comparison binary separation is softened" in row["reasons"]
        for row in result.rejected_bins
    )


def test_writer_is_loadable_by_the_runtime_table(tmp_path: Path) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    summary = write_calibration_table(
        [CalibrationSource("boey2025", path)], output=output
    )
    assert summary["rows"] == 1
    assert summary["schema_version"] == 4
    assert len(summary["release_input_sha256"]) == 64
    assert summary["table"]["release_input_sha256"] == summary[
        "release_input_sha256"
    ]
    assert summary["interpolation"] == {
        "profile_axis": "discrete_no_cross_profile_interpolation",
        "mass_ratio_axis": "piecewise_linear_with_complete_e_mass_separation_support",
        "eccentricity_axis": "piecewise_linear_with_complete_mass_separation_support",
        "mass_axis": "piecewise_linear_binary_to_soliton_mass_within_q_e_plane",
        "separation_axis": "piecewise_linear_reference_bin_centres_within_q_e_plane",
        "outer_half_bins": "nearest_accepted_bin_value",
        "missing_separation_bins": "crossing_prohibited",
        "spatial_systematics": "maximum_of_all_bracketing_rows",
        "extrapolation": "prohibited",
    }
    assert summary["sources"][0]["accepted_bins"] == 1
    assert summary["acceptance"][
        "minimum_separation_over_plummer_radius"
    ] == pytest.approx(2.0)
    assert summary["acceptance"][
        "maximum_eccentricity_mismatch"
    ] == pytest.approx(0.02)
    assert len(summary["sources"][0]["inputs"]) == 7
    assert all(
        len(record["sha256"]) == 64
        for record in summary["sources"][0]["inputs"]
    )
    assert summary["calibrated_domains"] == [
        {
            "profile_id": "boey2025",
            "schrodinger_poisson_similarity_parameter": (
                summary["calibrated_domains"][0][
                    "schrodinger_poisson_similarity_parameter"
                ]
            ),
            "binary_to_soliton_mass": 0.04,
            "mass_ratio_q": 1.0,
            "reference_eccentricity": 0.23,
            "source_case_ids": ["boey_each02pct"],
            "accepted_separation_bin_indices": [0],
            "minimum_separation_over_core_radius": 0.2,
            "maximum_separation_over_core_radius": 0.4,
            "maximum_spatial_systematic_fraction": {
                "orbital_power": 0.10,
                "orbital_torque": 0.10,
                "wave_total_energy_rate": 0.10,
            },
            "maximum_absolute_mean_eccentricity_mismatch": 0.0,
        }
    ]
    assert len(summary["table"]["sha256"]) == 64
    assert output.with_suffix(".summary.json").is_file()
    loaded = SubgridCalibrationTable.from_release(output)
    assert len(loaded.rows) == 1
    assert loaded.rows[0].mass_ratio_q == pytest.approx(1.0)
    assert loaded.rows[0].reference_eccentricity == pytest.approx(0.23)


def test_qe_candidate_cannot_be_released_without_doubled_box_control(
    tmp_path: Path,
) -> None:
    path, _ = _write_qe_box_pair(tmp_path)
    output = tmp_path / "subgrid.csv"
    with pytest.raises(ValueError, match="verified doubled-box control"):
        write_calibration_table([CalibrationSource("test", path)], output=output)
    assert not output.exists()
    assert not output.with_suffix(".summary.json").exists()


def _write_qe_box_pair(tmp_path: Path) -> tuple[Path, Path]:
    pair_path = _write_summary(tmp_path)
    pair = json.loads(pair_path.read_text())
    for name in ("n512", "n384"):
        metadata_path = tmp_path / name / "fdm_adapter_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["case_id"] = "qe_test"
        metadata_path.write_text(json.dumps(metadata))
        config_path = tmp_path / name / "config.uldm"
        config = json.loads(config_path.read_text())
        config["Temporal Step Factor"] = 1.0
        config["RK Steps"] = 36
        config["Matter Particles"]["Plummer Radius"] = 0.1
        config_path.write_text(json.dumps(config))
    edges = [0.4, 0.8, 1.2]
    pair["requested_separation_bin_edges_pc"] = edges
    pair["matched_separation"]["bin_edge_policy"] = "fixed_physical_edges"
    pair["matched_separation"]["separation_bin_edges_pc"] = edges
    pair["matched_separation"]["requested_bins"] = 2
    pair["matched_separation"]["retained_bins"] = 2
    for row in pair["runs"]:
        row["time_step_factor"] = 1.0
        row["nbody_rk4_substeps_per_wave_step"] = 9
        metadata = json.loads((Path(row["run"]) / "fdm_adapter_metadata.json").read_text())
        row["resolution"] = metadata["resolution"]
        row["cell_size_pc"] = metadata["box_size_pc"] / metadata["resolution"]
    pair_path.write_text(json.dumps(pair))
    large = tmp_path / "n1024"
    _write_run(large, resolution=1024, half_density_radius_pc=1.0)
    metadata_path = large / "fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["case_id"] = "qe_test"
    metadata["box_size_pc"] = 80.0
    metadata_path.write_text(json.dumps(metadata))
    config_path = large / "config.uldm"
    config = json.loads(config_path.read_text())
    config["Temporal Step Factor"] = 1.0
    config["RK Steps"] = 36
    config["Matter Particles"]["Plummer Radius"] = 0.1
    config_path.write_text(json.dumps(config))
    box = deepcopy(pair)
    box["reference_label"] = "n1024"
    box["runs"] = [deepcopy(pair["runs"][0]), deepcopy(pair["runs"][0])]
    box["runs"][0]["label"] = "n1024"
    box["runs"][0]["run"] = str(large)
    box["runs"][0]["resolution"] = 1024
    box["runs"][0]["cell_size_pc"] = 80.0 / 1024
    box["runs"][1]["label"] = "n512"
    for bin_row in box["matched_separation"]["bins"]:
        shared = deepcopy(bin_row["runs"][0])
        bin_row["runs"] = [deepcopy(shared), deepcopy(shared)]
        bin_row["runs"][0]["label"] = "n1024"
        bin_row["runs"][1]["label"] = "n512"
    box_path = tmp_path / "box_control.json"
    box_path.write_text(json.dumps(box))
    for name in ("n384", "n512", "n1024"):
        metadata_path = tmp_path / name / "fdm_adapter_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["semi_major_axis_pc"] = 0.6
        metadata["initial_separation_pc"] = 0.6
        metadata_path.write_text(json.dumps(metadata))
        config_path = tmp_path / name / "config.uldm"
        config = json.loads(config_path.read_text())
        config["Matter Particles"].update({
            "Mass Units": "solar_masses", "Position Units": "pc",
            "Velocity Units": "km/s",
            "Condition": [
                [2.0e7, [0.3, 0.0, 0.0], [0.0, 260.0, 0.0]],
                [2.0e7, [-0.3, 0.0, 0.0], [0.0, -260.0, 0.0]],
            ],
        })
        config["ULDM Solitons"].update({
            "Mass Units": "solar_masses", "Position Units": "pc",
            "Velocity Units": "km/s",
            "Condition": [[1.0e9, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0]],
        })
        config_path.write_text(json.dumps(config))
    return pair_path, box_path


def test_qe_box_control_compares_same_fixed_bin_without_releasing(
    tmp_path: Path,
) -> None:
    pair, box = _write_qe_box_pair(tmp_path)
    result = assess_qe_box_control(
        CalibrationSource("test", pair), CalibrationSource("test", box)
    )
    assert result["status"] == "box_control_candidates_not_released"
    assert result["candidate_bins"] == [0]
    assert result["box_controlled_candidate_bins"] == [0]
    assert result["production_calibration_row_admitted"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("kinetic_phase_layout", "separable_axis_v1"),
     ("wave_buffer_lifetime", "release_previous_state_before_fft_v1"),
     ("wave_density_layout", "real_imag_addcmul_v1"),
     ("compact_potential_layout", "x_slab32_inplace_rsqrt_v1"),
     ("potential_phase_layout", "complex_real_imag_inplace_trig_v1"),
     ("total_potential_lifetime", "recompute_before_first_kick_release_before_save_v1"),
     ("saved_energy_density_lifetime", "release_before_kinetic_fft_rebuild_for_output_v1"),
     ("backend", "pytorch_cuda")],
)
def test_qe_box_control_rejects_mixed_solver_settings(
    tmp_path: Path, field: str, value: str,
) -> None:
    pair, box = _write_qe_box_pair(tmp_path)
    fine_path = tmp_path / "n512" / "fdm_adapter_metadata.json"
    fine = json.loads(fine_path.read_text())
    fine[field] = value
    fine_path.write_text(json.dumps(fine))
    with pytest.raises(ValueError, match="numerical settings differ"):
        assess_qe_box_control(
            CalibrationSource("test", pair), CalibrationSource("test", box)
        )


def test_qe_box_control_accepts_matching_separable_solver_layout(
    tmp_path: Path,
) -> None:
    pair, box = _write_qe_box_pair(tmp_path)
    for run in ("n384", "n512", "n1024"):
        path = tmp_path / run / "fdm_adapter_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["backend"] = "pytorch_cuda"
        metadata["kinetic_phase_layout"] = "separable_axis_v1"
        metadata["wave_buffer_lifetime"] = "release_previous_state_before_fft_v1"
        metadata["wave_density_layout"] = "real_imag_addcmul_v1"
        metadata["compact_potential_layout"] = "x_slab32_inplace_rsqrt_v1"
        metadata["potential_phase_layout"] = "complex_real_imag_inplace_trig_v1"
        metadata["total_potential_lifetime"] = "recompute_before_first_kick_release_before_save_v1"
        metadata["saved_energy_density_lifetime"] = "release_before_kinetic_fft_rebuild_for_output_v1"
        path.write_text(json.dumps(metadata))
    result = assess_qe_box_control(
        CalibrationSource("test", pair), CalibrationSource("test", box)
    )
    assert result["box_controlled_candidate_bins"] == [0]


def test_qe_box_control_rejects_changed_coarse_time_step(
    tmp_path: Path,
) -> None:
    pair, box = _write_qe_box_pair(tmp_path)
    coarse_path = tmp_path / "n384" / "fdm_adapter_metadata.json"
    coarse = json.loads(coarse_path.read_text())
    coarse["time_step_factor"] = 2.0
    coarse_path.write_text(json.dumps(coarse))
    summary = json.loads(pair.read_text())
    summary["runs"][1]["time_step_factor"] = 2.0
    pair.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="numerical settings"):
        assess_qe_box_control(
            CalibrationSource("test", pair), CalibrationSource("test", box)
        )


def test_qe_box_control_requires_raw_diagnostics_by_default(
    tmp_path: Path,
) -> None:
    pair, box = _write_qe_box_pair(tmp_path)
    with pytest.raises(ValueError, match="raw diagnostic is absent"):
        _assess_qe_box_control(
            CalibrationSource("test", pair), CalibrationSource("test", box)
        )


def test_qe_box_control_rejects_wrong_box_size(tmp_path: Path) -> None:
    pair, box = _write_qe_box_pair(tmp_path)
    metadata_path = tmp_path / "n1024" / "fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["box_size_pc"] = 60.0
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="double box size"):
        assess_qe_box_control(
            CalibrationSource("test", pair), CalibrationSource("test", box)
        )


def test_qe_box_control_rejects_changed_initial_conditions(tmp_path: Path) -> None:
    pair, box = _write_qe_box_pair(tmp_path)
    config_path = tmp_path / "n1024" / "config.uldm"
    config = json.loads(config_path.read_text())
    config["Matter Particles"]["Condition"][0][2][1] += 1.0
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="do not share one q/e physical case"):
        assess_qe_box_control(
            CalibrationSource("test", pair), CalibrationSource("test", box)
        )


def test_qe_resolution_pair_rejects_changed_initial_orbit(tmp_path: Path) -> None:
    pair, _ = _write_qe_box_pair(tmp_path)
    config_path = tmp_path / "n384" / "config.uldm"
    config = json.loads(config_path.read_text())
    config["Matter Particles"]["Condition"][0][2][1] += 1.0
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="same physical case"):
        build_source_rows(CalibrationSource("test", pair))


def test_qe_resolution_pair_requires_consistent_axes(tmp_path: Path) -> None:
    pair, _ = _write_qe_box_pair(tmp_path)
    metadata_path = tmp_path / "n384" / "fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["semi_major_axis_pc"] = 0.7
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="inconsistent q/e initial-orbit axes"):
        build_source_rows(CalibrationSource("test", pair))


def test_qe_box_control_censors_missing_control_bin(tmp_path: Path) -> None:
    pair, box_path = _write_qe_box_pair(tmp_path)
    box = json.loads(box_path.read_text())
    box["matched_separation"]["bins"] = []
    box["matched_separation"]["retained_bins"] = 0
    box_path.write_text(json.dumps(box))
    result = assess_qe_box_control(
        CalibrationSource("test", pair), CalibrationSource("test", box_path)
    )
    assert result["status"] == "no_box_controlled_candidate_bins_censored"
    assert result["box_controlled_candidate_bins"] == []


def test_qe_box_control_censors_failed_hamiltonian_gate(tmp_path: Path) -> None:
    pair, box_path = _write_qe_box_pair(tmp_path)
    box = json.loads(box_path.read_text())
    box["runs"][0]["initial_resolved_energy_drift_over_transfer"] = 0.02
    box_path.write_text(json.dumps(box))
    result = assess_qe_box_control(
        CalibrationSource("test", pair), CalibrationSource("test", box_path)
    )
    assert result["candidate_bins"] == [0]
    assert result["box_controlled_candidate_bins"] == []
    assert result["status"] == "no_box_controlled_candidate_bins_censored"


def test_qe_box_control_rejects_mismatched_fixed_edges(tmp_path: Path) -> None:
    pair, box_path = _write_qe_box_pair(tmp_path)
    box = json.loads(box_path.read_text())
    box["matched_separation"]["separation_bin_edges_pc"][1] = 0.9
    box["requested_separation_bin_edges_pc"][1] = 0.9
    box["matched_separation"]["bins"][0]["upper_separation_pc"] = 0.9
    box["matched_separation"]["bins"][1]["lower_separation_pc"] = 0.9
    box_path.write_text(json.dumps(box))
    with pytest.raises(ValueError, match="different physical bins"):
        assess_qe_box_control(
            CalibrationSource("test", pair), CalibrationSource("test", box_path)
        )


def test_qe_box_control_rejects_mixed_design_binding(tmp_path: Path) -> None:
    pair_path, box_path = _write_qe_box_pair(tmp_path)
    pair = json.loads(pair_path.read_text())
    pair["qe_design_binding"] = {
        "comparison_kind": "resolution_pair", "design_sha256": "a" * 64,
    }
    pair_path.write_text(json.dumps(pair))
    with pytest.raises(ValueError, match="mix registered and unregistered"):
        assess_qe_box_control(
            CalibrationSource("test", pair_path), CalibrationSource("test", box_path)
        )
    box = json.loads(box_path.read_text())
    box["qe_design_binding"] = {
        "comparison_kind": "doubled_box", "design_sha256": "b" * 64,
    }
    box_path.write_text(json.dumps(box))
    with pytest.raises(ValueError, match="different registered designs"):
        assess_qe_box_control(
            CalibrationSource("test", pair_path), CalibrationSource("test", box_path)
        )


def test_qe_box_control_rejects_inconsistent_shared_run(tmp_path: Path) -> None:
    pair, box_path = _write_qe_box_pair(tmp_path)
    box = json.loads(box_path.read_text())
    box["matched_separation"]["bins"][0]["runs"][1]["rates"]["orbital_power"]["estimate"] *= 1.1
    box["matched_separation"]["bins"][0]["runs"][1]["fractional_rate_difference_from_reference"]["orbital_power"] = 0.1
    box_path.write_text(json.dumps(box))
    with pytest.raises(ValueError, match="shared fine-run rate differs"):
        assess_qe_box_control(
            CalibrationSource("test", pair), CalibrationSource("test", box_path)
        )


def test_qe_box_control_rejects_inconsistent_shared_osculating_axis(
    tmp_path: Path,
) -> None:
    pair, box_path = _write_qe_box_pair(tmp_path)
    box = json.loads(box_path.read_text())
    box["matched_separation"]["bins"][0]["runs"][1][
        "mean_semimajor_axis_osculating_pc"
    ] *= 1.01
    box_path.write_text(json.dumps(box))
    with pytest.raises(ValueError, match="shared fine-run bin differs"):
        assess_qe_box_control(
            CalibrationSource("test", pair), CalibrationSource("test", box_path)
        )


def test_qe_box_control_rejects_missing_osculating_axis(tmp_path: Path) -> None:
    pair_path, box = _write_qe_box_pair(tmp_path)
    pair = json.loads(pair_path.read_text())
    del pair["matched_separation"]["bins"][0]["runs"][0][
        "mean_semimajor_axis_osculating_pc"
    ]
    pair_path.write_text(json.dumps(pair))
    with pytest.raises(ValueError, match="lacks measured osculating"):
        assess_qe_box_control(
            CalibrationSource("test", pair_path), CalibrationSource("test", box)
        )


def test_qe_box_control_rejects_invalid_eccentricity_range(tmp_path: Path) -> None:
    pair_path, box = _write_qe_box_pair(tmp_path)
    pair = json.loads(pair_path.read_text())
    pair["matched_separation"]["bins"][0]["runs"][0][
        "minimum_orbit_mean_eccentricity"
    ] = -0.01
    pair_path.write_text(json.dumps(pair))
    with pytest.raises(ValueError, match="invalid osculating coordinates"):
        assess_qe_box_control(
            CalibrationSource("test", pair_path), CalibrationSource("test", box)
        )


def _mock_strict_box_assessment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    pair_path, box_path = _write_qe_box_pair(tmp_path)
    decision = assess_qe_box_control(
        CalibrationSource("test", pair_path),
        CalibrationSource("test", box_path),
    )
    verified = {}
    for role, path in (("resolution_pair", pair_path), ("doubled_box", box_path)):
        summary = json.loads(path.read_text())
        inputs = []
        for row in summary["runs"]:
            run = Path(row["run"])
            hashes = {}
            for name in qe_box_module._RAW_INPUTS:
                input_path = run / name
                if not input_path.exists():
                    input_path.write_text("fixture\n")
                hashes[name] = hashlib.sha256(input_path.read_bytes()).hexdigest()
            inputs.append({"label": row["label"], "run": str(run), "sha256": hashes})
        verified[role] = {
            "comparison_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_inputs": inputs,
        }
    decision["raw_diagnostics_verified"] = True
    decision["raw_verification"] = verified
    monkeypatch.setattr(qe_box_module, "assess_qe_box_control", lambda *args, **kwargs: decision)
    return pair_path, box_path


def test_qe_candidate_package_retains_only_box_controlled_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair, box = _mock_strict_box_assessment(tmp_path, monkeypatch)
    package = qe_box_module.prepare_qe_calibration_candidate(
        CalibrationSource("test", pair), CalibrationSource("test", box)
    )
    assert package["status"] == "qe_box_controlled_candidate_not_released"
    assert package["candidate_row_count"] == 1
    assert package["candidate_rows"][0]["separation_bin_index"] == 0
    observations = package["mapping_observations_not_released"]
    assert len(observations) == 1
    assert observations[0]["separation_bin_index"] == 0
    assert observations[0]["fine"]["mean_semimajor_axis_osculating_pc"] == (
        observations[0]["coarse"]["mean_semimajor_axis_osculating_pc"]
    )
    assert observations[0]["fine"]["mean_semimajor_axis_osculating_pc"] == (
        0.52 * (0.4 + 0.8)
    )
    assert observations[0]["doubled_box"]["complete_orbits"] == 20
    assert observations[0]["necessary_coordinate_overlap"]["status"] == (
        "rectangular_overlap_necessary_only"
    )
    assert observations[0]["necessary_coordinate_overlap"][
        "runtime_mapping_admitted"
    ] is False
    disjoint = deepcopy(observations[0])
    disjoint["doubled_box"]["minimum_orbit_mean_eccentricity"] = 0.4
    disjoint["doubled_box"]["mean_eccentricity_osculating"] = 0.45
    disjoint["doubled_box"]["maximum_orbit_mean_eccentricity"] = 0.5
    assert qe_box_module._necessary_mapping_coordinate_overlap(disjoint)[
        "status"
    ] == "no_common_coordinate_rectangle_censored"
    assert package["production_calibration_row_admitted"] is False


def test_qe_candidate_package_rejects_changed_raw_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair, box = _mock_strict_box_assessment(tmp_path, monkeypatch)
    path = tmp_path / "n1024" / "conservation_summary.json"
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="diagnostic changed after box assessment"):
        qe_box_module.prepare_qe_calibration_candidate(
            CalibrationSource("test", pair), CalibrationSource("test", box)
        )


def test_release_loader_rejects_qe_source_without_doubled_box_control(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    summary_path = output.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    summary["sources"][0]["source_case_id"] = "qe_test"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="verified doubled-box control"):
        SubgridCalibrationTable.from_release(output)


def test_generated_release_drives_a_conservative_residual_update(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    table = SubgridCalibrationTable.from_release(output)
    rates = physical_subgrid_rates(
        table,
        profile_id="boey2025",
        mass1_msun=2.0e7,
        mass2_msun=2.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=2.0,
        particle_mass_ev=1.0e-21,
        separation_pc=0.6,
        eccentricity=0.23,
    )
    with pytest.raises(ValueError, match="similarity parameter"):
        physical_subgrid_rates(
            table,
            profile_id="boey2025",
            mass1_msun=2.0e7,
            mass2_msun=2.0e7,
            soliton_mass_msun=1.0e9,
            core_radius_pc=2.0,
            particle_mass_ev=2.0e-21,
            separation_pc=0.6,
            eccentricity=0.23,
        )
    assert rates.dimensionless.dimensionless_orbital_power == pytest.approx(-1.0)
    assert rates.dimensionless.dimensionless_orbital_torque == pytest.approx(-2.0)
    step = advance_calibrated_exchange(
        rates,
        mass1_msun=2.0e7,
        mass2_msun=2.0e7,
        semimajor_axis_pc=0.6,
        eccentricity=0.1,
        time_step_myr=1.0e-8,
        resolved_orbital_power=0.25 * rates.orbital_power,
        resolved_orbital_torque=0.40 * rates.orbital_torque,
    )
    assert step.residual.residual_orbital_power == pytest.approx(
        0.75 * rates.orbital_power
    )
    assert step.residual.residual_orbital_torque == pytest.approx(
        0.60 * rates.orbital_torque
    )
    assert abs(step.energy_closure_relative_to_exchange) < 1.0e-10
    assert abs(step.angular_momentum_closure_relative_to_exchange) < 1.0e-10


def test_release_verification_is_durable_and_resumable(tmp_path: Path) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    verifier = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "verify_subgrid_calibration_release.py"
    )
    command = [
        sys.executable,
        str(verifier),
        str(output),
        "--required-profile",
        "boey2025",
        "--expected-source-count",
        "1",
        "--required-q-e-plane",
        "1.0,0.23",
    ]
    subprocess.run(command, check=True)
    verification_path = output.with_suffix(".verification.json")
    report = json.loads(verification_path.read_text())
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert report["status"] == "subgrid_calibration_release_verified"
    assert report["release_input_sha256"] == summary["release_input_sha256"]
    assert report["table"]["sha256"] == summary["table"]["sha256"]
    assert report["runtime"]["rows"] == 1
    assert report["accepted_q_e_planes"] == [
        {"mass_ratio_q": 1.0, "reference_eccentricity": 0.23}
    ]

    # A stopped pair publication cannot leave the previous success marker.
    with output.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    failed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert failed.returncode != 0
    assert not verification_path.exists()

    # A normal rerun repairs the release and regenerates the audit record.
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    subprocess.run(command, check=True)
    assert verification_path.is_file()


def test_release_loader_rejects_interrupted_pair_publish(tmp_path: Path) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)

    # This is the state produced if a newer CSV is replaced and the process is
    # stopped before publishing its matching summary commit marker.
    with output.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        SubgridCalibrationTable.from_release(output)

    # A normal rerun repairs the pair and makes it loadable again.
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    assert len(SubgridCalibrationTable.from_release(output).rows) == 1


def test_release_identity_changes_when_provenance_changes_but_rows_do_not(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    first_summary = write_calibration_table(
        [CalibrationSource("boey2025", path)], output=output
    )
    first_rows = SubgridCalibrationTable.from_csv(output).rows
    old_commit_marker = output.with_suffix(".summary.json").read_text()

    metadata_path = tmp_path / "n384" / "fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["provenance_note"] = "same physics, different input bytes"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True))
    second_summary = write_calibration_table(
        [CalibrationSource("boey2025", path)], output=output
    )
    assert SubgridCalibrationTable.from_csv(output).rows == first_rows
    assert second_summary["release_input_sha256"] != first_summary[
        "release_input_sha256"
    ]
    assert second_summary["table"]["sha256"] != first_summary["table"]["sha256"]

    # This reproduces a stop after the new CSV replace but before the new
    # commit-marker replace: the previous summary must not validate it.
    output.with_suffix(".summary.json").write_text(old_commit_marker)
    with pytest.raises(ValueError, match="table checksum does not match"):
        SubgridCalibrationTable.from_release(output)

    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    assert SubgridCalibrationTable.from_release(output).rows == first_rows


def test_release_loader_requires_a_commit_sidecar(tmp_path: Path) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    output.with_suffix(".summary.json").unlink()
    with pytest.raises(ValueError, match="commit sidecar is absent"):
        SubgridCalibrationTable.from_release(output)


@pytest.mark.parametrize(
    "missing_field",
    (
        "mass_ratio_q",
        "reference_eccentricity",
        "absolute_mean_eccentricity_mismatch",
    ),
)
def test_schema_four_release_rejects_blank_qe_provenance(
    tmp_path: Path, missing_field: str,
) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        records = list(reader)
    assert names is not None and len(records) == 1
    records[0][missing_field] = ""
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(records)
    summary_path = output.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    summary["table"]["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="q/e provenance fields are absent"):
        SubgridCalibrationTable.from_release(output)


def test_release_loader_rejects_relaxed_acceptance_criteria(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    summary_path = output.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    summary["acceptance"]["maximum_spatial_systematic_fraction"] = 0.21
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="acceptance criteria are unsafe"):
        SubgridCalibrationTable.from_release(output)


@pytest.mark.parametrize(
    "criterion,value",
    (
        ("maximum_spatial_systematic_fraction", 0.05),
        ("minimum_complete_orbits_per_bin", 100),
        ("minimum_core_radius_cells", 100.0),
    ),
)
def test_release_loader_applies_recorded_limits_to_each_row(
    tmp_path: Path, criterion: str, value: float,
) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    summary_path = output.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    summary["acceptance"][criterion] = value
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="row exceeds its recorded acceptance criteria"):
        SubgridCalibrationTable.from_release(output)


def test_release_loader_rejects_relaxed_eccentricity_criterion(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    summary_path = output.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    summary["acceptance"]["maximum_eccentricity_mismatch"] = 0.021
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="acceptance criteria are unsafe"):
        SubgridCalibrationTable.from_release(output)


def test_release_loader_applies_recorded_eccentricity_limit_to_rows(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table(
        [CalibrationSource("boey2025", path)],
        output=output,
        maximum_eccentricity_mismatch=0.01,
    )
    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        records = list(reader)
    assert fieldnames is not None
    records[0]["absolute_mean_eccentricity_mismatch"] = "0.015"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary_path = output.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    summary["table"]["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(
        ValueError,
        match="row exceeds the recorded eccentricity acceptance criterion",
    ):
        SubgridCalibrationTable.from_release(output)


def test_cli_rejects_relaxed_eccentricity_criterion(tmp_path: Path) -> None:
    path = _write_summary(tmp_path)
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_subgrid_calibration_table.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source",
            f"boey2025={path}",
            "--output",
            str(tmp_path / "subgrid.csv"),
            "--maximum-eccentricity-mismatch",
            "0.021",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "subgrid acceptance limits are invalid" in completed.stderr


def test_release_loader_rejects_a_provenance_count_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    summary_path = output.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    summary["sources"][0]["accepted_bins"] = 0
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="provenance row count does not close"):
        SubgridCalibrationTable.from_release(output)


def test_release_loader_requires_complete_input_provenance(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    write_calibration_table([CalibrationSource("boey2025", path)], output=output)
    summary_path = output.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    summary["sources"][0]["inputs"] = summary["sources"][0]["inputs"][:-1]
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="provenance roles are incomplete"):
        SubgridCalibrationTable.from_release(output)
