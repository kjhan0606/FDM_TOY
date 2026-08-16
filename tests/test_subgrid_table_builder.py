from __future__ import annotations

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
    assert summary["schema_version"] == 3
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
        }
    ]
    assert len(summary["table"]["sha256"]) == 64
    assert output.with_suffix(".summary.json").is_file()
    loaded = SubgridCalibrationTable.from_release(output)
    assert len(loaded.rows) == 1
    assert loaded.rows[0].mass_ratio_q == pytest.approx(1.0)
    assert loaded.rows[0].reference_eccentricity == pytest.approx(0.23)


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
