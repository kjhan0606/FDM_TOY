from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fdm_smbh_delay.exchange_scaling import exchange_scales
from fdm_smbh_delay.subgrid_calibration import SubgridCalibrationTable
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
                        "maximum_energy_error_over_transfer": 0.002,
                    },
                    {
                        "label": "n384",
                        "run": str(n384),
                        "maximum_energy_error_over_transfer": 0.003,
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


def test_builder_requires_wave_response_for_both_resolutions(
    tmp_path: Path,
) -> None:
    path = _write_summary(tmp_path)
    (tmp_path / "n384" / "wave_response_timeseries.csv").unlink()
    with pytest.raises(FileNotFoundError, match="requires sparse wave response"):
        build_source_rows(CalibrationSource("boey2025", path))


def test_writer_is_loadable_by_the_runtime_table(tmp_path: Path) -> None:
    path = _write_summary(tmp_path)
    output = tmp_path / "subgrid.csv"
    summary = write_calibration_table(
        [CalibrationSource("boey2025", path)], output=output
    )
    assert summary["rows"] == 1
    assert summary["sources"][0]["accepted_bins"] == 1
    assert output.with_suffix(".summary.json").is_file()
    loaded = SubgridCalibrationTable.from_csv(output)
    assert len(loaded.rows) == 1
