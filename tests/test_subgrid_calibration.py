from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from fdm_smbh_delay.subgrid_calibration import (
    ACCEPTED_STATUS,
    SubgridCalibrationRow,
    SubgridCalibrationTable,
    advance_calibrated_exchange,
    physical_subgrid_rates,
    residual_orbital_rates,
)


def _row(
    *,
    mass_fraction: float,
    bin_index: int,
    lower: float,
    upper: float,
    centre: float,
    power: float,
    torque: float,
    systematic: float,
    profile: str = "boey2025",
) -> SubgridCalibrationRow:
    return SubgridCalibrationRow(
        profile_id=profile,
        source_case_id=f"{profile}_{mass_fraction}",
        schrodinger_poisson_similarity_parameter=0.388,
        binary_to_soliton_mass=mass_fraction,
        separation_bin_index=bin_index,
        lower_separation_over_core_radius=lower,
        upper_separation_over_core_radius=upper,
        reference_mean_separation_over_core_radius=centre,
        dimensionless_orbital_power=power,
        dimensionless_orbital_torque=torque,
        dimensionless_wave_total_energy_rate=-power,
        orbital_power_spatial_systematic_fraction=systematic,
        orbital_torque_spatial_systematic_fraction=systematic + 0.01,
        wave_total_spatial_systematic_fraction=systematic + 0.02,
        reference_resolution=512,
        comparison_resolution=384,
        reference_complete_orbits=20,
        comparison_complete_orbits=18,
        reference_minimum_half_density_radius_over_cell_size=10.0,
        comparison_minimum_half_density_radius_over_cell_size=7.0,
        convergence_status=ACCEPTED_STATUS,
    )


def _table() -> SubgridCalibrationTable:
    rows = []
    for mass, offset in ((0.04, 0.0), (0.10, 1.0)):
        rows.extend(
            (
                _row(
                    mass_fraction=mass,
                    bin_index=0,
                    lower=0.2,
                    upper=0.4,
                    centre=0.3,
                    power=-1.0 - offset,
                    torque=-2.0 - offset,
                    systematic=0.05 + 0.01 * offset,
                ),
                _row(
                    mass_fraction=mass,
                    bin_index=1,
                    lower=0.4,
                    upper=0.6,
                    centre=0.5,
                    power=-3.0 - offset,
                    torque=-4.0 - offset,
                    systematic=0.07 + 0.01 * offset,
                ),
            )
        )
    return SubgridCalibrationTable(rows)


def test_interpolation_is_linear_and_systematic_is_conservative() -> None:
    result = _table().interpolate(
        profile_id="boey2025",
        binary_to_soliton_mass=0.07,
        separation_over_core_radius=0.4,
    )
    assert result.dimensionless_orbital_power == pytest.approx(-2.5)
    assert result.dimensionless_orbital_torque == pytest.approx(-3.5)
    assert result.dimensionless_wave_total_energy_rate == pytest.approx(2.5)
    assert result.orbital_power_spatial_systematic_fraction == pytest.approx(
        0.08
    )
    assert result.orbital_torque_spatial_systematic_fraction == pytest.approx(
        0.09
    )


@pytest.mark.parametrize(
    "profile,mass,separation",
    [
        ("unknown", 0.07, 0.4),
        ("boey2025", 0.02, 0.4),
        ("boey2025", 0.12, 0.4),
        ("boey2025", 0.07, 0.1),
        ("boey2025", 0.07, 0.7),
    ],
)
def test_interpolation_rejects_unknown_profiles_and_extrapolation(
    profile: str, mass: float, separation: float
) -> None:
    with pytest.raises(ValueError):
        _table().interpolate(
            profile_id=profile,
            binary_to_soliton_mass=mass,
            separation_over_core_radius=separation,
        )


def test_interpolation_rejects_a_missing_separation_bin() -> None:
    table = SubgridCalibrationTable(
        (
            _row(
                mass_fraction=0.04,
                bin_index=0,
                lower=0.2,
                upper=0.4,
                centre=0.3,
                power=-1.0,
                torque=-2.0,
                systematic=0.05,
            ),
            _row(
                mass_fraction=0.04,
                bin_index=2,
                lower=0.6,
                upper=0.8,
                centre=0.7,
                power=-3.0,
                torque=-4.0,
                systematic=0.08,
            ),
        )
    )
    with pytest.raises(ValueError, match="unmeasured gap"):
        table.interpolate(
            profile_id="boey2025",
            binary_to_soliton_mass=0.04,
            separation_over_core_radius=0.5,
        )


def test_csv_round_trip_is_a_regression_fixture(tmp_path: Path) -> None:
    original = _table()
    path = tmp_path / "subgrid.csv"
    names = [field.name for field in fields(SubgridCalibrationRow)]
    import csv

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in original.rows:
            writer.writerow({name: getattr(row, name) for name in names})
    loaded = SubgridCalibrationTable.from_csv(path)
    assert loaded.rows == original.rows


def test_physical_update_closes_energy_and_angular_momentum() -> None:
    rates = physical_subgrid_rates(
        _table(),
        profile_id="boey2025",
        mass1_msun=2.0e7,
        mass2_msun=2.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=2.0,
        separation_pc=0.8,
    )
    assert rates.dimensionless.binary_to_soliton_mass == pytest.approx(0.04)
    assert rates.dimensionless.separation_over_core_radius == pytest.approx(0.4)
    assert rates.orbital_power_spatial_systematic == pytest.approx(
        abs(rates.orbital_power) * 0.07
    )
    step = advance_calibrated_exchange(
        rates,
        mass1_msun=2.0e7,
        mass2_msun=2.0e7,
        semimajor_axis_pc=0.8,
        eccentricity=0.1,
        time_step_myr=1.0e-8,
    )
    assert abs(step.energy_closure_relative_to_exchange) < 1.0e-10
    assert abs(step.angular_momentum_closure_relative_to_exchange) < 1.0e-10


def test_physical_rates_follow_the_internal_unit_scaling() -> None:
    reference = physical_subgrid_rates(
        _table(),
        profile_id="boey2025",
        mass1_msun=2.0e7,
        mass2_msun=2.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=2.0,
        separation_pc=0.8,
    )
    mass_scale = 3.0
    length_scale = 5.0
    rescaled = physical_subgrid_rates(
        _table(),
        profile_id="boey2025",
        mass1_msun=mass_scale * 2.0e7,
        mass2_msun=mass_scale * 2.0e7,
        soliton_mass_msun=mass_scale * 1.0e9,
        core_radius_pc=length_scale * 2.0,
        separation_pc=length_scale * 0.8,
    )
    power_scale = mass_scale**2.5 / length_scale**2.5
    torque_scale = mass_scale**2 / length_scale
    assert rescaled.orbital_power == pytest.approx(
        power_scale * reference.orbital_power
    )
    assert rescaled.wave_total_energy_rate == pytest.approx(
        power_scale * reference.wave_total_energy_rate
    )
    assert rescaled.orbital_torque == pytest.approx(
        torque_scale * reference.orbital_torque
    )
    assert rescaled.orbital_power_spatial_systematic == pytest.approx(
        power_scale * reference.orbital_power_spatial_systematic
    )


def test_repeated_finite_updates_close_cumulative_exchange() -> None:
    rates = physical_subgrid_rates(
        _table(),
        profile_id="boey2025",
        mass1_msun=2.0e7,
        mass2_msun=2.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=2.0,
        separation_pc=0.8,
    )
    semimajor_axis = 0.8
    eccentricity = 0.1
    total_wave_energy = 0.0
    total_wave_angular_momentum = 0.0
    initial_energy = None
    initial_angular_momentum = None
    final_energy = None
    final_angular_momentum = None
    for _ in range(100):
        step = advance_calibrated_exchange(
            rates,
            mass1_msun=2.0e7,
            mass2_msun=2.0e7,
            semimajor_axis_pc=semimajor_axis,
            eccentricity=eccentricity,
            time_step_myr=1.0e-9,
        )
        if initial_energy is None:
            initial_energy = step.exchange.initial_orbital_energy
            initial_angular_momentum = (
                step.exchange.initial_orbital_angular_momentum
            )
        total_wave_energy += step.exchange.wave_energy_increment
        total_wave_angular_momentum += (
            step.exchange.wave_angular_momentum_increment
        )
        final_energy = step.exchange.final_orbital_energy
        final_angular_momentum = step.exchange.final_orbital_angular_momentum
        semimajor_axis = step.exchange.final_semimajor_axis_pc
        eccentricity = step.exchange.final_eccentricity
    assert final_energy - initial_energy + total_wave_energy == pytest.approx(
        0.0, abs=abs(total_wave_energy) * 1.0e-9
    )
    assert (
        final_angular_momentum
        - initial_angular_momentum
        + total_wave_angular_momentum
        == pytest.approx(0.0, abs=abs(total_wave_angular_momentum) * 1.0e-9)
    )


def test_residual_rates_remove_resolved_work_without_double_counting() -> None:
    rates = physical_subgrid_rates(
        _table(),
        profile_id="boey2025",
        mass1_msun=2.0e7,
        mass2_msun=2.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=2.0,
        separation_pc=0.8,
    )
    residual = residual_orbital_rates(
        rates,
        resolved_orbital_power=0.25 * rates.orbital_power,
        resolved_orbital_torque=0.40 * rates.orbital_torque,
    )
    assert residual.resolved_orbital_power + residual.residual_orbital_power == (
        pytest.approx(residual.target_orbital_power)
    )
    assert (
        residual.resolved_orbital_torque + residual.residual_orbital_torque
        == pytest.approx(residual.target_orbital_torque)
    )


def test_finite_update_applies_only_exchange_absent_from_the_resolved_wake() -> None:
    rates = physical_subgrid_rates(
        _table(),
        profile_id="boey2025",
        mass1_msun=2.0e7,
        mass2_msun=2.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=2.0,
        separation_pc=0.8,
    )
    time_step = 1.0e-8
    step = advance_calibrated_exchange(
        rates,
        mass1_msun=2.0e7,
        mass2_msun=2.0e7,
        semimajor_axis_pc=0.8,
        eccentricity=0.1,
        time_step_myr=time_step,
        resolved_orbital_power=0.25 * rates.orbital_power,
        resolved_orbital_torque=0.40 * rates.orbital_torque,
    )
    assert step.residual.residual_orbital_power == pytest.approx(
        0.75 * rates.orbital_power
    )
    assert step.residual.residual_orbital_torque == pytest.approx(
        0.60 * rates.orbital_torque
    )
    assert step.exchange.wave_energy_increment == pytest.approx(
        -step.residual.residual_orbital_power * time_step
    )
    assert step.exchange.wave_angular_momentum_increment == pytest.approx(
        -step.residual.residual_orbital_torque * time_step
    )
    assert abs(step.energy_closure_relative_to_exchange) < 1.0e-10
    assert abs(step.angular_momentum_closure_relative_to_exchange) < 1.0e-10


def test_row_rejects_unaccepted_or_underresolved_measurements() -> None:
    with pytest.raises(ValueError, match="spatial convergence gate"):
        SubgridCalibrationRow(
            **{
                **_row(
                    mass_fraction=0.04,
                    bin_index=0,
                    lower=0.2,
                    upper=0.4,
                    centre=0.3,
                    power=-1.0,
                    torque=-2.0,
                    systematic=0.05,
                ).__dict__,
                "convergence_status": "provisional",
            }
        )
    with pytest.raises(ValueError, match="numerical metadata"):
        SubgridCalibrationRow(
            **{
                **_row(
                    mass_fraction=0.04,
                    bin_index=0,
                    lower=0.2,
                    upper=0.4,
                    centre=0.3,
                    power=-1.0,
                    torque=-2.0,
                    systematic=0.05,
                ).__dict__,
                "comparison_complete_orbits": 7,
            }
        )
    with pytest.raises(ValueError, match="outside the accepted range"):
        SubgridCalibrationRow(
            **{
                **_row(
                    mass_fraction=0.04,
                    bin_index=0,
                    lower=0.2,
                    upper=0.4,
                    centre=0.3,
                    power=-1.0,
                    torque=-2.0,
                    systematic=0.05,
                ).__dict__,
                "orbital_power_spatial_systematic_fraction": 0.21,
            }
        )
