from __future__ import annotations

import pytest

from fdm_smbh_delay.gw import (
    peters_circular_time_myr,
    peters_orbital_rates,
    peters_time_myr,
)


def test_circular_peters_reference_value() -> None:
    assert peters_circular_time_myr(1.0e8, 1.0e8, 0.01) == pytest.approx(
        2.9064768606011304, rel=1.0e-14
    )


def test_eccentricity_shortens_gw_time() -> None:
    circular = peters_time_myr(1.0e8, 1.0e8, 0.01, 0.0)
    assert peters_time_myr(1.0e8, 1.0e8, 0.01, 0.7) < circular


def test_peters_mass_and_radius_scaling() -> None:
    base = peters_circular_time_myr(1.0e8, 1.0e8, 0.01)
    assert peters_circular_time_myr(1.0e8, 1.0e8, 0.02) == pytest.approx(16.0 * base)
    assert peters_circular_time_myr(2.0e8, 2.0e8, 0.01) == pytest.approx(base / 8.0)


def test_peters_instantaneous_rates_are_regular_for_circular_orbit() -> None:
    semimajor_axis = 0.01
    rates = peters_orbital_rates(1.0e8, 1.0e8, semimajor_axis, 0.0)
    coalescence_time = peters_circular_time_myr(1.0e8, 1.0e8, semimajor_axis)
    assert rates.semimajor_axis_rate_pc_myr < 0.0
    assert semimajor_axis / abs(rates.semimajor_axis_rate_pc_myr) == pytest.approx(
        4.0 * coalescence_time
    )
    assert rates.eccentricity_rate_per_myr == 0.0
    assert rates.eccentricity_squared_rate_per_myr == 0.0


def test_peters_eccentric_rates_shrink_and_circularize() -> None:
    rates = peters_orbital_rates(1.0e8, 5.0e7, 0.02, 0.7)
    assert rates.semimajor_axis_rate_pc_myr < 0.0
    assert rates.eccentricity_rate_per_myr < 0.0
    assert rates.eccentricity_squared_rate_per_myr < 0.0
