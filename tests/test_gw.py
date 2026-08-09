from __future__ import annotations

import pytest

from fdm_smbh_delay.gw import peters_circular_time_myr, peters_time_myr


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
