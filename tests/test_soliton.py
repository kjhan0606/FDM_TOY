from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.soliton import SchiveSoliton


def test_total_profile_normalization() -> None:
    soliton = SchiveSoliton.from_mass(1.0e9, 20.0, "total_profile")
    assert soliton.total_mass_msun == pytest.approx(1.0e9, rel=1.0e-14)
    assert soliton.enclosed_mass(1.0e6) == pytest.approx(1.0e9, rel=1.0e-10)


def test_within_rc_normalization_is_distinct() -> None:
    soliton = SchiveSoliton.from_mass(2.0e8, 20.0, "within_rc")
    assert soliton.enclosed_mass(20.0) == pytest.approx(2.0e8, rel=1.0e-14)
    assert soliton.total_mass_msun > 2.0e8


def test_potential_derivative_matches_enclosed_mass() -> None:
    soliton = SchiveSoliton.from_mass(1.0e9, 20.0, "total_profile")
    radius = 7.0
    step = 1.0e-4
    derivative = (soliton.potential(radius + step) - soliton.potential(radius - step)) / (2 * step)
    expected = G_INTERNAL * soliton.enclosed_mass(radius) / radius**2
    assert derivative == pytest.approx(expected, rel=2.0e-9)


def test_binding_proxy_scaling() -> None:
    base = SchiveSoliton.from_mass(1.0e8, 10.0, "total_profile")
    double_mass = SchiveSoliton.from_mass(2.0e8, 10.0, "total_profile")
    double_radius = SchiveSoliton.from_mass(1.0e8, 20.0, "total_profile")
    assert base.virial_binding_energy() > 0.0
    assert double_mass.virial_binding_energy() / base.virial_binding_energy() == pytest.approx(4.0)
    assert double_radius.virial_binding_energy() / base.virial_binding_energy() == pytest.approx(0.5)


def test_acceleration_is_central() -> None:
    soliton = SchiveSoliton.from_mass(1.0e8, 10.0, "total_profile")
    position = np.array([1.0, -2.0, 3.0])
    acceleration = soliton.acceleration(position)
    assert np.linalg.norm(np.cross(position, acceleration)) < 1.0e-12
    assert position @ acceleration < 0.0
