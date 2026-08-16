from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.galaxy_environment import (
    CompositePotential,
    DehnenProfile,
    GasBackground,
    NuclearEnvelope,
)


def test_dehnen_mass_and_potential_gradient() -> None:
    profile = DehnenProfile(1.0e10, 500.0, 1.0)
    assert profile.enclosed_mass(500.0) == pytest.approx(0.25e10)
    radius = 120.0
    step = 1.0e-3
    derivative = (profile.potential(radius + step) - profile.potential(radius - step)) / (
        2.0 * step
    )
    expected = G_INTERNAL * profile.enclosed_mass(radius) / radius**2
    assert derivative == pytest.approx(expected, rel=2.0e-10)


def test_composite_acceleration_uses_all_enclosed_mass() -> None:
    stars = DehnenProfile(1.0e10, 500.0, 1.0)
    gas = DehnenProfile(2.0e9, 800.0, 0.0)
    host = CompositePotential((stars, gas), central_point_mass_msun=1.0e8)
    position = np.array([100.0, 0.0, 0.0])
    acceleration = host.acceleration(position)
    expected = -G_INTERNAL * host.enclosed_mass(100.0) / 100.0**2
    assert acceleration[0] == pytest.approx(expected)
    assert np.allclose(acceleration[1:], 0.0)


def test_gas_rotation_is_tangential() -> None:
    gas_profile = DehnenProfile(1.0e9, 500.0, 1.0)
    gas = GasBackground(
        gas_profile,
        sound_speed_pc_myr=10.0,
        rotation_fraction=0.5,
        rotation_axis=np.array([0.0, 0.0, 1.0]),
    )
    position = np.array([100.0, 0.0, 0.0])
    velocity = gas.bulk_velocity(position, 1.0e10)
    assert velocity[1] > 0.0
    assert position @ velocity == pytest.approx(0.0)


def test_nuclear_envelope_mass_is_capped_by_initial_bound_mass() -> None:
    envelope = NuclearEnvelope(2.0e8, 10.0, 100.0)
    assert envelope.mass_within(100.0) == pytest.approx(2.0e8)
    assert envelope.mass_within(1000.0) == pytest.approx(2.0e8)
    assert envelope.mass_within(1.0) < envelope.mass_within(10.0)
