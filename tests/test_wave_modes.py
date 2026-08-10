import numpy as np
import pytest

from fdm_smbh_delay.wave_modes import (
    decompose_wave_mode_exchange,
    residual_wave_exchange,
)


def test_mode_decomposition_closes_power_and_torque() -> None:
    result = decompose_wave_mode_exchange(
        orbital_power=-10.0,
        orbital_torque=-2.0,
        orbital_frequency_myr_inverse=4.0,
    )
    assert result.wave_power == pytest.approx(10.0)
    assert result.wave_torque == pytest.approx(2.0)
    assert result.rotating_pattern_power == pytest.approx(8.0)
    assert result.radial_residual_power == pytest.approx(2.0)
    assert result.power_over_orbital_frequency_times_torque == pytest.approx(1.25)
    assert result.nonnegative_radial_residual


def test_zero_torque_assigns_all_power_to_radial_response() -> None:
    result = decompose_wave_mode_exchange(
        orbital_power=-3.0,
        orbital_torque=0.0,
        orbital_frequency_myr_inverse=5.0,
    )
    assert result.rotating_pattern_power == 0.0
    assert result.radial_residual_power == pytest.approx(3.0)
    assert result.effective_pattern_speed_myr_inverse is None


def test_negative_radial_residual_requires_additional_harmonics() -> None:
    result = decompose_wave_mode_exchange(
        orbital_power=-5.0,
        orbital_torque=-2.0,
        orbital_frequency_myr_inverse=4.0,
    )
    assert result.radial_residual_power == pytest.approx(-3.0)
    assert not result.nonnegative_radial_residual


def test_residual_exchange_subtracts_resolved_work_and_torque() -> None:
    result = residual_wave_exchange(
        orbital_energy_increment=-10.0,
        orbital_angular_momentum_increment=np.array([0.0, 0.0, -4.0]),
        resolved_wave_energy_increment=7.0,
        resolved_wave_angular_momentum_increment=np.array([0.0, 0.0, 1.5]),
    )
    assert result.target_wave_energy_increment == pytest.approx(10.0)
    np.testing.assert_allclose(
        result.target_wave_angular_momentum_increment, [0.0, 0.0, 4.0]
    )
    assert result.residual_wave_energy_increment == pytest.approx(3.0)
    np.testing.assert_allclose(
        result.residual_wave_angular_momentum_increment, [0.0, 0.0, 2.5]
    )
    assert result.energy_closure_error == pytest.approx(0.0)
    np.testing.assert_allclose(result.angular_momentum_closure_error, 0.0)


def test_no_residual_when_multipole_supplies_the_target() -> None:
    result = residual_wave_exchange(
        orbital_energy_increment=-2.0,
        orbital_angular_momentum_increment=np.array([0.5, -0.25, -1.0]),
        resolved_wave_energy_increment=2.0,
        resolved_wave_angular_momentum_increment=np.array([-0.5, 0.25, 1.0]),
    )
    assert result.residual_wave_energy_increment == pytest.approx(0.0)
    np.testing.assert_allclose(result.residual_wave_angular_momentum_increment, 0.0)
