import pytest

from fdm_smbh_delay.wave_modes import decompose_wave_mode_exchange


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
