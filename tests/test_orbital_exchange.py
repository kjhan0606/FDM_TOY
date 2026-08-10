import numpy as np
import pytest

from fdm_smbh_delay.orbital_exchange import (
    advance_keplerian_exchange,
    keplerian_elements_from_relative_state,
    keplerian_exchange_rates,
)


def test_circular_relative_state_recovers_keplerian_elements() -> None:
    elements = keplerian_elements_from_relative_state(
        total_mass=3.0,
        displacement=np.array([2.0, 0.0, 0.0]),
        relative_velocity=np.array([0.0, np.sqrt(1.5), 0.0]),
        gravitational_constant=1.0,
    )
    assert elements.semimajor_axis == pytest.approx(2.0)
    assert elements.eccentricity == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(
        elements.specific_angular_momentum, [0.0, 0.0, np.sqrt(6.0)]
    )


def test_unbound_relative_state_has_no_semimajor_axis() -> None:
    elements = keplerian_elements_from_relative_state(
        total_mass=1.0,
        displacement=np.array([1.0, 0.0, 0.0]),
        relative_velocity=np.array([0.0, 2.0, 0.0]),
        gravitational_constant=1.0,
    )
    assert elements.specific_energy > 0.0
    assert elements.semimajor_axis is None


def test_negative_power_shrinks_semimajor_axis_and_closes_exchange() -> None:
    result = keplerian_exchange_rates(
        mass1_msun=1.0e8,
        mass2_msun=5.0e7,
        semimajor_axis_pc=0.5,
        eccentricity=0.4,
        orbital_power=-2.0e10,
        orbital_torque=-3.0e7,
    )
    assert result.semimajor_axis_rate_pc_myr < 0.0
    assert result.wave_energy_rate == pytest.approx(2.0e10)
    assert result.wave_angular_momentum_rate == pytest.approx(3.0e7)
    assert result.eccentricity_rate_per_myr == pytest.approx(
        result.eccentricity_squared_rate_per_myr / 0.8
    )


def test_circular_relation_preserves_zero_eccentricity() -> None:
    reference = keplerian_exchange_rates(
        mass1_msun=1.0e8,
        mass2_msun=1.0e8,
        semimajor_axis_pc=1.0,
        eccentricity=0.0,
        orbital_power=-1.0e10,
        orbital_torque=0.0,
    )
    circular_torque = (
        0.5
        * reference.orbital_angular_momentum
        * reference.semimajor_axis_rate_pc_myr
    )
    result = keplerian_exchange_rates(
        mass1_msun=1.0e8,
        mass2_msun=1.0e8,
        semimajor_axis_pc=1.0,
        eccentricity=0.0,
        orbital_power=-1.0e10,
        orbital_torque=circular_torque,
    )
    assert result.eccentricity_squared_rate_per_myr == pytest.approx(0.0)
    assert result.eccentricity_rate_per_myr == pytest.approx(0.0)


def test_non_circular_rate_at_e_zero_returns_only_e_squared_rate() -> None:
    result = keplerian_exchange_rates(
        mass1_msun=1.0e8,
        mass2_msun=1.0e8,
        semimajor_axis_pc=1.0,
        eccentricity=0.0,
        orbital_power=-1.0e10,
        orbital_torque=0.0,
    )
    assert result.eccentricity_squared_rate_per_myr < 0.0
    assert result.eccentricity_rate_per_myr is None


def test_finite_exchange_recovers_target_circular_orbit() -> None:
    initial = keplerian_exchange_rates(
        mass1_msun=1.0e8,
        mass2_msun=5.0e7,
        semimajor_axis_pc=1.0,
        eccentricity=0.0,
        orbital_power=0.0,
        orbital_torque=0.0,
    )
    target = keplerian_exchange_rates(
        mass1_msun=1.0e8,
        mass2_msun=5.0e7,
        semimajor_axis_pc=0.9,
        eccentricity=0.0,
        orbital_power=0.0,
        orbital_torque=0.0,
    )
    time_step = 0.01
    step = advance_keplerian_exchange(
        mass1_msun=1.0e8,
        mass2_msun=5.0e7,
        semimajor_axis_pc=1.0,
        eccentricity=0.0,
        orbital_power=(target.orbital_energy - initial.orbital_energy) / time_step,
        orbital_torque=(
            target.orbital_angular_momentum - initial.orbital_angular_momentum
        )
        / time_step,
        time_step_myr=time_step,
    )
    assert step.final_semimajor_axis_pc == pytest.approx(0.9)
    assert step.final_eccentricity == pytest.approx(0.0, abs=2.0e-8)
    assert step.wave_energy_increment == pytest.approx(
        initial.orbital_energy - target.orbital_energy
    )
    assert step.wave_angular_momentum_increment == pytest.approx(
        initial.orbital_angular_momentum - target.orbital_angular_momentum
    )


def test_finite_exchange_rejects_unbound_energy() -> None:
    initial = keplerian_exchange_rates(
        mass1_msun=1.0,
        mass2_msun=1.0,
        semimajor_axis_pc=1.0,
        eccentricity=0.2,
        orbital_power=0.0,
        orbital_torque=0.0,
    )
    with pytest.raises(ValueError, match="unbound"):
        advance_keplerian_exchange(
            mass1_msun=1.0,
            mass2_msun=1.0,
            semimajor_axis_pc=1.0,
            eccentricity=0.2,
            orbital_power=-2.0 * initial.orbital_energy,
            orbital_torque=0.0,
            time_step_myr=1.0,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("mass1_msun", 0.0),
        ("mass2_msun", -1.0),
        ("semimajor_axis_pc", 0.0),
        ("eccentricity", 1.0),
    ],
)
def test_invalid_orbital_inputs(field: str, value: float) -> None:
    arguments = {
        "mass1_msun": 1.0e8,
        "mass2_msun": 1.0e8,
        "semimajor_axis_pc": 1.0,
        "eccentricity": 0.3,
        "orbital_power": -1.0,
        "orbital_torque": -1.0,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        keplerian_exchange_rates(**arguments)
