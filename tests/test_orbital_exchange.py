import pytest

from fdm_smbh_delay.orbital_exchange import keplerian_exchange_rates


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
