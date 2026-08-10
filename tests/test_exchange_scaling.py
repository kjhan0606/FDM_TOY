import pytest

from fdm_smbh_delay.exchange_scaling import (
    exchange_scales,
    schrodinger_poisson_similarity_parameter,
)


def test_exchange_scales_close_rate_definitions() -> None:
    scales = exchange_scales(
        mass1_msun=1.0e8,
        mass2_msun=5.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=2.0,
    )
    assert scales.orbital_power_msun_pc2_myr3 == pytest.approx(
        scales.orbital_energy_msun_pc2_myr2
        / scales.soliton_dynamical_time_myr
    )
    assert scales.orbital_torque_msun_pc2_myr2 == pytest.approx(
        scales.orbital_angular_momentum_msun_pc2_myr
        / scales.soliton_dynamical_time_myr
    )


def test_exchange_scales_reject_non_positive_inputs() -> None:
    with pytest.raises(ValueError):
        exchange_scales(
            mass1_msun=0.0,
            mass2_msun=1.0,
            soliton_mass_msun=1.0,
            core_radius_pc=1.0,
        )


def test_soliton_similarity_parameter_is_invariant_under_fdm_scaling() -> None:
    reference = schrodinger_poisson_similarity_parameter(
        particle_mass_ev=1.0e-21,
        soliton_mass_msun=1.0006508307763383e9,
        core_radius_pc=2.2,
    )
    rescaled = schrodinger_poisson_similarity_parameter(
        particle_mass_ev=3.0e-22,
        soliton_mass_msun=3.335502769254603e9,
        core_radius_pc=7.333333333333333,
    )
    assert reference == pytest.approx(rescaled)
