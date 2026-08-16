from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.environmental_friction import (
    fdm_dynamical_friction,
    gaseous_dynamical_friction,
    maxwellian_slow_fraction,
    ostriker_gas_coefficient,
    stellar_dynamical_friction,
)
from fdm_smbh_delay.galaxy_environment import (
    FDMBackground,
    GasBackground,
    DehnenProfile,
    StellarBackground,
)
from fdm_smbh_delay.soliton import SchiveSoliton


def test_stellar_friction_opposes_relative_motion() -> None:
    profile = DehnenProfile(1.0e10, 500.0, 1.0)
    background = StellarBackground(profile, 100.0, np.zeros(3))
    velocity = np.array([0.0, 150.0, 0.0])
    result = stellar_dynamical_friction(
        perturber_mass_msun=1.0e8,
        position_pc=np.array([100.0, 0.0, 0.0]),
        velocity_pc_myr=velocity,
        background=background,
        minimum_impact_parameter_pc=0.1,
    )
    assert result.acceleration_pc_myr2 @ velocity < 0.0
    assert result.coefficient > 0.0
    assert 0.0 < maxwellian_slow_fraction(150.0, 100.0) < 1.0


def test_explicit_core_stalling_bracket_suppresses_stellar_force() -> None:
    profile = DehnenProfile(1.0e8, 1000.0, 0.0)
    background = StellarBackground(profile, 100.0, np.zeros(3))
    result = stellar_dynamical_friction(
        perturber_mass_msun=1.0e8,
        position_pc=np.array([10.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 100.0, 0.0]),
        background=background,
        minimum_impact_parameter_pc=0.1,
        core_stall_enclosed_mass_ratio=1.0,
    )
    assert result.stalled
    assert np.allclose(result.acceleration_pc_myr2, 0.0)


def test_ostriker_coefficient_and_gas_force_are_dissipative() -> None:
    assert ostriker_gas_coefficient(0.0, 3.0) == 0.0
    assert ostriker_gas_coefficient(0.5, 3.0) > 0.0
    assert ostriker_gas_coefficient(2.0, 3.0) > 0.0
    profile = DehnenProfile(1.0e9, 500.0, 1.0)
    background = GasBackground(profile, sound_speed_pc_myr=50.0)
    velocity = np.array([0.0, 100.0, 0.0])
    result = gaseous_dynamical_friction(
        perturber_mass_msun=1.0e8,
        position_pc=np.array([100.0, 0.0, 0.0]),
        velocity_pc_myr=velocity,
        total_enclosed_mass_msun=1.0e10,
        background=background,
        coulomb_logarithm=3.0,
    )
    assert result.acceleration_pc_myr2 @ result.relative_velocity_pc_myr < 0.0


def test_live_fdm_wake_disables_analytic_drag() -> None:
    soliton = SchiveSoliton.from_mass(1.0e9, 100.0, "total_profile")
    common = dict(
        soliton=soliton,
        particle_mass_ev=1.0e-21,
        alpha_df=0.341,
        bulk_velocity_pc_myr=np.zeros(3),
    )
    live = fdm_dynamical_friction(
        perturber_mass_msun=1.0e8,
        position_pc=np.array([50.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 100.0, 0.0]),
        background=FDMBackground(**common, mode="live_resolved"),
    )
    analytic = fdm_dynamical_friction(
        perturber_mass_msun=1.0e8,
        position_pc=np.array([50.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 100.0, 0.0]),
        background=FDMBackground(**common, mode="analytic_unresolved"),
    )
    assert live.live_wake_resolved and not live.analytic_applied
    assert np.allclose(live.acceleration_pc_myr2, 0.0)
    assert analytic.analytic_applied
    assert analytic.acceleration_pc_myr2 @ np.array([0.0, 100.0, 0.0]) < 0.0
