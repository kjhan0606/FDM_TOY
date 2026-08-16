from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.binary_evolution import (
    BinaryEvolutionConfig,
    BoundBinaryModel,
    BoundBinaryState,
    GasMigrationModel,
    StellarHardeningModel,
    UncalibratedBinaryState,
    advance_bound_binary_rk4,
    binary_rate_budget,
    calibrated_qe_fdm_rate_provider,
    find_gw_transition_pc,
    integrate_bound_binary,
    legacy_circular_fdm_rate_provider,
    orbital_invariants,
)
from fdm_smbh_delay.constants import KM_S_TO_PC_MYR
from fdm_smbh_delay.subgrid_calibration import InterpolatedSubgridRates


def _stellar_model() -> BoundBinaryModel:
    return BoundBinaryModel(
        1.0e8,
        1.0e8,
        stellar=StellarHardeningModel(
            density_msun_pc3=1.0e4,
            velocity_dispersion_pc_myr=200.0 * KM_S_TO_PC_MYR,
            hardening_coefficient=15.0,
            eccentricity_growth_coefficient=0.1,
        ),
    )


def test_stellar_hardening_obeys_inverse_axis_law() -> None:
    model = _stellar_model()
    axis = 1.5
    rates = binary_rate_budget(model, semimajor_axis_pc=axis, eccentricity_squared=0.04)
    expected = model.stellar.inverse_semimajor_axis_rate_per_pc_myr
    assert -rates.stellar.semimajor_axis_rate_pc_myr / axis**2 == pytest.approx(
        expected
    )
    assert rates.stellar.eccentricity_squared_rate_per_myr > 0.0


def test_finite_step_closes_energy_and_angular_momentum() -> None:
    model = BoundBinaryModel(
        1.0e8,
        5.0e7,
        gas=GasMigrationModel(100.0, eccentricity_damping_timescale_myr=20.0),
    )
    initial = BoundBinaryState(0.0, 0.5, 0.3**2)
    initial_energy, initial_angular_momentum = orbital_invariants(model, 0.5, 0.3**2)
    final = advance_bound_binary_rk4(initial, model, 1.0e-4)
    final_energy, final_angular_momentum = orbital_invariants(
        model, final.semimajor_axis_pc, final.eccentricity_squared
    )
    assert final_energy + sum(final.extracted_energy_by_channel) == pytest.approx(
        initial_energy, rel=2.0e-15
    )
    assert final_angular_momentum + sum(
        final.extracted_angular_momentum_by_channel
    ) == pytest.approx(initial_angular_momentum, rel=2.0e-15)
    assert final.eccentricity < initial.eccentricity


def test_fixed_e_gw_transition_is_event_specific() -> None:
    model = _stellar_model()
    transition_circular = find_gw_transition_pc(
        model,
        eccentricity=0.0,
        minimum_semimajor_axis_pc=1.0e-5,
        maximum_semimajor_axis_pc=10.0,
    )
    transition_eccentric = find_gw_transition_pc(
        model,
        eccentricity=0.7,
        minimum_semimajor_axis_pc=1.0e-5,
        maximum_semimajor_axis_pc=10.0,
    )
    assert transition_circular is not None
    assert transition_eccentric is not None
    assert 1.0e-5 < transition_circular < 1.0
    assert transition_eccentric > transition_circular


def test_binary_evolution_stops_at_gw_transition_and_returns_peters_delay() -> None:
    model = _stellar_model()
    result = integrate_bound_binary(
        initial_state=BoundBinaryState(0.0, 1.0, 0.0),
        model=model,
        config=BinaryEvolutionConfig(
            maximum_time_myr=5000.0,
            maximum_step_myr=1.0,
            target_semimajor_axis_pc=1.0e-5,
            timestep_fraction=0.02,
        ),
    )
    assert result.status == "gw_transition"
    assert result.gw_completion_delay_myr is not None
    assert result.gravitational_wave_segment.status == "complete"
    assert result.environment_fdm_segment.status == "complete"
    assert max(abs(sample.energy_closure_error) for sample in result.samples) < 1.0
    assert max(
        abs(sample.angular_momentum_closure_error) for sample in result.samples
    ) < 1.0


def test_binary_resume_matches_uninterrupted_solution() -> None:
    model = _stellar_model()
    config = BinaryEvolutionConfig(
        maximum_time_myr=0.1,
        maximum_step_myr=0.001,
        target_semimajor_axis_pc=1.0e-5,
        stop_at_gw_transition=False,
    )
    initial = BoundBinaryState(0.0, 1.0, 0.2**2)
    uninterrupted = integrate_bound_binary(
        initial_state=initial, model=model, config=config
    )
    partial = integrate_bound_binary(
        initial_state=initial, model=model, config=config, step_budget=7
    )
    assert partial.status == "checkpoint"
    assert partial.environment_fdm_segment.status == "missing"
    resumed = integrate_bound_binary(
        initial_state=partial.final_state, model=model, config=config
    )
    assert resumed.status == uninterrupted.status == "timeout"
    assert resumed.final_state == uninterrupted.final_state


def test_uncalibrated_fdm_state_is_not_silently_extrapolated() -> None:
    def unavailable(_axis: float, _eccentricity: float):
        raise UncalibratedBinaryState("outside accepted q-e-separation domain")

    model = BoundBinaryModel(1.0e8, 1.0e8, fdm_rate_provider=unavailable)
    result = integrate_bound_binary(
        initial_state=BoundBinaryState(0.0, 1.0, 0.0),
        model=model,
        config=BinaryEvolutionConfig(10.0, 0.1, 0.01),
    )
    assert result.status == "uncalibrated"
    assert result.environment_fdm_segment.status == "censored"
    assert result.environment_fdm_segment.reason == result.reason


def test_legacy_fdm_adapter_rejects_unmeasured_q_and_e() -> None:
    class OnePointTable:
        def interpolate(self, **_coordinates):
            return InterpolatedSubgridRates(
                profile_id="koo",
                schrodinger_poisson_similarity_parameter=1.0,
                binary_to_soliton_mass=0.2,
                separation_over_core_radius=0.2,
                dimensionless_orbital_power=-1.0e-3,
                dimensionless_orbital_torque=-1.0e-3,
                dimensionless_wave_total_energy_rate=1.0e-3,
                orbital_power_spatial_systematic_fraction=0.1,
                orbital_torque_spatial_systematic_fraction=0.1,
                wave_total_spatial_systematic_fraction=0.1,
            )

    unequal = legacy_circular_fdm_rate_provider(
        OnePointTable(),
        profile_id="koo",
        mass1_msun=1.0e8,
        mass2_msun=5.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=5.0,
    )
    with pytest.raises(UncalibratedBinaryState, match="mass ratio"):
        unequal(1.0, 0.0)

    circular = legacy_circular_fdm_rate_provider(
        OnePointTable(),
        profile_id="koo",
        mass1_msun=1.0e8,
        mass2_msun=1.0e8,
        soliton_mass_msun=1.0e9,
        core_radius_pc=5.0,
    )
    assert circular(1.0, 0.0).calibration_id == "legacy-v2:koo"
    with pytest.raises(UncalibratedBinaryState, match="eccentricity"):
        circular(1.0, 0.1)


def test_qe_fdm_adapter_passes_runtime_e_and_censors_missing_plane() -> None:
    class ExactPlaneTable:
        def interpolate(self, **coordinates):
            if coordinates["mass_ratio_q"] != pytest.approx(0.3):
                raise ValueError("requested mass-ratio/eccentricity plane is absent")
            if coordinates["eccentricity"] != pytest.approx(0.3):
                raise ValueError("requested mass-ratio/eccentricity plane is absent")
            if coordinates["separation_over_core_radius"] != pytest.approx(0.2):
                raise ValueError("separation lies outside the calibrated range")
            return InterpolatedSubgridRates(
                profile_id="boey2025",
                schrodinger_poisson_similarity_parameter=1.0,
                binary_to_soliton_mass=0.13,
                separation_over_core_radius=0.2,
                dimensionless_orbital_power=-1.0e-3,
                dimensionless_orbital_torque=-2.0e-3,
                dimensionless_wave_total_energy_rate=1.0e-3,
                orbital_power_spatial_systematic_fraction=0.1,
                orbital_torque_spatial_systematic_fraction=0.1,
                wave_total_spatial_systematic_fraction=0.1,
                mass_ratio_q=0.3,
                reference_eccentricity=0.3,
            )

    provider = calibrated_qe_fdm_rate_provider(
        ExactPlaneTable(),
        profile_id="boey2025",
        mass1_msun=1.0e8,
        mass2_msun=3.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=5.0,
    )
    rates = provider(1.0, 0.3)
    assert rates.calibration_id == "v3:boey2025:q=0.3:e=0.3"
    with pytest.raises(UncalibratedBinaryState, match="plane is absent"):
        provider(1.0, 0.2)
    with pytest.raises(UncalibratedBinaryState, match="outside"):
        provider(0.5, 0.3)
