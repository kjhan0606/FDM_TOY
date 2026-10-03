from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import fdm_smbh_delay.binary_evolution as binary_evolution

from fdm_smbh_delay.binary_evolution import (
    BinaryEvolutionConfig,
    BoundBinaryModel,
    BoundBinaryState,
    FDMExchangeRates,
    GasMigrationModel,
    MeanSeparationEstimate,
    StellarHardeningModel,
    UncalibratedBinaryState,
    advance_bound_binary_rk4,
    binary_rate_budget,
    calibrated_qe_fdm_rate_provider,
    find_gw_transition_pc,
    gw_dominates_environment,
    integrate_bound_binary,
    legacy_circular_fdm_rate_provider,
    orbital_invariants,
)
from fdm_smbh_delay.constants import KM_S_TO_PC_MYR
from fdm_smbh_delay.exchange_scaling import schrodinger_poisson_similarity_parameter
from fdm_smbh_delay.orbital_exchange import keplerian_time_mean_separation_pc
from fdm_smbh_delay.subgrid_calibration import (
    InterpolatedSubgridRates,
    SubgridCalibrationRow,
    SubgridCalibrationTable,
)


def _particle_mass_for_similarity(similarity: float, core_radius_pc: float) -> float:
    reference_mass_ev = 1.0e-21
    reference_similarity = schrodinger_poisson_similarity_parameter(
        particle_mass_ev=reference_mass_ev,
        soliton_mass_msun=1.0e9,
        core_radius_pc=core_radius_pc,
    )
    return reference_mass_ev * np.sqrt(reference_similarity / similarity)


def _synthetic_kepler_mean(axis_pc: float, eccentricity: float) -> MeanSeparationEstimate:
    """A zero-width synthetic fixture, not an accepted physical mapping."""

    mean_pc = keplerian_time_mean_separation_pc(axis_pc, eccentricity)
    return MeanSeparationEstimate(mean_pc, mean_pc, mean_pc, "synthetic-kepler")


def _test_release(table: SubgridCalibrationTable, schema: int) -> SubgridCalibrationTable:
    """Mark a synthetic table as release-loaded within this test module only."""

    table.release_schema_version = schema
    table.release_table_sha256 = "a" * 64
    return table


def _synthetic_row(
    *, profile_id: str, similarity: float, mass_ratio: float, eccentricity: float
) -> SubgridCalibrationRow:
    return SubgridCalibrationRow(
        profile_id=profile_id,
        source_case_id="synthetic-test-only",
        schrodinger_poisson_similarity_parameter=similarity,
        binary_to_soliton_mass=0.2 if mass_ratio == 1.0 else 0.13,
        separation_bin_index=0,
        lower_separation_over_core_radius=0.19,
        upper_separation_over_core_radius=0.22,
        reference_mean_separation_over_core_radius=0.209,
        dimensionless_orbital_power=-1.0e-3,
        dimensionless_orbital_torque=-2.0e-3,
        dimensionless_wave_total_energy_rate=1.0e-3,
        orbital_power_spatial_systematic_fraction=0.1,
        orbital_torque_spatial_systematic_fraction=0.1,
        wave_total_spatial_systematic_fraction=0.1,
        reference_resolution=512,
        comparison_resolution=384,
        reference_complete_orbits=12,
        comparison_complete_orbits=12,
        reference_minimum_half_density_radius_over_cell_size=8.0,
        comparison_minimum_half_density_radius_over_cell_size=6.0,
        mass_ratio_q=mass_ratio,
        reference_eccentricity=eccentricity,
    )


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
    assert max(
        sample.energy_closure_relative_error for sample in result.samples
    ) < 1.0e-6
    assert max(
        sample.angular_momentum_closure_relative_error for sample in result.samples
    ) < 1.0e-6


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


def test_missing_environment_does_not_trigger_gw_transition() -> None:
    model = BoundBinaryModel(1.0e8, 1.0e8)
    rates = binary_rate_budget(model, semimajor_axis_pc=1.0, eccentricity_squared=0.0)
    assert not rates.environmental_channels_present
    assert not gw_dominates_environment(rates)
    result = integrate_bound_binary(
        initial_state=BoundBinaryState(0.0, 1.0, 0.0),
        model=model,
        config=BinaryEvolutionConfig(10.0, 0.1, 1.0e-5),
    )
    assert result.status == "censored"
    assert result.environment_fdm_segment.status == "censored"
    assert "no environmental channel" in result.reason


def test_expanding_environment_is_not_relabelled_as_gw_dominated() -> None:
    def expanding(_axis: float, _eccentricity: float) -> FDMExchangeRates:
        return FDMExchangeRates(1.0e-6, 0.0, "expanding-test")

    model = BoundBinaryModel(1.0e8, 1.0e8, fdm_rate_provider=expanding)
    rates = binary_rate_budget(model, semimajor_axis_pc=1.0, eccentricity_squared=0.0)
    assert rates.environmental_semimajor_axis_rate_pc_myr > 0.0
    assert not gw_dominates_environment(rates)
    result = integrate_bound_binary(
        initial_state=BoundBinaryState(0.0, 1.0, 0.0),
        model=model,
        config=BinaryEvolutionConfig(10.0, 0.1, 1.0e-5),
    )
    assert result.status == "stalled"
    assert result.environment_fdm_segment.status == "censored"


def test_legacy_fdm_adapter_rejects_unmeasured_q_and_e() -> None:
    table = SubgridCalibrationTable(
        (_synthetic_row(profile_id="koo", similarity=1.0, mass_ratio=1.0, eccentricity=0.0),)
    )
    with pytest.raises(ValueError, match="verified schema-v2 release"):
        legacy_circular_fdm_rate_provider(
            table,
            profile_id="koo",
            mass1_msun=1.0e8,
            mass2_msun=1.0e8,
            soliton_mass_msun=1.0e9,
            core_radius_pc=5.0,
            particle_mass_ev=_particle_mass_for_similarity(1.0, 5.0),
        )
    table = _test_release(table, 2)

    unequal = legacy_circular_fdm_rate_provider(
        table,
        profile_id="koo",
        mass1_msun=1.0e8,
        mass2_msun=5.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=5.0,
        particle_mass_ev=_particle_mass_for_similarity(1.0, 5.0),
    )
    with pytest.raises(UncalibratedBinaryState, match="mass ratio"):
        unequal(1.0, 0.0)

    circular = legacy_circular_fdm_rate_provider(
        table,
        profile_id="koo",
        mass1_msun=1.0e8,
        mass2_msun=1.0e8,
        soliton_mass_msun=1.0e9,
        core_radius_pc=5.0,
        particle_mass_ev=_particle_mass_for_similarity(1.0, 5.0),
    )
    assert circular(1.0, 0.0).calibration_id == "legacy-v2:koo"
    with pytest.raises(UncalibratedBinaryState, match="eccentricity"):
        circular(1.0, 0.1)


def test_qe_fdm_adapter_passes_runtime_e_and_censors_missing_plane() -> None:
    table = _test_release(
        SubgridCalibrationTable(
            (_synthetic_row(profile_id="boey2025", similarity=1.0, mass_ratio=0.3, eccentricity=0.3),)
        ),
        5,
    )

    provider = calibrated_qe_fdm_rate_provider(
        table,
        profile_id="boey2025",
        mass1_msun=1.0e8,
        mass2_msun=3.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=5.0,
        particle_mass_ev=_particle_mass_for_similarity(1.0, 5.0),
        mean_separation_provider=_synthetic_kepler_mean,
    )
    rates = provider(1.0, 0.3)
    assert rates.calibration_id == (
        f"v5:boey2025:table={'a' * 64}:eta=1:q=0.3:rmap=synthetic-kepler"
    )
    with pytest.raises(UncalibratedBinaryState, match="calibrated range"):
        provider(1.0, 0.2)
    with pytest.raises(UncalibratedBinaryState, match="outside"):
        provider(0.5, 0.3)
    unsupported_similarity = calibrated_qe_fdm_rate_provider(
        table,
        profile_id="boey2025",
        mass1_msun=1.0e8,
        mass2_msun=3.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=5.0,
        particle_mass_ev=2.0 * _particle_mass_for_similarity(1.0, 5.0),
        mean_separation_provider=_synthetic_kepler_mean,
    )
    with pytest.raises(UncalibratedBinaryState, match="similarity parameter"):
        unsupported_similarity(1.0, 0.3)


def test_qe_provider_requires_mapping_and_checks_measured_bin_boundary() -> None:
    core_radius = 5.0
    particle_mass = 1.0e-21
    similarity = schrodinger_poisson_similarity_parameter(
        particle_mass_ev=particle_mass,
        soliton_mass_msun=1.0e9,
        core_radius_pc=core_radius,
    )
    row = SubgridCalibrationRow(
        profile_id="test_soliton",
        source_case_id="qe_measured_e030",
        schrodinger_poisson_similarity_parameter=similarity,
        binary_to_soliton_mass=0.13,
        separation_bin_index=0,
        lower_separation_over_core_radius=0.205,
        upper_separation_over_core_radius=0.215,
        reference_mean_separation_over_core_radius=0.209,
        dimensionless_orbital_power=-1.0e-3,
        dimensionless_orbital_torque=-2.0e-3,
        dimensionless_wave_total_energy_rate=1.0e-3,
        orbital_power_spatial_systematic_fraction=0.1,
        orbital_torque_spatial_systematic_fraction=0.1,
        wave_total_spatial_systematic_fraction=0.1,
        reference_resolution=512,
        comparison_resolution=384,
        reference_complete_orbits=12,
        comparison_complete_orbits=12,
        reference_minimum_half_density_radius_over_cell_size=8.0,
        comparison_minimum_half_density_radius_over_cell_size=6.0,
        mass_ratio_q=0.3,
        reference_eccentricity=0.3,
    )
    table = _test_release(SubgridCalibrationTable((row,)), 5)
    provider_without_mapping = calibrated_qe_fdm_rate_provider(
        table,
        profile_id="test_soliton",
        mass1_msun=1.0e8,
        mass2_msun=3.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=core_radius,
        particle_mass_ev=particle_mass,
    )
    with pytest.raises(UncalibratedBinaryState, match="mapping is unavailable"):
        provider_without_mapping(1.0, 0.3)
    provider = calibrated_qe_fdm_rate_provider(
        table,
        profile_id="test_soliton",
        mass1_msun=1.0e8,
        mass2_msun=3.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=core_radius,
        particle_mass_ev=particle_mass,
        mean_separation_provider=_synthetic_kepler_mean,
    )
    assert provider(1.0, 0.3).orbital_power_msun_pc2_myr3 < 0.0
    with pytest.raises(UncalibratedBinaryState, match="separation"):
        provider(0.95, 0.3)
    with pytest.raises(UncalibratedBinaryState, match="separation"):
        provider(1.05, 0.3)

    def interval(_axis: float, _eccentricity: float) -> MeanSeparationEstimate:
        return MeanSeparationEstimate(1.045, 1.04, 1.06, "synthetic-interval")

    interval_provider = calibrated_qe_fdm_rate_provider(
        table,
        profile_id="test_soliton",
        mass1_msun=1.0e8,
        mass2_msun=3.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=core_radius,
        particle_mass_ev=particle_mass,
        mean_separation_provider=interval,
    )
    assert "rmap=synthetic-interval" in interval_provider(1.0, 0.3).calibration_id
    second_row = replace(
        row,
        separation_bin_index=2,
        lower_separation_over_core_radius=0.225,
        upper_separation_over_core_radius=0.235,
        reference_mean_separation_over_core_radius=0.229,
    )
    gap_table = _test_release(SubgridCalibrationTable((row, second_row)), 5)
    gap_provider = calibrated_qe_fdm_rate_provider(
        gap_table,
        profile_id="test_soliton",
        mass1_msun=1.0e8,
        mass2_msun=3.0e7,
        soliton_mass_msun=1.0e9,
        core_radius_pc=core_radius,
        particle_mass_ev=particle_mass,
        mean_separation_provider=lambda _a, _e: MeanSeparationEstimate(
            1.045, 1.04, 1.15, "synthetic-cross-gap"
        ),
    )
    with pytest.raises(UncalibratedBinaryState, match="gap"):
        gap_provider(1.0, 0.3)
    result = integrate_bound_binary(
        initial_state=BoundBinaryState(0.0, 1.0, 0.3**2),
        model=BoundBinaryModel(1.0e8, 3.0e7, fdm_rate_provider=gap_provider),
        config=BinaryEvolutionConfig(10.0, 0.1, 0.01),
    )
    assert result.status == "uncalibrated"
    assert result.environment_fdm_segment.status == "censored"


def test_qe_provider_rejects_unstructured_mapping_and_invalid_envelope() -> None:
    with pytest.raises(ValueError, match="invalid"):
        MeanSeparationEstimate(1.0, 1.1, 1.2, "synthetic")
    with pytest.raises(ValueError, match="invalid"):
        MeanSeparationEstimate(1.0, 1.0, np.inf, "synthetic")
    with pytest.raises(ValueError, match="invalid"):
        MeanSeparationEstimate(1.0, 1.0, 1.0, "")
    with pytest.raises(ValueError, match="invalid"):
        MeanSeparationEstimate(True, 1.0, 1.0, "synthetic")
    with pytest.raises(ValueError, match="verified schema-v5 release"):
        calibrated_qe_fdm_rate_provider(
            SubgridCalibrationTable(
                (_synthetic_row(profile_id="test", similarity=1.0, mass_ratio=1.0, eccentricity=0.0),)
            ),
            profile_id="test",
            mass1_msun=1.0e8,
            mass2_msun=1.0e8,
            soliton_mass_msun=1.0e9,
            core_radius_pc=5.0,
            particle_mass_ev=1.0e-21,
        )
    table = _test_release(
        SubgridCalibrationTable(
            (_synthetic_row(profile_id="test", similarity=1.0, mass_ratio=1.0, eccentricity=0.0),)
        ),
        5,
    )
    provider = calibrated_qe_fdm_rate_provider(
        table,
        profile_id="test",
        mass1_msun=1.0e8,
        mass2_msun=1.0e8,
        soliton_mass_msun=1.0e9,
        core_radius_pc=5.0,
        particle_mass_ev=1.0e-21,
        mean_separation_provider=lambda _a, _e: 1.0,
    )
    with pytest.raises(UncalibratedBinaryState, match="validated interval"):
        provider(1.0, 0.0)


def test_unsupported_rk_endpoint_is_not_accepted_or_checkpointed(monkeypatch) -> None:
    def bounded_rate(axis_pc: float, _eccentricity: float) -> FDMExchangeRates:
        if axis_pc < 0.9:
            raise UncalibratedBinaryState("outside measured axis support")
        return FDMExchangeRates(-1.0, -1.0, "synthetic-bounded")

    model = BoundBinaryModel(1.0e8, 1.0e8, fdm_rate_provider=bounded_rate)
    initial = BoundBinaryState(0.0, 1.0, 0.3**2)

    def unsupported_endpoint(state, _model, time_step):
        return replace(
            state,
            elapsed_myr=state.elapsed_myr + time_step,
            semimajor_axis_pc=0.89,
            completed_steps=state.completed_steps + 1,
        )

    monkeypatch.setattr(
        binary_evolution, "advance_bound_binary_rk4", unsupported_endpoint
    )
    result = integrate_bound_binary(
        initial_state=initial,
        model=model,
        config=BinaryEvolutionConfig(
            target_semimajor_axis_pc=0.01,
            maximum_time_myr=1.0,
            maximum_step_myr=0.001,
            sample_interval_steps=2,
            stop_at_gw_transition=False,
        ),
        step_budget=1,
    )
    assert result.status == "uncalibrated"
    assert result.final_state.semimajor_axis_pc == initial.semimajor_axis_pc
    assert result.samples[-1].semimajor_axis_pc == pytest.approx(1.0)
    assert "outside measured axis support" in result.reason


def test_rate_budget_and_restart_preserve_fdm_calibration_identity() -> None:
    def rate_a(_axis: float, _eccentricity: float) -> FDMExchangeRates:
        return FDMExchangeRates(-1.0e8, -1.0e6, "release-a:rmap=map-a")

    def rate_b(_axis: float, _eccentricity: float) -> FDMExchangeRates:
        return FDMExchangeRates(-1.0e8, -1.0e6, "release-a:rmap=map-b")

    model_a = BoundBinaryModel(1.0e8, 1.0e8, fdm_rate_provider=rate_a)
    budget = binary_rate_budget(model_a, semimajor_axis_pc=1.0, eccentricity_squared=0.0)
    assert budget.fdm_calibration_id == "release-a:rmap=map-a"
    config = BinaryEvolutionConfig(1.0, 1.0e-4, 0.01, stop_at_gw_transition=False)
    checkpoint = integrate_bound_binary(
        initial_state=BoundBinaryState(0.0, 1.0, 0.0),
        model=model_a,
        config=config,
        step_budget=1,
    )
    assert checkpoint.status == "checkpoint"
    assert checkpoint.final_state.fdm_calibration_id == "release-a:rmap=map-a"
    with pytest.raises(ValueError, match="calibration identity"):
        integrate_bound_binary(
            initial_state=checkpoint.final_state,
            model=BoundBinaryModel(1.0e8, 1.0e8, fdm_rate_provider=rate_b),
            config=config,
        )


@pytest.mark.parametrize(
    ("reservoir", "reason"),
    [
        ("extracted_energy_by_channel", "energy closure"),
        ("extracted_angular_momentum_by_channel", "angular-momentum closure"),
    ],
)
def test_restart_closure_baseline_is_not_reset(
    reservoir: str, reason: str
) -> None:
    model = _stellar_model()
    config = BinaryEvolutionConfig(1.0, 1.0e-4, 0.01, stop_at_gw_transition=False)
    checkpoint = integrate_bound_binary(
        initial_state=BoundBinaryState(0.0, 1.0, 0.0),
        model=model,
        config=config,
        step_budget=1,
    ).final_state
    corrupted = replace(checkpoint, **{reservoir: (1.0e30, 0.0, 0.0, 0.0)})
    result = integrate_bound_binary(initial_state=corrupted, model=model, config=config)
    assert result.status == "invalid"
    assert reason in result.reason
    assert result.environment_fdm_segment.status == "invalid"


def test_unsampled_candidate_must_pass_closure_gate(monkeypatch) -> None:
    model = _stellar_model()

    def corrupt_step(state, _model, time_step):
        return replace(
            state,
            elapsed_myr=state.elapsed_myr + time_step,
            completed_steps=state.completed_steps + 1,
            extracted_energy_by_channel=(1.0e30, 0.0, 0.0, 0.0),
        )

    monkeypatch.setattr(binary_evolution, "advance_bound_binary_rk4", corrupt_step)
    result = integrate_bound_binary(
        initial_state=BoundBinaryState(0.0, 1.0, 0.0),
        model=model,
        config=BinaryEvolutionConfig(
            1.0, 1.0e-4, 0.01, sample_interval_steps=100, stop_at_gw_transition=False
        ),
        step_budget=1,
    )
    assert result.status == "invalid"
    assert "energy closure gate failed" in result.reason
    assert result.final_state.completed_steps == 0
