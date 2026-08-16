from __future__ import annotations

import pytest

from fdm_smbh_delay.calibration import (
    component_masses,
    designed_parameter_cases,
    exchange_calibration_eligibility,
    literature_anchor_cases,
    run_specifications,
    structured_parameter_cases,
)


def test_component_masses_preserve_total_and_q() -> None:
    mass1, mass2 = component_masses(
        soliton_mass_msun=1.0e9,
        binary_to_soliton_mass=0.1,
        mass_ratio_q=0.3,
    )
    assert mass1 + mass2 == pytest.approx(1.0e8)
    assert mass2 / mass1 == pytest.approx(0.3)


def test_exchange_calibration_eligibility_separates_phase_selection() -> None:
    secular_only = exchange_calibration_eligibility(
        before_first_underresolved_orbit=True,
        initial_resolved_energy_conservation_passed=True,
        half_density_radius_spatially_resolved=True,
        wave_mode_time_offset_over_orbital_period=0.75,
    )
    assert secular_only.secular
    assert not secular_only.phase_dependent

    phase_eligible = exchange_calibration_eligibility(
        before_first_underresolved_orbit=True,
        initial_resolved_energy_conservation_passed=True,
        half_density_radius_spatially_resolved=True,
        wave_mode_time_offset_over_orbital_period=-0.5,
    )
    assert phase_eligible.secular
    assert phase_eligible.phase_dependent


@pytest.mark.parametrize(
    "failed_requirement",
    [
        "before_first_underresolved_orbit",
        "initial_resolved_energy_conservation_passed",
        "half_density_radius_spatially_resolved",
    ],
)
def test_exchange_calibration_eligibility_requires_all_secular_checks(
    failed_requirement: str,
) -> None:
    arguments = {
        "before_first_underresolved_orbit": True,
        "initial_resolved_energy_conservation_passed": True,
        "half_density_radius_spatially_resolved": True,
        "wave_mode_time_offset_over_orbital_period": 0.0,
    }
    arguments[failed_requirement] = False
    eligibility = exchange_calibration_eligibility(**arguments)
    assert not eligibility.secular
    assert not eligibility.phase_dependent


def test_exchange_calibration_eligibility_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        exchange_calibration_eligibility(
            before_first_underresolved_orbit=True,
            initial_resolved_energy_conservation_passed=True,
            half_density_radius_spatially_resolved=True,
            wave_mode_time_offset_over_orbital_period=0.0,
            maximum_wave_mode_time_offset_over_orbital_period=float("nan"),
        )


def test_literature_anchor_initial_conditions() -> None:
    anchors = literature_anchor_cases()
    assert len(anchors) == 6
    boey = next(case for case in anchors if case.case_id == "boey_each10pct")
    assert boey.initial_separation_pc == pytest.approx(3.0)
    assert boey.mass1_msun == pytest.approx(0.1 * boey.soliton_mass_msun)
    assert boey.target_duration_myr == pytest.approx(0.8)
    assert boey.schrodinger_poisson_similarity_parameter == pytest.approx(
        0.3881923084
    )


def test_structured_grid_tiers_and_no_analytic_drag() -> None:
    cases = structured_parameter_cases(
        q_values=[1.0, 0.3, 0.1],
        eccentricities=[0.0, 0.3, 0.6],
        binary_mass_fractions=[0.04, 0.1, 0.2],
        semi_major_axis_over_core=[0.1, 0.4, 1.36],
        fiducial=(1.0, 0.0, 0.1, 0.4),
    )
    assert len(cases) == 81
    assert sum(case.tier == 1 for case in cases) == 9
    runs = run_specifications(
        cases[:1],
        box_over_core_radius=45.4545454545,
        resolutions_by_tier={0: [512], 1: [512], 2: [1024]},
    )
    assert len(runs) == 1
    assert runs[0].analytic_fdm_drag is False


def test_explicit_qe_design_preserves_apocentre_and_small_axis() -> None:
    cases = designed_parameter_cases(
        [
            {
                "case_id": "qe_q030_e030_a005",
                "tier": 2,
                "mass_ratio_q": 0.3,
                "eccentricity": 0.3,
                "binary_to_soliton_mass": 0.1,
                "semi_major_axis_over_core_radius": 0.05,
            }
        ]
    )
    assert len(cases) == 1
    case = cases[0]
    assert case.mass2_msun / case.mass1_msun == pytest.approx(0.3)
    assert case.initial_separation_pc == pytest.approx(
        case.semi_major_axis_pc * 1.3
    )
    assert case.semi_major_axis_over_core_radius == pytest.approx(0.05)
