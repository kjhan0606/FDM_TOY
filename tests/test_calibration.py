from __future__ import annotations

import pytest

from fdm_smbh_delay.calibration import (
    component_masses,
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


def test_literature_anchor_initial_conditions() -> None:
    anchors = literature_anchor_cases()
    assert len(anchors) == 6
    boey = next(case for case in anchors if case.case_id == "boey_each10pct")
    assert boey.initial_separation_pc == pytest.approx(3.0)
    assert boey.mass1_msun == pytest.approx(0.1 * boey.soliton_mass_msun)
    assert boey.target_duration_myr == pytest.approx(0.8)


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
