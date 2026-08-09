from __future__ import annotations

import numpy as np

from fdm_smbh_delay.orbit import integrate_case


def test_no_drag_orbit_conserves_energy_and_com(case_factory) -> None:
    result = integrate_case(case_factory(drag=False, max_time=0.1, output_samples=150))
    assert result.summary["max_energy_budget_relerr"] < 1.0e-8
    assert np.max(np.abs(result.timeseries["E_to_fdm"])) == 0.0
    m1 = m2 = 1.0e6
    com_x = (m1 * result.timeseries["r1_x_pc"] + m2 * result.timeseries["r2_x_pc"]) / (m1 + m2)
    com_y = (m1 * result.timeseries["r1_y_pc"] + m2 * result.timeseries["r2_y_pc"]) / (m1 + m2)
    assert np.max(np.hypot(com_x, com_y)) < 1.0e-12


def test_drag_energy_and_wave_exchange_ledgers_close(case_factory) -> None:
    case = case_factory(
        drag=True,
        mass1=5.0e7,
        mass2=5.0e7,
        soliton_mass=1.0e9,
        core_radius=2.0,
        max_time=0.01,
        output_samples=80,
    )
    result = integrate_case(case)
    assert result.timeseries["E_to_fdm"][-1] > 0.0
    assert result.timeseries["E_to_fdm_rest"][-1] > 0.0
    assert result.summary["max_energy_budget_relerr"] < 1.0e-6
    assert np.allclose(result.summary["P_to_fdm"], 0.0, atol=1.0e-7)


def test_inward_bound_target_crossing_is_detected(case_factory) -> None:
    case = case_factory(
        drag=True,
        mass1=5.0e7,
        mass2=5.0e7,
        soliton_mass=1.0e9,
        core_radius=2.0,
        stop=0.995,
        max_time=0.05,
        output_samples=30,
    )
    result = integrate_case(case)
    assert result.summary["status"] == "reached_target"
    assert result.summary["t_fdm_myr"] is not None
    assert result.summary["D_final_pc"] <= 0.995000001
