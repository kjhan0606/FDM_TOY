from __future__ import annotations

import numpy as np
import pytest
from scipy.special import sici

from fdm_smbh_delay.soliton import SchiveSoliton
from fdm_smbh_delay.wave_drag import drag_coefficient, evaluate_drag


def test_small_q_series() -> None:
    q = np.array([0.0, 1.0e-8, 1.0e-5, 1.0e-3])
    expected = q**2 / 3.0 - q**4 / 30.0 + 2.0 * q**6 / 945.0
    assert np.allclose(drag_coefficient(q), expected, rtol=1.0e-15, atol=0.0)


def test_series_overlaps_full_expression() -> None:
    q = 0.02
    x = 2.0 * q
    _, ci = sici(x)
    full = np.euler_gamma + np.log(x) - ci + np.sin(x) / x - 1.0
    series = q**2 / 3.0 - q**4 / 30.0 + 2.0 * q**6 / 945.0
    assert full == pytest.approx(series, rel=1.0e-9)
    assert drag_coefficient(q) == pytest.approx(full, rel=1.0e-14)


def test_drag_opposes_fdm_relative_velocity() -> None:
    soliton = SchiveSoliton.from_mass(1.0e9, 2.0, "total_profile")
    velocity = np.array([10.0, -2.0, 1.0])
    result = evaluate_drag(
        mass_msun=1.0e8,
        position_pc=np.array([0.5, 0.0, 0.0]),
        velocity_pc_myr=velocity,
        separation_pc=1.0,
        soliton=soliton,
        m_fdm_ev=1.0e-21,
        alpha_df=0.341,
        bulk_velocity_pc_myr=np.zeros(3),
        velocity_floor_pc_myr=1.0e-12,
    )
    assert result.force_msun_pc_myr2 @ velocity < 0.0
    assert np.linalg.norm(np.cross(result.force_msun_pc_myr2, velocity)) < 1.0e-6
