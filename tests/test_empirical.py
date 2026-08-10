from __future__ import annotations

import pytest

from fdm_smbh_delay.empirical import (
    BOEY_2025_FITS,
    koo_q0_pc_m5half_myr,
    koo_separation_pc,
    koo_time_between_myr,
)


def test_koo_curve_inverse() -> None:
    q0 = 0.7
    elapsed = koo_time_between_myr(1.0, 0.2, q0)
    assert koo_separation_pc(elapsed, 1.0, q0) == pytest.approx(0.2, rel=1.0e-13)


def test_boey_public_fit_replay_and_ordering() -> None:
    delays = {
        ratio: fit.time_between_myr(1.0, 0.076)
        for ratio, fit in BOEY_2025_FITS.items()
    }
    assert delays[10] == pytest.approx(5.631046309858295, rel=1.0e-14)
    assert delays[10] < delays[5] < delays[2]
    for ratio, fit in BOEY_2025_FITS.items():
        time_at_one_pc = (
            (fit.a_pc / 1.0) ** (1.0 / fit.c) - 1.0
        ) / fit.b_per_myr
        assert fit.separation_pc(time_at_one_pc + delays[ratio]) == pytest.approx(
            0.076, rel=1.0e-13
        )


def test_koo_equation18_effective_mass() -> None:
    q0 = koo_q0_pc_m5half_myr(
        soliton_mass_msun=1.0e9,
        black_hole_mass_msun=1.0e8,
        particle_mass_ev=1.0e-21,
    )
    assert q0 == pytest.approx(1.324 * (1.0 + 2.0 * 2.192 * 0.1) ** 4)
