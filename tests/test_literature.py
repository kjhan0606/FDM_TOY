from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.constants import KM_S_TO_PC_MYR
from fdm_smbh_delay.literature import (
    boey2025_schive_soliton,
    koo2024_equivalent_schive_soliton,
)
from fdm_smbh_delay.orbit import make_orbital_state


def test_koo_profile_matches_quoted_mass_and_density() -> None:
    soliton = koo2024_equivalent_schive_soliton()
    assert soliton.central_density_msun_pc3 == pytest.approx(7.05e6)
    assert soliton.total_mass_msun == pytest.approx(1.0e9)
    assert soliton.core_radius_pc == pytest.approx(2.304728369104316)


def test_boey_profile_and_initial_speed_match_fiducial() -> None:
    soliton = boey2025_schive_soliton()
    assert soliton.central_density_msun_pc3 == pytest.approx(8.110784782460214e6)
    assert soliton.total_mass_msun == pytest.approx(1.0e9, rel=7.0e-4)
    state = make_orbital_state(
        mass1_msun=1.0e8,
        mass2_msun=1.0e8,
        separation_pc=3.0,
        eccentricity=0.0,
        soliton=soliton,
    )
    speed_km_s = np.linalg.norm(state.velocity1_pc_myr) / KM_S_TO_PC_MYR
    assert speed_km_s == pytest.approx(584.14, rel=1.0e-3)
