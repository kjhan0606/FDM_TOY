from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from fdm_smbh_delay.config import (
    BinaryConfig,
    CaseConfig,
    FDMConfig,
    IntegrationConfig,
    ModelConfig,
)


@pytest.fixture
def case_factory() -> Callable[..., CaseConfig]:
    def build(
        *,
        drag: bool = False,
        mass1: float = 1.0e6,
        mass2: float = 1.0e6,
        soliton_mass: float = 1.0e7,
        core_radius: float = 10.0,
        particle_mass: float = 1.0e-21,
        separation: float = 1.0,
        stop: float = 0.01,
        max_time: float = 0.1,
        output_samples: int = 100,
    ) -> CaseConfig:
        return CaseConfig(
            model=ModelConfig(
                name="wave_df_3d",
                alpha_df=0.341,
                drag=drag,
                fdm_bulk_velocity_pc_myr=np.zeros(3),
                velocity_floor_pc_myr=1.0e-12,
            ),
            binary=BinaryConfig(
                mass1_msun=mass1,
                mass2_msun=mass2,
                separation_pc=separation,
                eccentricity=0.0,
                orbit="circular",
                position1_pc=None,
                position2_pc=None,
                velocity1_pc_myr=None,
                velocity2_pc_myr=None,
            ),
            fdm=FDMConfig(
                particle_mass_ev=particle_mass,
                core_radius_pc=core_radius,
                profile="schive_fit",
                mass_definition="total_profile",
                soliton_mass_msun=soliton_mass,
                central_density_msun_pc3=None,
            ),
            integration=IntegrationConfig(
                stop_separation_pc=stop,
                max_time_myr=max_time,
                output_samples=output_samples,
                rtol=1.0e-10,
                atol=1.0e-12,
                max_step_myr=float("inf"),
                energy_budget_relerr_limit=1.0e-6,
            ),
            raw={"test": True},
        )

    return build
