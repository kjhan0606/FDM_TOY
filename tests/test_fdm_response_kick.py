from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.fdm_orbital_response import project_fdm_response_family_to_orbit
from fdm_smbh_delay.fdm_outer_response import FDMOuterResponseTable
from fdm_smbh_delay.fdm_outer_response_family import FDMOuterResponseFamily
from fdm_smbh_delay.fdm_response_kick import sample_candidate_fdm_velocity_kick


def _projected(*, convention: str = "velocity_covariance_rate"):
    table = FDMOuterResponseTable(
        radii_pc=np.array([10.0, 20.0]),
        drift_acceleration_pc_myr2=np.array([[1.0, -2.0, 0.0]] * 2),
        diffusion_tensor_pc2_myr3=np.array([np.diag([4.0, 9.0, 0.0])] * 2),
        response_status="calibrated", component_frame="orbital_rtn",
        diffusion_convention=convention,
    )
    family = FDMOuterResponseFamily(
        mass_ratios_q=(0.2,), eccentricities=(0.3,),
        tables=((table,),), source_sha256=(("a" * 64,),),
    )
    return project_fdm_response_family_to_orbit(
        family, position_pc=np.array([15.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 1.0, 0.0]),
        mass_ratio_q=0.2, eccentricity=0.3,
    )


def test_candidate_kick_is_restart_stable_and_has_explicit_covariance_rate() -> None:
    projected = _projected()
    first = sample_candidate_fdm_velocity_kick(
        projected, time_step_myr=0.25, random_seed=41, completed_steps=7,
    )
    resumed = sample_candidate_fdm_velocity_kick(
        projected, time_step_myr=0.25, random_seed=41, completed_steps=7,
    )
    assert first.status == "candidate_pending_coherence_and_physical_validation"
    assert first.velocity_increment_pc_myr == pytest.approx(resumed.velocity_increment_pc_myr)
    assert first.deterministic_increment_pc_myr == pytest.approx([0.25, -0.5, 0.0])
    assert first.stochastic_increment_pc_myr[2] == pytest.approx(0.0)
    assert first.source_sha256 == ("a" * 64,)
    assert not np.allclose(first.velocity_increment_pc_myr,
        sample_candidate_fdm_velocity_kick(
            projected, time_step_myr=0.25, random_seed=41, completed_steps=8,
        ).velocity_increment_pc_myr)


def test_candidate_kick_censors_legacy_ambiguous_diffusion() -> None:
    result = sample_candidate_fdm_velocity_kick(
        _projected(convention="unspecified"),
        time_step_myr=0.25, random_seed=41, completed_steps=7,
    )
    assert result.status == "censored"
    assert result.velocity_increment_pc_myr is None
    assert "convention" in result.reason


def test_sampled_kick_covariance_uses_full_rate_not_half_rate() -> None:
    projected = _projected()
    draws = np.array([
        sample_candidate_fdm_velocity_kick(
            projected, time_step_myr=0.25, random_seed=41,
            completed_steps=index,
        ).stochastic_increment_pc_myr
        for index in range(1000)
    ])
    # D_v = diag(4, 9, 0) pc^2/Myr^3; at dt=0.25 Myr the
    # sample covariance should approach diag(1, 2.25, 0) pc^2/Myr^2.
    assert np.diag(np.cov(draws.T)) == pytest.approx([1.0, 2.25, 0.0], rel=0.13, abs=0.02)


@pytest.mark.parametrize("time_step,seed,index", [
    (0.0, 1, 0), (float("nan"), 1, 0), (1.0, -1, 0), (1.0, 1, -1),
])
def test_candidate_kick_rejects_invalid_restart_controls(time_step, seed, index) -> None:
    with pytest.raises(ValueError, match="time step|seed"):
        sample_candidate_fdm_velocity_kick(
            _projected(), time_step_myr=time_step,
            random_seed=seed, completed_steps=index,
        )
