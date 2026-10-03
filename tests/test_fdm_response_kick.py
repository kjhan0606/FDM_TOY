from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fdm_smbh_delay.fdm_orbital_response import project_fdm_response_family_to_orbit
from fdm_smbh_delay.fdm_outer_response import FDMOuterResponseTable
from fdm_smbh_delay.fdm_outer_response_family import FDMOuterResponseFamily
from fdm_smbh_delay.fdm_outer_halo import FDMOuterHaloClosure
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


def _closures(*, coherence_time_myr: float = 0.01, status: str = "calibrated"):
    closure = FDMOuterHaloClosure(
        radii_pc=np.array([10.0, 20.0]),
        mass_current_msun_pc2_myr=np.zeros((2, 3)),
        coherence_time_myr=np.array([coherence_time_myr] * 2),
        de_broglie_wavelength_pc=np.array([1.0, 1.0]),
        velocity_diffusion_pc2_myr3=np.zeros(2),
        density_gradient_scale_pc=np.array([5.0, 5.0]),
        closure_status=status,
    )
    return {"a" * 64: closure}


def test_candidate_kick_is_restart_stable_and_has_explicit_covariance_rate() -> None:
    projected = _projected()
    first = sample_candidate_fdm_velocity_kick(
        projected, time_step_myr=0.25, random_seed=41, completed_steps=7,
        closure_by_source_sha256=_closures(),
    )
    resumed = sample_candidate_fdm_velocity_kick(
        projected, time_step_myr=0.25, random_seed=41, completed_steps=7,
        closure_by_source_sha256=_closures(),
    )
    assert first.status == "candidate_pending_physical_validation"
    assert first.maximum_coherence_time_myr == pytest.approx(0.01)
    assert first.velocity_increment_pc_myr == pytest.approx(resumed.velocity_increment_pc_myr)
    assert first.deterministic_increment_pc_myr == pytest.approx([0.25, -0.5, 0.0])
    assert first.stochastic_increment_pc_myr[2] == pytest.approx(0.0)
    assert first.source_sha256 == ("a" * 64,)
    assert not np.allclose(first.velocity_increment_pc_myr,
        sample_candidate_fdm_velocity_kick(
            projected, time_step_myr=0.25, random_seed=41, completed_steps=8,
            closure_by_source_sha256=_closures(),
        ).velocity_increment_pc_myr)


def test_candidate_kick_censors_legacy_ambiguous_diffusion() -> None:
    result = sample_candidate_fdm_velocity_kick(
        _projected(convention="unspecified"),
        time_step_myr=0.25, random_seed=41, completed_steps=7,
        closure_by_source_sha256=_closures(),
    )
    assert result.status == "censored"
    assert result.velocity_increment_pc_myr is None
    assert "convention" in result.reason


def test_sampled_kick_covariance_uses_full_rate_not_half_rate() -> None:
    projected = _projected()
    closures = _closures()
    draws = np.array([
        sample_candidate_fdm_velocity_kick(
            projected, time_step_myr=0.25, random_seed=41,
            completed_steps=index,
            closure_by_source_sha256=closures,
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
            closure_by_source_sha256=_closures(),
        )


def test_candidate_kick_censors_missing_uncalibrated_or_unsupported_coherence() -> None:
    projected = _projected()
    for closures, reason in (
        ({}, "lacks a measured coherence"),
        (_closures(status="uncalibrated"), "not calibrated"),
        (_closures(coherence_time_myr=0.03), "colored-noise model"),
    ):
        result = sample_candidate_fdm_velocity_kick(
            projected, time_step_myr=0.25, random_seed=41,
            completed_steps=7, closure_by_source_sha256=closures,
        )
        assert result.status == "censored"
        assert result.velocity_increment_pc_myr is None
        assert reason in result.reason

    narrow = _closures()["a" * 64]
    narrow = FDMOuterHaloClosure(
        radii_pc=np.array([11.0, 14.0]),
        mass_current_msun_pc2_myr=narrow.mass_current_msun_pc2_myr,
        coherence_time_myr=narrow.coherence_time_myr,
        de_broglie_wavelength_pc=narrow.de_broglie_wavelength_pc,
        velocity_diffusion_pc2_myr3=narrow.velocity_diffusion_pc2_myr3,
        density_gradient_scale_pc=narrow.density_gradient_scale_pc,
        closure_status="calibrated",
    )
    result = sample_candidate_fdm_velocity_kick(
        projected, time_step_myr=0.25, random_seed=41,
        completed_steps=7, closure_by_source_sha256={"a" * 64: narrow},
    )
    assert result.status == "censored"
    assert "outside tabulated support" in result.reason


def test_qe_corner_coherence_gate_uses_slowest_source() -> None:
    projected = replace(_projected(), source_sha256=("a" * 64, "b" * 64))
    closures = {
        **_closures(coherence_time_myr=0.01),
        "b" * 64: _closures(coherence_time_myr=0.03)["a" * 64],
    }
    rejected = sample_candidate_fdm_velocity_kick(
        projected, time_step_myr=0.25, random_seed=41,
        completed_steps=7, closure_by_source_sha256=closures,
    )
    assert rejected.status == "censored"
    accepted = sample_candidate_fdm_velocity_kick(
        projected, time_step_myr=0.31, random_seed=41,
        completed_steps=7, closure_by_source_sha256=closures,
    )
    assert accepted.status == "candidate_pending_physical_validation"
    assert accepted.maximum_coherence_time_myr == pytest.approx(0.03)
