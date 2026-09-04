from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.fdm_outer_halo import FDMOuterHaloClosure


def _closure(*, status: str = "uncalibrated") -> FDMOuterHaloClosure:
    radii = np.array([10.0, 30.0, 100.0, 300.0])
    return FDMOuterHaloClosure(
        radii_pc=radii,
        mass_current_msun_pc2_myr=np.column_stack(
            (0.01 * radii, -0.02 * radii, 0.03 * radii)
        ),
        coherence_time_myr=np.array([0.4, 0.6, 1.0, 1.8]),
        de_broglie_wavelength_pc=np.array([30.0, 35.0, 50.0, 75.0]),
        velocity_diffusion_pc2_myr3=np.array([0.0, 0.01, 0.02, 0.03]),
        density_gradient_scale_pc=np.array([15.0, 25.0, 70.0, 180.0]),
        closure_status=status,
    )


def test_outer_closure_interpolates_supported_wave_diagnostics() -> None:
    closure = _closure()
    values = closure.evaluate(50.0)
    assert values["mass_current_msun_pc2_myr"].shape == (3,)
    assert values["coherence_time_myr"] > 0.0
    assert values["de_broglie_wavelength_pc"] > 0.0
    assert values["velocity_diffusion_pc2_myr3"] >= 0.0


def test_outer_closure_rejects_extrapolation() -> None:
    closure = _closure()
    with pytest.raises(ValueError, match="outside tabulated support"):
        closure.evaluate(5.0)
    with pytest.raises(ValueError, match="outside tabulated support"):
        closure.evaluate(301.0)


def test_outer_closure_round_trips_and_requires_valid_status() -> None:
    closure = _closure(status="calibrated")
    decoded = FDMOuterHaloClosure.from_dict(closure.as_dict())
    assert decoded.closure_status == "calibrated"
    assert decoded.evaluate(100.0)["coherence_time_myr"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unsupported outer-halo closure status"):
        _closure(status="analytic")


def test_outer_closure_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        FDMOuterHaloClosure(
            radii_pc=np.array([10.0, 20.0]),
            mass_current_msun_pc2_myr=np.ones((2, 2)),
            coherence_time_myr=np.ones(2),
            de_broglie_wavelength_pc=np.ones(2),
            velocity_diffusion_pc2_myr3=np.ones(2),
            density_gradient_scale_pc=np.ones(2),
        )
