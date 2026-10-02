from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.fdm_orbital_response import (
    FDMOrbitalResponseSupport,
    project_fdm_outer_response_to_orbit,
)
from fdm_smbh_delay.fdm_outer_response import FDMOuterResponseTable


def _table(*, frame: str = "orbital_rtn", status: str = "calibrated") -> FDMOuterResponseTable:
    return FDMOuterResponseTable(
        radii_pc=np.array([10.0, 20.0]),
        drift_acceleration_pc_myr2=np.array([[1.0, -2.0, 3.0]] * 2),
        diffusion_tensor_pc2_myr3=np.array([np.diag([1.0, 2.0, 3.0])] * 2),
        response_status=status,
        component_frame=frame,
    )


def _support() -> FDMOrbitalResponseSupport:
    return FDMOrbitalResponseSupport(0.1, 0.5, 0.0, 0.8)


def test_orbital_response_projection_rotates_covariantly_and_keeps_psd() -> None:
    response = _table()
    first = project_fdm_outer_response_to_orbit(
        response, _support(), position_pc=np.array([15.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 5.0, 0.0]),
        mass_ratio_q=0.2, eccentricity=0.3,
    )
    assert first.status == "projected_candidate_pending_source_calibration"
    assert first.drift_acceleration_pc_myr2 == pytest.approx([1.0, -2.0, 3.0])
    assert first.diffusion_tensor_pc2_myr3 == pytest.approx(np.diag([1.0, 2.0, 3.0]))
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turned = project_fdm_outer_response_to_orbit(
        response, _support(), position_pc=rotation @ np.array([15.0, 0.0, 0.0]),
        velocity_pc_myr=rotation @ np.array([0.0, 5.0, 0.0]),
        mass_ratio_q=0.2, eccentricity=0.3,
    )
    assert turned.drift_acceleration_pc_myr2 == pytest.approx(
        rotation @ first.drift_acceleration_pc_myr2
    )
    assert turned.diffusion_tensor_pc2_myr3 == pytest.approx(
        rotation @ first.diffusion_tensor_pc2_myr3 @ rotation.T
    )
    assert np.min(np.linalg.eigvalsh(turned.diffusion_tensor_pc2_myr3)) >= 0.0
    assert FDMOuterResponseTable.from_dict(response.as_dict()).component_frame == "orbital_rtn"


@pytest.mark.parametrize(
    ("q", "eccentricity", "radius", "frame", "status", "velocity"),
    [
        (0.6, 0.3, 15.0, "orbital_rtn", "calibrated", (0.0, 5.0, 0.0)),
        (0.2, 0.9, 15.0, "orbital_rtn", "calibrated", (0.0, 5.0, 0.0)),
        (0.2, 0.3, 25.0, "orbital_rtn", "calibrated", (0.0, 5.0, 0.0)),
        (0.2, 0.3, 15.0, "unspecified", "calibrated", (0.0, 5.0, 0.0)),
        (0.2, 0.3, 15.0, "orbital_rtn", "uncalibrated", (0.0, 5.0, 0.0)),
        (0.2, 0.3, 15.0, "orbital_rtn", "calibrated", (5.0, 0.0, 0.0)),
    ],
)
def test_orbital_response_censors_unsupported_state_without_zero_force(
    q: float, eccentricity: float, radius: float, frame: str,
    status: str, velocity: tuple[float, float, float],
) -> None:
    response = project_fdm_outer_response_to_orbit(
        _table(frame=frame, status=status), _support(),
        position_pc=np.array([radius, 0.0, 0.0]),
        velocity_pc_myr=np.array(velocity),
        mass_ratio_q=q, eccentricity=eccentricity,
    )
    assert response.status == "censored"
    assert response.drift_acceleration_pc_myr2 is None
    assert response.diffusion_tensor_pc2_myr3 is None


def test_orbital_response_support_validates_physical_bounds() -> None:
    with pytest.raises(ValueError, match="q/e support"):
        FDMOrbitalResponseSupport(0.6, 0.5, 0.0, 0.8)
    with pytest.raises(ValueError, match="component frame"):
        _table(frame="unknown")
