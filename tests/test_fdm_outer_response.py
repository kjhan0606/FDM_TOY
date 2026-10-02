from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.fdm_outer_response import FDMOuterResponseTable


def _table(*, status: str = "calibrated") -> FDMOuterResponseTable:
    radii = np.array([10.0, 20.0, 40.0, 80.0])
    diffusion = np.array(
        [np.diag([0.01 * radius, 0.02 * radius, 0.03 * radius]) for radius in radii]
    )
    return FDMOuterResponseTable(
        radii_pc=radii,
        drift_acceleration_pc_myr2=np.column_stack(
            (-0.1 * radii, 0.02 * radii, np.zeros(4))
        ),
        diffusion_tensor_pc2_myr3=diffusion,
        response_status=status,
    )


def test_calibrated_response_evaluates_without_extrapolation() -> None:
    response = _table()
    values = response.evaluate(30.0)
    assert values["drift_acceleration_pc_myr2"].shape == (3,)
    assert values["diffusion_tensor_pc2_myr3"].shape == (3, 3)
    assert response.decision(30.0)["status"] == "available"


def test_uncalibrated_response_is_censored_not_zero_drag() -> None:
    response = _table(status="uncalibrated")
    with pytest.raises(ValueError, match="not calibrated"):
        response.evaluate(30.0)
    decision = response.decision(30.0)
    assert decision["status"] == "censored"
    assert "uncalibrated" in decision["reason"]


def test_response_support_outside_range_is_censored() -> None:
    decision = _table().decision(5.0)
    assert decision["status"] == "censored"
    assert "outside tabulated support" in decision["reason"]


def test_response_rejects_non_positive_semidefinite_diffusion() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        FDMOuterResponseTable(
            radii_pc=np.array([10.0, 20.0]),
            drift_acceleration_pc_myr2=np.zeros((2, 3)),
            diffusion_tensor_pc2_myr3=np.array(
                [np.diag([1.0, -1.0, 1.0]), np.eye(3)]
            ),
        )


def test_diffusion_interpolation_stays_inside_psd_cone() -> None:
    vectors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]]
    )
    tensors = np.einsum("ni,nj->nij", vectors, vectors)
    response = FDMOuterResponseTable(
        radii_pc=np.array([10.0, 20.0, 40.0]),
        drift_acceleration_pc_myr2=np.zeros((3, 3)),
        diffusion_tensor_pc2_myr3=tensors,
        response_status="calibrated",
    )
    for radius, expected in (
        (10.0, tensors[0]),
        (15.0, 0.5 * (tensors[0] + tensors[1])),
        (20.0, tensors[1]),
        (30.0, 0.5 * (tensors[1] + tensors[2])),
        (40.0, tensors[2]),
    ):
        measured = response.evaluate(radius)["diffusion_tensor_pc2_myr3"]
        assert measured == pytest.approx(expected)
        assert np.min(np.linalg.eigvalsh(measured)) >= -1.0e-12
    assert response.decision(41.0)["status"] == "censored"


def test_response_snapshots_inputs_and_cannot_be_mutated_after_construction() -> None:
    radii = np.array([10.0, 20.0])
    drift = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    diffusion = np.array([np.eye(3), 2.0 * np.eye(3)])
    response = FDMOuterResponseTable(
        radii_pc=radii,
        drift_acceleration_pc_myr2=drift,
        diffusion_tensor_pc2_myr3=diffusion,
        response_status="calibrated",
    )
    original = response.evaluate(15.0)
    record = response.as_dict()
    radii[:] = [1.0, 2.0]
    drift[:] = 999.0
    diffusion[:] = -np.eye(3)
    assert response.evaluate(15.0)["drift_acceleration_pc_myr2"] == pytest.approx(
        original["drift_acceleration_pc_myr2"]
    )
    assert response.evaluate(15.0)["diffusion_tensor_pc2_myr3"] == pytest.approx(
        original["diffusion_tensor_pc2_myr3"]
    )
    assert response.as_dict() == record
    for array in (
        response.radii_pc,
        response.drift_acceleration_pc_myr2,
        response.diffusion_tensor_pc2_myr3,
    ):
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
