from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.lagramses import (
    NumericalMergeScale,
    legacy_source_binding_proxy,
    minimum_image,
    numerical_merge_radius_pc,
    physically_bound_pair,
)


def test_documented_numerical_merge_radii() -> None:
    matched = numerical_merge_radius_pc(
        box_size_mpc_h=1.2, levelmax=12, hubble_h=0.6766, rmerge=4.0
    )
    production = NumericalMergeScale(128.0, 15, 0.6766, 4.0)
    assert matched == pytest.approx(1732.0056163)
    assert production.cell_size_pc == pytest.approx(5773.3520544)
    assert production.merge_radius_pc == pytest.approx(23093.4082176)


def test_minimum_image() -> None:
    displacement = np.array([9.0, -9.0, 1.0])
    assert np.allclose(minimum_image(displacement, 10.0), [-1.0, 1.0, 1.0])


def test_legacy_proxy_reproduces_inverse_squared_source_expression() -> None:
    kinetic, proxy, accepted = legacy_source_binding_proxy(
        2.0,
        3.0,
        0.25,
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        0.5,
    )
    assert kinetic == pytest.approx(2.4)
    assert proxy == pytest.approx(12.0)
    assert accepted


def test_standard_physical_binding_check() -> None:
    assert physically_bound_pair(
        1.0e8,
        1.0e8,
        1.0,
        np.array([0.0, 100.0, 0.0]),
        np.array([0.0, -100.0, 0.0]),
    )
    assert not physically_bound_pair(
        1.0e8,
        1.0e8,
        1.0,
        np.array([0.0, 1000.0, 0.0]),
        np.array([0.0, -1000.0, 0.0]),
    )
