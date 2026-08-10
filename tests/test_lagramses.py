from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.lagramses import (
    NumericalMergeScale,
    capture_boundary_from_files,
    legacy_source_binding_proxy,
    minimum_image,
    numerical_merge_radius_pc,
    pair_orbital_state,
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


def test_capture_boundary_from_archived_files(tmp_path) -> None:
    info = tmp_path / "info_00001.txt"
    namelist = tmp_path / "namelist.txt"
    info.write_text("levelmax = 15\nH0 = 67.66\naexp = 0.5\n", encoding="utf-8")
    namelist.write_text("&PHYSICS_PARAMS\nrmerge = 4.0d0\n/\n", encoding="utf-8")
    boundary = capture_boundary_from_files(
        info_path=info,
        namelist_path=namelist,
        box_size_mpc_h=128.0,
    )
    assert boundary.rmerge_origin == "namelist"
    assert boundary.scale.merge_radius_pc == pytest.approx(23093.4082176)


def test_pair_orbital_state_recovers_kepler_elements() -> None:
    mass1 = 1.0e8
    mass2 = 5.0e7
    separation = 2.0
    total_mass = mass1 + mass2
    relative_speed = np.sqrt(G_INTERNAL * total_mass / separation)
    state = pair_orbital_state(
        member_ids=(11, 22),
        masses_msun=(mass1, mass2),
        positions_pc=np.array([[2.0 / 3.0, 0.0, 0.0], [-4.0 / 3.0, 0.0, 0.0]]),
        velocities_pc_myr=np.array(
            [[0.0, relative_speed / 3.0, 0.0], [0.0, -2.0 * relative_speed / 3.0, 0.0]]
        ),
    )
    assert state.bound
    assert state.mass_ratio == pytest.approx(0.5)
    assert state.semi_major_axis_pc == pytest.approx(separation, rel=2.0e-5)
    assert state.eccentricity == pytest.approx(0.0, abs=2.0e-5)


def test_pair_orbital_state_uses_minimum_image() -> None:
    state = pair_orbital_state(
        member_ids=(1, 2),
        masses_msun=(1.0e8, 1.0e8),
        positions_pc=np.array([[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        velocities_pc_myr=np.zeros((2, 3)),
        periodic_box_pc=10.0,
    )
    assert state.separation_pc == pytest.approx(0.4)
