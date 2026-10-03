from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.qe_box_control import (
    _joint_orbit_coordinate_proximity,
    _resolved_orbits_in_fixed_bin,
)


_DTYPE = [
    ("mean_separation_pc", "f8"),
    ("mean_semimajor_axis_osculating_pc", "f8"),
    ("mean_eccentricity_osculating", "f8"),
    ("mean_separation_over_cell_size", "f8"),
    ("end_time_myr", "f8"),
]


def _orbits(points: list[tuple[float, float]]) -> np.ndarray:
    return np.array(
        [
            (1.0, axis, eccentricity, 4.0, float(index + 1))
            for index, (axis, eccentricity) in enumerate(points)
        ],
        dtype=_DTYPE,
    )


def test_separate_axis_and_e_ranges_do_not_establish_joint_orbit_match() -> None:
    fine = _orbits([(1.0, 0.0), (2.0, 0.5)])
    coarse = _orbits([(1.0, 0.5), (2.0, 0.0)])
    result = _joint_orbit_coordinate_proximity(
        {"fine": fine, "coarse": coarse, "doubled_box": fine}
    )
    assert result["status"] == "closest_joint_orbit_triplet_diagnostic_only"
    assert result["minimum_max_log_axis_or_e_mismatch"] == pytest.approx(0.5)
    assert result["joint_a_e_support_verified"] is False
    assert result["runtime_mapping_admitted"] is False


def test_close_triplet_remains_a_non_released_diagnostic() -> None:
    result = _joint_orbit_coordinate_proximity(
        {
            "fine": _orbits([(1.0, 0.2)]),
            "coarse": _orbits([(1.005, 0.21)]),
            "doubled_box": _orbits([(0.998, 0.195)]),
        }
    )
    assert result["minimum_max_log_axis_or_e_mismatch"] == pytest.approx(0.01)
    assert result["closest_triplet"]["fine"]["measured_over_kepler_mean_ratio"] == pytest.approx(
        1.0 / 1.02
    )
    assert result["runtime_mapping_admitted"] is False


def test_fixed_bin_uses_only_initially_resolved_complete_orbits() -> None:
    orbit = _orbits([(1.0, 0.2), (1.0, 0.2), (1.0, 0.2)])
    orbit["mean_separation_pc"] = (0.43, 0.435, 0.44)
    run = {
        "orbit_series": orbit,
        "conservation": {"initial_spatially_resolved_duration_myr": 2.0},
    }
    selected = _resolved_orbits_in_fixed_bin(
        run, 0.43, 0.44, include_upper=True
    )
    assert selected.size == 2
    assert selected["mean_separation_pc"].tolist() == pytest.approx([0.43, 0.435])


def test_empty_or_invalid_joint_orbits_do_not_create_mapping() -> None:
    empty = _orbits([])
    result = _joint_orbit_coordinate_proximity(
        {"fine": empty, "coarse": _orbits([(1.0, 0.2)]),
         "doubled_box": _orbits([(1.0, 0.2)])}
    )
    assert result["status"] == "no_joint_orbit_sample_censored"
    invalid = _orbits([(1.0, 0.2)])
    invalid["mean_eccentricity_osculating"] = 1.0
    with pytest.raises(ValueError, match="invalid orbit coordinates"):
        _joint_orbit_coordinate_proximity(
            {"fine": invalid, "coarse": _orbits([(1.0, 0.2)]),
             "doubled_box": _orbits([(1.0, 0.2)])}
        )
