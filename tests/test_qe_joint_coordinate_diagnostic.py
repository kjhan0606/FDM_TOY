from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.qe_box_control import (
    _joint_orbit_hull_diagnostic,
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


def test_joint_hull_reports_measured_2d_overlap_without_releasing_mapping() -> None:
    points = [(1.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.4, 0.2)]
    fine = _orbits(points + [(3.0, 0.8)])
    coarse = _orbits(points)
    doubled = _orbits(points)
    fine["mean_separation_pc"] = 1.0
    coarse["mean_separation_pc"] = 1.1
    doubled["mean_separation_pc"] = 0.9
    result = _joint_orbit_hull_diagnostic(
        {"fine": fine, "coarse": coarse, "doubled_box": doubled}
    )
    assert result["status"] == "joint_2d_hull_overlap_diagnostic_only"
    assert result["fine_orbits_tested"] == 5
    assert result["fine_orbits_inside_both_other_hulls"] == 4
    assert result["maximum_absolute_coarse_minus_fine_fraction"] == pytest.approx(0.1)
    assert result["maximum_absolute_doubled_box_minus_fine_fraction"] == pytest.approx(0.1)
    assert result["convex_hull_may_bridge_unsampled_holes"] is True
    assert result["joint_a_e_support_verified"] is False
    assert result["runtime_mapping_admitted"] is False


def test_joint_hull_censors_degenerate_or_disjoint_samples() -> None:
    diagonal = _orbits([(1.0, 0.2), (1.5, 0.2), (2.0, 0.2)])
    result = _joint_orbit_hull_diagnostic(
        {"fine": diagonal, "coarse": diagonal, "doubled_box": diagonal}
    )
    assert result["status"] == "degenerate_2d_coordinate_support_censored"
    fine = _orbits([(1.0, 0.0), (2.0, 0.0), (1.0, 0.2)])
    coarse = _orbits([(1.0, 0.3), (2.0, 0.3), (2.0, 0.5)])
    result = _joint_orbit_hull_diagnostic(
        {"fine": fine, "coarse": coarse, "doubled_box": fine}
    )
    assert result["status"] == "no_common_2d_hull_sample_censored"
    assert result["runtime_mapping_admitted"] is False


def test_joint_hull_rejects_non_single_valued_revisited_state() -> None:
    points = [(1.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.4, 0.2)]
    fine = _orbits(points + [(1.4, 0.2)])
    fine["mean_separation_pc"][-1] = 1.2
    control = _orbits(points)
    result = _joint_orbit_hull_diagnostic(
        {"fine": fine, "coarse": control, "doubled_box": control}
    )
    assert result["status"] == "non_single_valued_orbit_coordinates_censored"
    assert result["conflicting_role"] == "fine"
    assert result["conflicting_revisit_relative_spread"] == pytest.approx(0.2)
    assert result["runtime_mapping_admitted"] is False
