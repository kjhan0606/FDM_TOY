import numpy as np
import pytest

from scripts.audit_one_orbit_prefix import orbital_coverage


def test_coverage_measures_signed_unwrapped_turns_and_radial_extrema():
    angle = np.linspace(0.0, -2.4 * np.pi, 19)
    radius = 1.0 + 0.2 * np.cos(angle)
    states = np.zeros((19, 2, 6))
    states[:, 1, 0] = radius * np.cos(angle)
    states[:, 1, 1] = radius * np.sin(angle)
    result = orbital_coverage(states, 0.5)
    assert result["one_projected_turn_reached"] is True
    assert result["projected_azimuthal_turns"] == pytest.approx(1.2)
    assert result["sampled_radial_turning_points"] >= 2
    assert result["minimum_saved_separation_pc"] < result["initial_separation_pc"]


def test_coverage_rejects_undersampled_or_invalid_phase():
    states = np.zeros((4, 2, 6))
    angle = np.array([0, 0.2, 2.1, 2.4])
    states[:, 1, 0] = np.cos(angle)
    states[:, 1, 1] = np.sin(angle)
    with pytest.raises(ValueError, match="undersampled"):
        orbital_coverage(states, 1.0)
    states[:, 1, :2] = 0.0
    with pytest.raises(ValueError, match="zero"):
        orbital_coverage(states, 1.0)
