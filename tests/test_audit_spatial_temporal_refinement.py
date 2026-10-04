import numpy as np
import pytest

from scripts.audit_spatial_temporal_refinement import (
    compare_matched_series, config_matches_except_save_count,
)


def test_matched_half_step_series_uses_same_physical_times():
    coarse = 0.572 - np.linspace(0, 80, 81) * 1e-6
    fine = 0.572 - np.linspace(0, 160, 161) * 0.5e-6
    fine[-1] += 2e-9
    result = compare_matched_series(coarse, fine)
    assert result["endpoint_pc"] == pytest.approx(2e-9, abs=1e-16)
    assert result["maximum_absolute_prefix_pc"] == pytest.approx(2e-9, abs=1e-16)
    with pytest.raises(ValueError, match="finite initial"):
        compare_matched_series(coarse, fine[:-1])
    changed = fine.copy()
    changed[0] += 1e-8
    with pytest.raises(ValueError, match="finite initial"):
        compare_matched_series(coarse, changed)


def test_config_only_doubles_save_count():
    old = {"Save Options": {"Number": 468000, "Flags": "Energy NBody"},
           "Matter Particles": {"Plummer Radius": 0.05}}
    half = {"Save Options": {"Number": 936000, "Flags": "Energy NBody"},
            "Matter Particles": {"Plummer Radius": 0.05}}
    assert config_matches_except_save_count(old, half)
    half["Matter Particles"]["Plummer Radius"] = 0.04
    assert not config_matches_except_save_count(old, half)
    half["Matter Particles"]["Plummer Radius"] = 0.05
    half["Save Options"]["Number"] = 936001
    assert not config_matches_except_save_count(old, half)
