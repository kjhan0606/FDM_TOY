import numpy as np
import pytest

from scripts.audit_spatial_temporal_refinement import compare_matched_series


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
