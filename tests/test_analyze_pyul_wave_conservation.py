"""Focused transfer-normalized conservation regressions."""

import numpy as np
import pytest

from scripts.analyze_pyul_wave_run import (
    _energy_error_over_transfer,
    _energy_error_timeseries_over_transfer,
)


def _transient_history_inputs() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    # Prefix Hamiltonian error is 0.012 after the first step. Increasing
    # physical transfer makes later ratios 0.004 and 0.0003.
    combined = np.array([0.0, 0.012, 0.012, 0.012])
    transfer = np.array([0.0, 1.0, 3.0, 40.0])
    return combined, (transfer,)


def test_transient_peak_cannot_be_hidden_by_passing_final_ratio() -> None:
    combined, components = _transient_history_inputs()
    history = _energy_error_timeseries_over_transfer(combined, components)
    assert history == pytest.approx([0.0, 0.012, 0.004, 0.0003])
    assert _energy_error_over_transfer(combined, components) == pytest.approx(0.012)
    assert history[-1] == pytest.approx(0.0003)


def test_resolved_truncation_excludes_only_samples_after_cutoff() -> None:
    combined, components = _transient_history_inputs()
    assert _energy_error_over_transfer(
        combined[:2], tuple(component[:2] for component in components)
    ) == pytest.approx(0.012)
    assert _energy_error_over_transfer(
        combined[:3], tuple(component[:3] for component in components)
    ) == pytest.approx(0.012)
