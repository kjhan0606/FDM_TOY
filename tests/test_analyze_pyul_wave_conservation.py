"""Focused transfer-normalized conservation regressions."""

import json

import numpy as np
import pytest

from scripts.analyze_pyul_wave_run import (
    _diagnostic_status,
    _energy_error_over_transfer,
    _energy_error_timeseries_over_transfer,
)


def test_partial_torch_analysis_requires_explicit_diagnostic_mode(tmp_path) -> None:
    (tmp_path / "fdm_adapter_metadata.json").write_text(json.dumps({
        "backend": "pytorch_cuda", "diagnostic_stop_after_save": 2,
    }))
    summary = tmp_path / "torch_run_summary.json"
    summary.write_text(json.dumps({"status": "diagnostic_partial"}))
    with pytest.raises(ValueError, match="complete Torch evolution"):
        _diagnostic_status(tmp_path, allow_partial=False)
    assert _diagnostic_status(tmp_path, allow_partial=True) == "diagnostic_partial"
    summary.write_text(json.dumps({"status": "diagnostic_complete"}))
    with pytest.raises(ValueError, match="requires diagnostic_partial"):
        _diagnostic_status(tmp_path, allow_partial=True)
    summary.write_text(json.dumps({"status": "complete"}))
    with pytest.raises(ValueError, match="complete Torch evolution"):
        _diagnostic_status(tmp_path, allow_partial=False)
    with pytest.raises(ValueError, match="requires diagnostic_partial"):
        _diagnostic_status(tmp_path, allow_partial=True)


def test_interrupted_torch_diagnostic_without_summary_cannot_be_diagnosed(
    tmp_path,
) -> None:
    metadata = tmp_path / "fdm_adapter_metadata.json"
    metadata.write_text(json.dumps({
        "backend": "pytorch_cuda", "diagnostic_stop_after_save": 2,
    }))
    for allow_partial in (False, True):
        with pytest.raises(ValueError, match="requires a run summary"):
            _diagnostic_status(tmp_path, allow_partial=allow_partial)
    metadata.write_text(json.dumps({"backend": "pytorch_cuda"}))
    with pytest.raises(ValueError, match="requires a run summary"):
        _diagnostic_status(tmp_path, allow_partial=False)
    metadata.write_text(json.dumps({"backend": "pyul_nbody"}))
    assert _diagnostic_status(tmp_path, allow_partial=False) == "diagnosed"


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
