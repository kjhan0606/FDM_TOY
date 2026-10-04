"""Regression checks for reduction-order attribution comparison."""

from copy import deepcopy

import pytest

from scripts.compare_coupled_kick_precision import compare_level


def _record(method: str) -> dict:
    return {
        "status": "coupled_factorized_kick_diagnostic_not_a_calibration_release",
        "calibration_eligible": False,
        "momentum_estimator": method,
        "run": "/scratch/candidate/f100",
        "wave_steps": 10,
        "time_step_factor": 1.0,
        "initial_wave_sha256": "initial-wave",
        "initial_body_sha256": "initial-body",
        "final_wave_sha256": "final-wave",
        "final_body_sha256": "final-body",
        "source_sha256": {
            "scripts/probe_coupled_kick_attribution.py": method,
            "src/fdm_smbh_delay/torch_wave.py": "same-operator",
        },
        "final_total_momentum_residual_code": [1.0, 0.0, 0.0],
        "wave_compact_plus_body_attribution_code": [1.0, 0.0, 0.0],
        "wave_compact_plus_body_alternate_order_attribution_code": [
            1.0, 0.0, 0.0,
        ],
        "wave_self_attribution_code": [0.0, 0.0, 0.0],
        "wave_self_alternate_order_attribution_code": [0.0, 0.0, 0.0],
        "wave_drift_attribution_code": [0.0, 0.0, 0.0],
        "source_commit": method,
        "wave_relative_replay_difference": 1e-15,
    }


def test_interaction_pair_survives_changed_reduction() -> None:
    old = _record("marginal")
    new = _record("reordered")
    new["wave_compact_plus_body_attribution_code"] = [1.0001, 0.0, 0.0]
    result = compare_level(old, new, "f100")
    assert result["components"]["wave_compact_plus_body_attribution_code"][
        "difference_over_endpoint_residual"
    ] == pytest.approx(0.0001)


def test_changed_operator_source_fails_closed() -> None:
    old = _record("marginal")
    new = _record("reordered")
    new["source_sha256"]["src/fdm_smbh_delay/torch_wave.py"] = "changed"
    with pytest.raises(ValueError, match="numerical operator sources differ"):
        compare_level(old, new, "f100")


def test_large_interaction_change_fails_closed() -> None:
    old = _record("marginal")
    new = deepcopy(_record("reordered"))
    new["wave_compact_plus_body_attribution_code"] = [1.02, 0.0, 0.0]
    with pytest.raises(ValueError, match="reduction-order sensitive"):
        compare_level(old, new, "f100")
