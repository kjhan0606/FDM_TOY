"""Checks for interaction-only versus joint half-kick accounting."""

from copy import deepcopy

import pytest

from scripts.summarize_coupled_kick_attribution import assess_level


def _row() -> dict:
    return {
        "step": 1,
        "wave_compact_first": [0.6, 0.0, 0.0],
        "wave_compact_first_alternate": [0.5995, 0.0, 0.0],
        "body_first": [0.4, 0.0, 0.0],
        "wave_self_first": [0.001, 0.0, 0.0],
        "wave_self_first_alternate": [0.0015, 0.0, 0.0],
        "first_half_kick_defect": [1.001, 0.0, 0.0],
        "wave_compact_second": [1.5, 0.0, 0.0],
        "wave_compact_second_alternate": [1.4995, 0.0, 0.0],
        "body_second": [0.5, 0.0, 0.0],
        "wave_self_second": [0.002, 0.0, 0.0],
        "wave_self_second_alternate": [0.0025, 0.0, 0.0],
        "second_half_kick_defect": [2.002, 0.0, 0.0],
        "wave_drift": [0.003, 0.0, 0.0],
    }


def _payload() -> dict:
    return {
        "status": "coupled_factorized_kick_diagnostic_not_a_calibration_release",
        "calibration_eligible": False,
        "time_step_factor": 1.0,
        "wave_steps": 1,
        "run": "/scratch/candidate/f100",
        "per_step": [_row()],
        "final_total_momentum_residual_code": [3.006, 0.0, 0.0],
        "wave_compact_plus_body_attribution_code": [3.0, 0.0, 0.0],
        "wave_compact_plus_body_alternate_order_attribution_code": [
            2.999, 0.0, 0.0,
        ],
        "cpu_gpu_endpoint_momentum_gap_code": 1e-9,
    }


def test_interaction_only_half_kick_excludes_self() -> None:
    result = assess_level(_payload(), label="f100", factor=1.0, steps=1)
    assert result["maximum_interaction_only_half_kick_defect_code"] == 2.0
    assert result["maximum_joint_half_kick_change_code"] == 2.002
    assert result["interaction_pair_vector_gap_over_residual"] < 0.01
    assert result["alternate_interaction_pair_vector_gap_over_residual"] < 0.01


def test_joint_half_kick_closure_fails_closed() -> None:
    payload = deepcopy(_payload())
    payload["per_step"][0]["first_half_kick_defect"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="joint half-kick closure"):
        assess_level(payload, label="f100", factor=1.0, steps=1)


def test_interaction_dominance_fails_closed() -> None:
    payload = deepcopy(_payload())
    payload["per_step"][0]["wave_self_second"] = [0.2, 0.0, 0.0]
    payload["per_step"][0]["wave_self_second_alternate"] = [
        0.2005, 0.0, 0.0,
    ]
    payload["per_step"][0]["second_half_kick_defect"] = [2.2, 0.0, 0.0]
    payload["final_total_momentum_residual_code"] = [3.204, 0.0, 0.0]
    with pytest.raises(ValueError, match="interaction attribution does not dominate"):
        assess_level(payload, label="f100", factor=1.0, steps=1)
