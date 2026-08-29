from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from fdm_smbh_delay.dual_soliton_relaxation import (
    assess_dual_soliton_relaxation,
)


def _identity(path: Path, *, status: str = "runtime_seed_identity_verified") -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "status": status, "seed_case_id": "case-a"}),
        encoding="utf-8",
    )


def _evidence(identity_path: Path) -> dict[str, object]:
    times = [0.0, 0.1, 0.2, 0.3]
    return {
        "schema_version": 1,
        "seed_case_id": "case-a",
        "runtime_identity_path": str(identity_path),
        "relaxation_window_start_code": 0.1,
        "sample_times_code": times,
        "component_core_mass_code": [
            [4.0, 3.0],
            [4.0, 3.0],
            [4.02, 3.01],
            [4.01, 2.99],
        ],
        "component_core_radius_box": [
            [0.1, 0.1],
            [0.1, 0.1],
            [0.1005, 0.1005],
            [0.1002, 0.1001],
        ],
        "component_centres_box": [
            [[0.35, 0.5, 0.5], [0.65, 0.5, 0.5]],
            [[0.35, 0.5, 0.5], [0.65, 0.5, 0.5]],
            [[0.351, 0.5, 0.5], [0.649, 0.5, 0.5]],
            [[0.352, 0.5, 0.5], [0.648, 0.5, 0.5]],
        ],
        "relative_wave_mass_error": [0.0, 5.0e-4, 6.0e-4, 7.0e-4],
        "relative_hamiltonian_error": [0.0, 4.0e-4, 7.0e-4, 8.0e-4],
        "relative_angular_momentum_error": [0.0, 3.0e-4, 4.0e-4, 5.0e-4],
        "thresholds": {
            "maximum_relative_component_mass_drift": 0.02,
            "maximum_relative_component_radius_drift": 0.02,
            "maximum_relative_wave_mass_error": 1.0e-3,
            "maximum_relative_hamiltonian_error": 1.0e-3,
            "maximum_relative_angular_momentum_error": 1.0e-3,
            "minimum_core_separation_to_radius_ratio": 2.0,
        },
    }


def test_relaxation_assessment_passes_only_a_verified_initial_window(tmp_path: Path) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_evidence(identity)), encoding="utf-8")
    assessment = assess_dual_soliton_relaxation(evidence)
    assert assessment.passed
    assert assessment.metrics["minimum_core_separation_to_radius_ratio"] > 2.0
    assert assessment.as_dict()["status"] == "relaxation_conservation_passed"


def test_relaxation_assessment_rejects_component_drift_or_core_overlap(tmp_path: Path) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    record = _evidence(identity)
    record["component_core_mass_code"][3][0] = 4.2
    record["component_centres_box"][3][1][0] = 0.5
    evidence = tmp_path / "failed-evidence.json"
    evidence.write_text(json.dumps(record), encoding="utf-8")
    assessment = assess_dual_soliton_relaxation(evidence)
    assert not assessment.passed
    assert any("mass_drift" in reason for reason in assessment.reasons)
    assert any("separation guard" in reason for reason in assessment.reasons)


def test_relaxation_assessment_requires_verified_runtime_identity(tmp_path: Path) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity, status="runtime_seed_identity_not_verified")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_evidence(identity)), encoding="utf-8")
    with pytest.raises(ValueError, match="verified runtime seed identity"):
        assess_dual_soliton_relaxation(evidence)


def test_relaxation_cli_writes_atomic_assessment(tmp_path: Path) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_evidence(identity)), encoding="utf-8")
    output = tmp_path / "assessment" / "result.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/assess_dual_soliton_relaxation.py",
            str(evidence),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "relaxation_conservation_passed"
