from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from fdm_smbh_delay.dual_soliton_relaxation import (
    assess_dual_soliton_relaxation,
    materialize_dual_soliton_relaxation_diagnostic_provenance,
    materialize_dual_soliton_relaxation_sample_ledger,
    read_verified_dual_soliton_relaxation_sample_ledger,
)
from fdm_smbh_delay.dual_soliton_preflight import (
    validate_pure_fdm_dual_soliton_runtime_identity,
)
from fdm_smbh_delay.dual_soliton_seed import (
    materialize_pure_fdm_dual_soliton_seed,
    read_materialized_pure_fdm_dual_soliton_seed,
)
from fdm_smbh_delay.fdm_zoom_runtime_identity import (
    assess_fdm_declared_zoom_runtime_outputs,
)
from test_fdm_zoom_seed_binding import (
    _declared_run_binding,
    _write_fdm_runtime_output,
)


def _identity(path: Path, *, status: str = "runtime_seed_identity_verified") -> None:
    if status != "runtime_seed_identity_verified":
        path.write_text(
            json.dumps({"schema_version": 1, "status": status, "seed_case_id": "case-a"}),
            encoding="utf-8",
        )
        return
    declared, namelist, seed_manifest = _declared_run_binding(path.parent)
    outputs = [
        _write_fdm_runtime_output(
            path.parent,
            number=number,
            run_namelist=namelist,
            seed_manifest=seed_manifest,
        )
        for number in (1, 2, 3, 4)
    ]
    output_decision = assess_fdm_declared_zoom_runtime_outputs(declared, outputs)
    assert output_decision.verified
    output_identity = path.parent / "fdm-runtime-output-identity.json"
    output_identity.write_text(json.dumps(output_decision.as_dict()), encoding="utf-8")
    provenance = outputs[0] / "fdm_outer_wave_provenance_00001.txt"
    identity = validate_pure_fdm_dual_soliton_runtime_identity(
        seed_manifest_path=seed_manifest,
        provenance_path=provenance,
    )
    path.write_text(
        json.dumps(identity.as_dict()),
        encoding="utf-8",
    )


def _sample_ledger(identity_path: Path) -> Path:
    output_identity = identity_path.parent / "fdm-runtime-output-identity.json"
    output_record = json.loads(output_identity.read_text(encoding="utf-8"))
    raw_provenance_paths = [
        item["raw_fdm_provenance"]["path"] for item in output_record["complete_outputs"]
    ]
    manifest = identity_path.parent / "sample-manifest.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "raw_fdm_provenance_paths": raw_provenance_paths}
        ),
        encoding="utf-8",
    )
    ledger = identity_path.parent / "sample-ledger.json"
    materialize_dual_soliton_relaxation_sample_ledger(
        runtime_identity_path=identity_path,
        runtime_output_identity_path=output_identity,
        sample_manifest_path=manifest,
        output_path=ledger,
    )
    return ledger


def _diagnostic_provenance(
    sample_ledger_path: Path, *, failing: bool = False
) -> Path:
    diagnostics = {
        "sample_times_code": [0.1, 0.2, 0.3, 0.4],
        "component_core_mass_code": [
            [4.0, 3.0],
            [4.0, 3.0],
            [4.02, 3.01],
            [4.01 if not failing else 4.2, 2.99],
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
            [[0.352, 0.5, 0.5], [0.648 if not failing else 0.5, 0.5, 0.5]],
        ],
        "relative_wave_mass_error": [0.0, 5.0e-4, 6.0e-4, 7.0e-4],
        "relative_hamiltonian_error": [0.0, 4.0e-4, 7.0e-4, 8.0e-4],
        "relative_angular_momentum_error": [0.0, 3.0e-4, 4.0e-4, 5.0e-4],
    }
    extractor = sample_ledger_path.parent / "extract_relaxation.py"
    extractor.write_text("# declared diagnostic extractor\n", encoding="utf-8")
    manifest = sample_ledger_path.parent / "diagnostic-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extractor_path": str(extractor),
                "extractor_version": "test-v1",
                "diagnostics": diagnostics,
            }
        ),
        encoding="utf-8",
    )
    output = sample_ledger_path.parent / (
        "failed-diagnostic-provenance.json" if failing else "diagnostic-provenance.json"
    )
    materialize_dual_soliton_relaxation_diagnostic_provenance(
        sample_ledger_path=sample_ledger_path,
        diagnostic_manifest_path=manifest,
        output_path=output,
    )
    return output


def _evidence(
    identity_path: Path, sample_ledger_path: Path, diagnostic_provenance_path: Path
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "seed_case_id": "capture-seed",
        "runtime_identity_path": str(identity_path),
        "sample_ledger_path": str(sample_ledger_path),
        "diagnostic_provenance_path": str(diagnostic_provenance_path),
        "relaxation_window_start_code": 0.2,
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
    sample_ledger = _sample_ledger(identity)
    diagnostic_provenance = _diagnostic_provenance(sample_ledger)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(_evidence(identity, sample_ledger, diagnostic_provenance)), encoding="utf-8"
    )
    assessment = assess_dual_soliton_relaxation(evidence)
    assert assessment.conditionally_within_thresholds
    assert assessment.metrics["minimum_core_separation_to_radius_ratio"] > 2.0
    assert (
        assessment.as_dict()["status"]
        == "relaxation_conservation_declared_series_within_thresholds"
    )


def test_relaxation_assessment_rejects_component_drift_or_core_overlap(tmp_path: Path) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    sample_ledger = _sample_ledger(identity)
    diagnostic_provenance = _diagnostic_provenance(sample_ledger, failing=True)
    record = _evidence(identity, sample_ledger, diagnostic_provenance)
    evidence = tmp_path / "failed-evidence.json"
    evidence.write_text(json.dumps(record), encoding="utf-8")
    assessment = assess_dual_soliton_relaxation(evidence)
    assert not assessment.conditionally_within_thresholds
    assert any("mass_drift" in reason for reason in assessment.reasons)
    assert any("separation guard" in reason for reason in assessment.reasons)


def test_relaxation_assessment_requires_verified_runtime_identity(tmp_path: Path) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity, status="runtime_seed_identity_not_verified")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            _evidence(
                identity,
                tmp_path / "unused-ledger.json",
                tmp_path / "unused-diagnostics.json",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="verified runtime seed identity"):
        assess_dual_soliton_relaxation(evidence)


def test_relaxation_cli_writes_atomic_assessment(tmp_path: Path) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    sample_ledger = _sample_ledger(identity)
    diagnostic_provenance = _diagnostic_provenance(sample_ledger)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(_evidence(identity, sample_ledger, diagnostic_provenance)), encoding="utf-8"
    )
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
    assert (
        json.loads(output.read_text(encoding="utf-8"))["status"]
        == "relaxation_conservation_declared_series_within_thresholds"
    )


def test_relaxation_rejects_a_runtime_provenance_changed_after_sample_binding(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    sample_ledger = _sample_ledger(identity)
    diagnostic_provenance = _diagnostic_provenance(sample_ledger)
    provenance = tmp_path / "output_00001" / "fdm_outer_wave_provenance_00001.txt"
    provenance.write_text(
        provenance.read_text(encoding="utf-8") + "# altered after identity\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(_evidence(identity, sample_ledger, diagnostic_provenance)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="verified runtime seed identity"):
        assess_dual_soliton_relaxation(evidence)


def test_relaxation_rejects_a_changed_wave_snapshot_after_sample_binding(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    sample_ledger = _sample_ledger(identity)
    diagnostic_provenance = _diagnostic_provenance(sample_ledger)
    wave_snapshot = tmp_path / "output_00002" / "fdm_00002.out00001"
    wave_snapshot.write_text("changed wave sample\n", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(_evidence(identity, sample_ledger, diagnostic_provenance)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="verified sample ledger"):
        assess_dual_soliton_relaxation(evidence)


def test_relaxation_sample_ledger_requires_every_output_in_the_verified_set(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    sample_ledger = _sample_ledger(identity)
    record = json.loads(sample_ledger.read_text(encoding="utf-8"))
    record["samples"] = record["samples"][:-1]
    sample_ledger.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="raw provenance paths differ"):
        read_verified_dual_soliton_relaxation_sample_ledger(sample_ledger)


def test_relaxation_sample_ledger_rejects_a_different_seed_with_the_same_case_and_solitons(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    original_seed = read_materialized_pure_fdm_dual_soliton_seed(
        tmp_path / "seed" / "dual_soliton_seed_manifest.json"
    )
    changed_first_sink = replace(
        original_seed.sinks[0], angular_momentum_code=(0.0, 0.0, 9.0e-5)
    )
    changed_seed = replace(
        original_seed, sinks=(changed_first_sink, original_seed.sinks[1])
    )
    materialize_pure_fdm_dual_soliton_seed(changed_seed, tmp_path / "changed-seed")
    raw = tmp_path / "output_00001" / "fdm_outer_wave_provenance_00001.txt"
    changed_identity = validate_pure_fdm_dual_soliton_runtime_identity(
        seed_manifest_path=tmp_path / "changed-seed" / "dual_soliton_seed_manifest.json",
        provenance_path=raw,
    )
    assert changed_identity.verified
    changed_identity_path = tmp_path / "changed-runtime-identity.json"
    changed_identity_path.write_text(json.dumps(changed_identity.as_dict()), encoding="utf-8")
    output_identity = tmp_path / "fdm-runtime-output-identity.json"
    output_record = json.loads(output_identity.read_text(encoding="utf-8"))
    manifest = tmp_path / "changed-seed-sample-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "raw_fdm_provenance_paths": [
                    item["raw_fdm_provenance"]["path"]
                    for item in output_record["complete_outputs"]
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="seed manifest differs"):
        materialize_dual_soliton_relaxation_sample_ledger(
            runtime_identity_path=changed_identity_path,
            runtime_output_identity_path=output_identity,
            sample_manifest_path=manifest,
            output_path=tmp_path / "changed-seed-sample-ledger.json",
        )


def test_relaxation_rejects_a_tampered_nested_runtime_decision(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    sample_ledger = _sample_ledger(identity)
    diagnostic_provenance = _diagnostic_provenance(sample_ledger)
    identity_record = json.loads(identity.read_text(encoding="utf-8"))
    identity_record["raw_provenance_decision"]["status"] = "censored"
    identity.write_text(json.dumps(identity_record), encoding="utf-8")
    ledger_record = json.loads(sample_ledger.read_text(encoding="utf-8"))
    ledger_record["sources"]["runtime_identity"]["sha256"] = hashlib.sha256(
        identity.read_bytes()
    ).hexdigest()
    sample_ledger.write_text(json.dumps(ledger_record), encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(_evidence(identity, sample_ledger, diagnostic_provenance)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="verified runtime seed identity"):
        assess_dual_soliton_relaxation(evidence)
