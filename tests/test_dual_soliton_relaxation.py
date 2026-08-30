from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from fdm_smbh_delay.dual_soliton_relaxation import (
    assess_dual_soliton_relaxation,
    materialize_dual_soliton_relaxation_diagnostic_provenance,
    materialize_dual_soliton_relaxation_sample_ledger,
)
from fdm_smbh_delay.dual_soliton_preflight import (
    validate_pure_fdm_dual_soliton_runtime_identity,
)
from fdm_smbh_delay.dual_soliton_seed import (
    DualSMBHSinkSeed,
    DualSolitonComponent,
    PureFDMDualSolitonSeed,
    materialize_pure_fdm_dual_soliton_seed,
)


def _identity(path: Path, *, status: str = "runtime_seed_identity_verified") -> None:
    if status != "runtime_seed_identity_verified":
        path.write_text(
            json.dumps({"schema_version": 1, "status": status, "seed_case_id": "case-a"}),
            encoding="utf-8",
        )
        return
    seed = PureFDMDualSolitonSeed(
        case_id="case-a",
        dark_matter_model="fdm",
        stellar_status="absent",
        gas_status="absent",
        box_length_code=1.0,
        m_axion_ev=1.0e-21,
        profile_c=0.091,
        solitons=(
            DualSolitonComponent(4.0, 0.1, (0.35, 0.5, 0.5), (0.0, 0.02, 0.0), 0.0),
            DualSolitonComponent(3.0, 0.1, (0.65, 0.5, 0.5), (0.0, -0.02, 0.0), 1.0),
        ),
        sinks=(
            DualSMBHSinkSeed(
                1.0e-4,
                5.0e-5,
                (0.35, 0.5, 0.5),
                (0.0, 0.02, 0.0),
                (0.0, 0.0, 1.0e-5),
            ),
            DualSMBHSinkSeed(
                8.0e-5,
                2.5e-5,
                (0.65, 0.5, 0.5),
                (0.0, -0.02, 0.0),
                (0.0, 0.0, -1.0e-5),
            ),
        ),
    )
    materialized = path.parent / "seed"
    materialize_pure_fdm_dual_soliton_seed(seed, materialized)
    provenance = path.parent / "fdm_outer_wave_provenance.txt"
    provenance.write_text(
        "# fdm_outer_wave_provenance_v2\n"
        "time_code = 1.0d0\n"
        "aexp = 5.0d-1\n"
        "nstep_coarse = 42\n"
        "m_axion_ev = 1.0d-21\n"
        "hbar_code = 2.0d-3\n"
        "fdm_use_hjm = F\n"
        "fdm_first_wave_level = 0\n"
        "analytic_fdm_drag_enabled = F\n"
        "force_accounting = resolved_wave_only\n"
        "leaf_mass_code = 3.0d0\n"
        "integrated_current_code = 1.0d-2 -2.0d-2 3.0d-2\n"
        "leaf_cell_count = 100.0d0\n"
        "complete_current_stencil_cell_count = 98.0d0\n"
        "complete_current_stencil_fraction = 9.8d-1\n"
        "psi_snapshot_prefix = fdm_00042.out\n"
        "fdm_dual_soliton_ic = T\n"
        "fdm_dual_soliton_profile_c = 9.1d-2\n"
        "fdm_dual_soliton_rho0 = 4.0d0 3.0d0\n"
        "fdm_dual_soliton_rc_box = 1.0d-1 1.0d-1\n"
        "fdm_dual_soliton_center_box_1 = 3.5d-1 5.0d-1 5.0d-1\n"
        "fdm_dual_soliton_center_box_2 = 6.5d-1 5.0d-1 5.0d-1\n"
        "fdm_dual_soliton_velocity_1 = 0.0d0 2.0d-2 0.0d0\n"
        "fdm_dual_soliton_velocity_2 = 0.0d0 -2.0d-2 0.0d0\n"
        "fdm_dual_soliton_phase = 0.0d0 1.0d0\n",
        encoding="utf-8",
    )
    identity = validate_pure_fdm_dual_soliton_runtime_identity(
        seed_manifest_path=materialized / "dual_soliton_seed_manifest.json",
        provenance_path=provenance,
    )
    path.write_text(
        json.dumps(identity.as_dict()),
        encoding="utf-8",
    )


def _sample_ledger(identity_path: Path) -> Path:
    runtime_provenance = identity_path.parent / "fdm_outer_wave_provenance.txt"
    runtime_text = runtime_provenance.read_text(encoding="utf-8")
    raw_provenance_paths = []
    for index, (time_code, aexp, nstep_coarse) in enumerate(
        ((0.0, 0.45, 1), (0.1, 0.46, 5), (0.2, 0.47, 9))
    ):
        output = identity_path.parent / f"output_{index:05d}"
        output.mkdir()
        prefix = f"fdm_{index:05d}.out"
        provenance = output / "fdm_outer_wave_provenance.txt"
        provenance.write_text(
            runtime_text.replace("time_code = 1.0d0", f"time_code = {time_code:.1f}d0")
            .replace("aexp = 5.0d-1", f"aexp = {aexp:.2f}d0")
            .replace("nstep_coarse = 42", f"nstep_coarse = {nstep_coarse}")
            .replace("psi_snapshot_prefix = fdm_00042.out", f"psi_snapshot_prefix = {prefix}"),
            encoding="utf-8",
        )
        for shard in ("00001", "00002"):
            (output / f"{prefix}{shard}").write_text(
                f"wave {index} shard {shard}\n", encoding="utf-8"
            )
            (output / f"amr_{index:05d}.out{shard}").write_text(
                f"AMR {index} shard {shard}\n", encoding="utf-8"
            )
        raw_provenance_paths.append(str(provenance))
    for shard in ("00001", "00002"):
        (identity_path.parent / f"fdm_00042.out{shard}").write_text(
            f"wave runtime shard {shard}\n", encoding="utf-8"
        )
        (identity_path.parent / f"amr_00042.out{shard}").write_text(
            f"AMR runtime shard {shard}\n", encoding="utf-8"
        )
    raw_provenance_paths.append(str(runtime_provenance))
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
        sample_manifest_path=manifest,
        output_path=ledger,
    )
    return ledger


def _diagnostic_provenance(
    sample_ledger_path: Path, *, failing: bool = False
) -> Path:
    diagnostics = {
        "sample_times_code": [0.0, 0.1, 0.2, 1.0],
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
        "seed_case_id": "case-a",
        "runtime_identity_path": str(identity_path),
        "sample_ledger_path": str(sample_ledger_path),
        "diagnostic_provenance_path": str(diagnostic_provenance_path),
        "relaxation_window_start_code": 0.1,
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
    provenance = tmp_path / "fdm_outer_wave_provenance.txt"
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
