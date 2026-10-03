from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from astropy import units as u
import numpy as np

from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.dm_comparison import (
    assess_dm_comparison_physics_inputs,
    assess_dm_comparison_smoke_outputs,
    preflight_dm_comparison_family,
    read_dm_comparison_capture_registration,
    read_dm_comparison_family_manifest,
    read_dm_comparison_physics_input,
    register_dm_comparison_capture_ensemble,
)


MODELS = ("cdm", "sidm", "fdm")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _capture_rows() -> list[dict[str, object]]:
    unit_length = (1.0 * u.pc).to_value(u.cm)
    unit_velocity = (1.0 * u.pc / u.Myr).to_value(u.cm / u.s)
    unit_mass = (1.0 * u.Msun).to_value(u.g)
    speed = np.sqrt(G_INTERNAL * 2.0e8)
    uid = "capture-7-9"
    begin: dict[str, object] = {
        "schema_version": 1,
        "record_type": "event_begin",
        "event_uid": uid,
        "classification": "BINARY",
        "nstep_coarse": 10,
        "ilevel": 1,
        "nmember": 2,
        "expected_pairs": 1,
        "aexp": 0.5,
        "redshift": 1.0,
        "t_code": 0.2,
        "texp": 0.4,
        "merge_radius_code": 2.0,
        "unit_length_cgs": unit_length,
        "unit_velocity_cgs": unit_velocity,
        "unit_mass_cgs": unit_mass,
        "boxlen": 100.0,
        "complete": False,
    }
    rows: list[dict[str, object]] = [begin]
    for index, (sink_id, x, vy) in enumerate(
        ((7, 0.5, 0.5 * speed), (9, -0.5, -0.5 * speed)), start=1
    ):
        rows.append(
            {
                "schema_version": 1,
                "record_type": "member",
                "event_uid": uid,
                "member_index": index,
                "sink_id": sink_id,
                "mass_code": 1.0e8,
                "position_code": [x, 0.0, 0.0],
                "velocity_code": [0.0, vy, 0.0],
                "formation_time_code": 0.0,
                "accreted_mass_code": 0.0,
                "spin_magnitude": 0.5,
                "spin_direction": [0.0, 0.0, 1.0],
                "gas_angular_momentum_code": [0.0, 0.0, 0.0],
            }
        )
    rows.extend(
        (
            {
                "schema_version": 1,
                "record_type": "pair",
                "event_uid": uid,
                "pair_index": 1,
                "sink_id_1": 7,
                "sink_id_2": 9,
                "within_rmerge": True,
                "two_body_bound": True,
                "legacy_pair_bound": True,
            },
            {
                "schema_version": 1,
                "record_type": "event_end",
                "event_uid": uid,
                "nmember": 2,
                "npair": 1,
                "complete": True,
            },
        )
    )
    return rows


def _write_ledger(path: Path) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _capture_rows()),
        encoding="utf-8",
    )


def _provenance_records(model: str) -> dict[str, str]:
    records = {
        "dark_matter_model": model,
        "pic_enabled": ".true." if model in {"cdm", "sidm"} else ".false.",
        "sidm_enabled": ".true." if model == "sidm" else ".false.",
        "fdm_enabled": ".true." if model == "fdm" else ".false.",
        "nstep_coarse": "12",
        "time_code": "1.0d0",
        "aexp": "5.0d-1",
        "build_git_hash": "a" * 40,
        "namelist_copy": "namelist.txt",
        "compilation_copy": "compilation.txt",
        "smbh_capture_ledger_enabled": ".true.",
        "smbh_capture_ledger_file": "smbh_capture_ledger_v1.jsonl",
    }
    if model == "cdm":
        records["dm_transport"] = "collisionless_nbody"
        records["force_accounting"] = "resolved_collisionless_only"
    elif model == "sidm":
        records.update(
            {
                "sidm_cross_section_cm2_g": "1.0d0",
                "sidm_type": "constant",
                "sidm_v0_km_s": "100.0d0",
                "sidm_power": "-4.0d0",
                "sidm_angular": "isotropic",
                "sidm_inelastic": ".false.",
                "sidm_max_scatter_probability": "1.0d-2",
                "force_accounting": "resolved_collisionless_plus_scattering",
            }
        )
    else:
        records.update(
            {
                "m_axion_ev": "1.0d-22",
                "fdm_use_hjm": ".false.",
                "fdm_first_wave_level": "0",
                "fdm_outer_ledger_enabled": ".true.",
                "fdm_force_accounting": "resolved_wave_only",
            }
        )
    return records


def _write_provenance(path: Path, model: str) -> None:
    path.write_text(
        "# dm_run_provenance_v1\n"
        + "".join(f"{key} = {value}\n" for key, value in _provenance_records(model).items()),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    shared_paths: dict[str, Path] = {}
    for name, content in {
        "initial_conditions": b"matched-initial-conditions\n",
        "baryon_configuration": b"stars=available\ngas=absent\n",
        "smbh_seed_catalog": b"authoritative-seed-catalog\n",
    }.items():
        artifact = inputs / f"{name}.dat"
        artifact.write_bytes(content)
        shared_paths[name] = artifact
    provenance_paths: dict[str, Path] = {}
    ledger_paths: dict[str, Path] = {}
    for model in MODELS:
        run = tmp_path / "runs" / model
        output = run / "output_00012"
        output.mkdir(parents=True)
        ledger = run / "smbh_capture_ledger_v1.jsonl"
        _write_ledger(ledger)
        ledger_paths[model] = ledger
        provenance = output / "dm_run_provenance_00012.txt"
        _write_provenance(provenance, model)
        provenance_paths[model] = provenance
        (output / "COMPLETE").write_text("00012\n", encoding="utf-8")
    manifest = tmp_path / "family.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "family_id": "matched-three-model-smoke",
            "shared_inputs": {
                name: {"path": str(path.relative_to(tmp_path)), "sha256": _sha256(path)}
                for name, path in shared_paths.items()
            },
            "run_provenance": {
                model: str(path.relative_to(tmp_path)) for model, path in provenance_paths.items()
            },
        },
    )
    registration = tmp_path / "capture_registration.json"
    _write_json(
        registration,
        {
            "schema_version": 1,
            "family_manifest_path": "family.json",
            "family_manifest_sha256": _sha256(manifest),
            "captures": {
                model: {
                    "ledger_path": str(path.relative_to(tmp_path)),
                    "event_uid": "capture-7-9",
                }
                for model, path in ledger_paths.items()
            },
        },
    )
    return {"manifest": manifest, "registration": registration, **shared_paths}


def _write_physics_input(tmp_path: Path, ensemble: Path) -> Path:
    artifacts: dict[str, dict[str, dict[str, str]]] = {}
    names = {
        "cdm": ("environment_profile", "force_ledger", "conservation_ledger"),
        "sidm": (
            "environment_profile",
            "force_ledger",
            "conservation_ledger",
            "scattering_ledger",
        ),
        "fdm": (
            "environment_profile",
            "force_ledger",
            "conservation_ledger",
            "wave_ledger",
            "field_snapshot_index",
        ),
    }
    for model, model_names in names.items():
        artifacts[model] = {}
        for name in model_names:
            path = tmp_path / "artifacts" / model / f"{name}.dat"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{model} {name}\n", encoding="utf-8")
            artifacts[model][name] = {
                "path": str(path.relative_to(tmp_path)),
                "sha256": _sha256(path),
            }
    physics = tmp_path / "physics_inputs.json"
    _write_json(
        physics,
        {
            "schema_version": 1,
            "capture_ensemble_path": str(ensemble.relative_to(tmp_path)),
            "capture_ensemble_sha256": _sha256(ensemble),
            "artifacts": artifacts,
        },
    )
    return physics


def test_v1_physics_input_is_not_ready_without_normal_output_inventory_attestations(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    preflight = preflight_dm_comparison_family(read_dm_comparison_family_manifest(paths["manifest"]))
    assert preflight.ready
    smoke = assess_dm_comparison_smoke_outputs(preflight)
    assert smoke.verified
    ensemble = register_dm_comparison_capture_ensemble(
        read_dm_comparison_capture_registration(paths["registration"])
    )
    assert ensemble.registered
    ensemble_path = tmp_path / "capture_ensemble.json"
    _write_json(ensemble_path, ensemble.as_dict())
    assessment = assess_dm_comparison_physics_inputs(
        read_dm_comparison_physics_input(_write_physics_input(tmp_path, ensemble_path))
    )
    assert not assessment.ready_for_model_specific_analysis
    assert assessment.reasons == ("physics input schema lacks normal-output inventory assessments",)


def test_preflight_and_smoke_fail_closed_on_changed_input_or_incomplete_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["initial_conditions"].write_bytes(b"changed\n")
    preflight = preflight_dm_comparison_family(read_dm_comparison_family_manifest(paths["manifest"]))
    assert not preflight.ready
    assert "initial_conditions" in preflight.reasons[0]

    paths = _fixture(tmp_path / "fresh")
    marker = tmp_path / "fresh" / "runs" / "sidm" / "output_00012" / "COMPLETE"
    marker.write_text("wrong-output\n", encoding="utf-8")
    smoke = assess_dm_comparison_smoke_outputs(
        preflight_dm_comparison_family(read_dm_comparison_family_manifest(paths["manifest"]))
    )
    assert not smoke.verified
    assert any("sidm output COMPLETE marker" in reason for reason in smoke.reasons)


def test_physics_input_rejects_tampered_model_specific_evidence(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    ensemble = register_dm_comparison_capture_ensemble(
        read_dm_comparison_capture_registration(paths["registration"])
    )
    ensemble_path = tmp_path / "capture_ensemble.json"
    _write_json(ensemble_path, ensemble.as_dict())
    physics = _write_physics_input(tmp_path, ensemble_path)
    record = json.loads(physics.read_text(encoding="utf-8"))
    record["artifacts"]["fdm"]["wave_ledger"]["sha256"] = "0" * 64
    _write_json(physics, record)
    assessment = assess_dm_comparison_physics_inputs(read_dm_comparison_physics_input(physics))
    assert not assessment.ready_for_model_specific_analysis
    assert any("fdm wave_ledger" in reason for reason in assessment.reasons)


def test_cli_writes_one_atomic_record_for_each_contract_stage(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    preflight_path = tmp_path / "results" / "preflight.json"
    smoke_path = tmp_path / "results" / "smoke.json"
    ensemble_path = tmp_path / "results" / "ensemble.json"
    commands = (
        ("scripts/preflight_dm_comparison_family.py", paths["manifest"], preflight_path),
        ("scripts/assess_dm_comparison_smoke.py", paths["manifest"], smoke_path),
        ("scripts/register_dm_comparison_capture_ensemble.py", paths["registration"], ensemble_path),
    )
    for script, source, output in commands:
        subprocess.run(
            [sys.executable, script, str(source), str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(output.read_text(encoding="utf-8"))["status"].endswith(("ready", "verified", "registered"))
    physics_path = _write_physics_input(tmp_path, ensemble_path)
    output = tmp_path / "results" / "physics.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/assess_dm_comparison_physics_inputs.py",
            str(physics_path),
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "dm_comparison_physics_inputs_not_verified"
    assert "lacks normal-output inventory" in record["reasons"][0]


def test_family_preflight_can_gate_on_the_selected_writer_source(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    writer = tmp_path / "output_amr.kjhan.f90"
    writer.write_text(
        "write(unit,*) 'dm_transport = collisionless_nbody'\n"
        "write(unit,*) 'sidm_max_scatter_probability = ', pmax_value\n",
        encoding="utf-8",
    )
    output = tmp_path / "results" / "preflight-with-writer.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/preflight_dm_comparison_family.py",
            str(paths["manifest"]),
            str(output),
            "--writer-source",
            str(writer),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "comparison_family_preflight_not_ready"
    assert record["writer_source_audit"]["status"] == "source_token_missing"
    assert any("writer source" in reason for reason in record["reasons"])
