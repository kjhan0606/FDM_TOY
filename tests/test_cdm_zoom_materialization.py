from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdm_smbh_delay.capture_ledger import read_capture_ledger
from fdm_smbh_delay.cdm_zoom_materialization import (
    assess_cdm_noncompacting_zoom_run_inputs,
    materialize_cdm_noncompacting_zoom_run_contract,
)
from fdm_smbh_delay.cdm_zoom_plan import load_cdm_noncompacting_zoom_plan
from fdm_smbh_delay.cdm_zoom_runtime_identity import (
    assess_cdm_noncompacting_zoom_runtime_identity,
)


def _capture_binding(root: Path) -> Path:
    ledger = root / "original_capture_ledger.jsonl"
    rows = [
        {
            "schema_version": 1,
            "record_type": "event_begin",
            "event_uid": "capture-1-2",
            "classification": "BINARY",
            "nstep_coarse": 1,
            "ilevel": 1,
            "nmember": 2,
            "expected_pairs": 1,
            "aexp": 0.5,
            "redshift": 1.0,
            "t_code": 1.0,
            "texp": 1.0,
            "merge_radius_code": 1.0,
            "unit_length_cgs": 3.0856775814913673e18,
            "unit_velocity_cgs": 977792.2216807892,
            "unit_mass_cgs": 1.98847e33,
            "boxlen": 100.0,
            "complete": False,
        },
        {
            "schema_version": 1,
            "record_type": "member",
            "event_uid": "capture-1-2",
            "member_index": 1,
            "sink_id": 1,
            "mass_code": 1.0e8,
            "position_code": [0.0, 0.0, 0.0],
            "velocity_code": [0.0, 0.0, 0.0],
            "formation_time_code": 0.0,
            "accreted_mass_code": 0.0,
            "spin_magnitude": 0.0,
            "spin_direction": [0.0, 0.0, 1.0],
            "gas_angular_momentum_code": [0.0, 0.0, 0.0],
        },
        {
            "schema_version": 1,
            "record_type": "member",
            "event_uid": "capture-1-2",
            "member_index": 2,
            "sink_id": 2,
            "mass_code": 5.0e7,
            "position_code": [1.0, 0.0, 0.0],
            "velocity_code": [0.0, 0.0, 0.0],
            "formation_time_code": 0.0,
            "accreted_mass_code": 0.0,
            "spin_magnitude": 0.0,
            "spin_direction": [0.0, 0.0, 1.0],
            "gas_angular_momentum_code": [0.0, 0.0, 0.0],
        },
        {
            "schema_version": 1,
            "record_type": "pair",
            "event_uid": "capture-1-2",
            "pair_index": 1,
            "sink_id_1": 1,
            "sink_id_2": 2,
            "within_rmerge": True,
            "two_body_bound": True,
            "legacy_pair_bound": True,
        },
        {
            "schema_version": 1,
            "record_type": "event_end",
            "event_uid": "capture-1-2",
            "nmember": 2,
            "npair": 1,
            "complete": True,
        },
    ]
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    event = read_capture_ledger(ledger).events[0]
    binding = root / "capture_dm_run_binding.json"
    binding.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "capture_dm_run_provenance_bound",
                "interpretation": "capture-to-run provenance only",
                "capture_event": {
                    "event_uid": event.event_uid,
                    "event_sha256": event.event_sha256,
                    "ledger_path": str(ledger),
                },
                "run_provenance": {"dark_matter_model": "cdm"},
                "reasons": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return binding


def _namelist(
    *,
    rmerge: str = "0.0d0",
    ledger_file: str = "zoom_capture.jsonl",
    execution_identity: dict[str, str] | None = None,
) -> str:
    lines = [
        "&PHYSICS_PARAMS",
        "levelmax=21",
        "smbh=.true.",
        f"rmerge={rmerge}",
        "smbh_capture_ledger=.true.",
        f"smbh_capture_ledger_file='{ledger_file}'",
    ]
    if execution_identity is not None:
        lines.extend(f"{name}='{value}'" for name, value in sorted(execution_identity.items()))
    return "\n".join((*lines, "/", ""))


def _arguments(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    binding = _capture_binding(tmp_path)
    namelist = tmp_path / "zoom.nml"
    namelist.write_text(_namelist(), encoding="utf-8")
    compilation = tmp_path / "compilation.txt"
    compilation.write_text("build provenance\n", encoding="utf-8")
    artifacts = {}
    for name in (
        "host_orbit_initial_conditions",
        "initial_conditions",
        "baryon_configuration",
        "sink_initial_conditions",
    ):
        artifact = tmp_path / f"{name}.dat"
        artifact.write_text(name + "\n", encoding="utf-8")
        artifacts[name] = artifact
    plan = load_cdm_noncompacting_zoom_plan("configs/cdm_noncompacting_zoom_grid.yaml")
    arguments: dict[str, object] = {
        "specification_path": "configs/cdm_noncompacting_zoom_grid.yaml",
        "case_id": plan.grid.cases[0].case_id,
        "capture_binding_path": binding,
        "capture_event_uid": "capture-1-2",
        "primary_sink_id": 1,
        "secondary_sink_id": 2,
        "run_namelist_path": namelist,
        "capture_ledger_file": "zoom_capture.jsonl",
        "expected_build_git_hash": "a" * 40,
        "expected_compilation_path": compilation,
        "case_input_artifact_paths": artifacts,
    }
    provisional, _ = assess_cdm_noncompacting_zoom_run_inputs(**arguments)
    namelist.write_text(
        _namelist(
            execution_identity=(
                dict(provisional.execution_identity)
                | dict(provisional.model_execution_identity)
            )
        ),
        encoding="utf-8",
    )
    return arguments


def _write_runtime_output(
    root: Path,
    *,
    number: int,
    namelist: str,
    ledger_file: str = "zoom_capture.jsonl",
    time_code: float = 1.0,
    build_git_hash: str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    compilation_text: str = "build provenance\n",
) -> Path:
    label = f"{number:05d}"
    directory = root / f"output_{label}"
    directory.mkdir()
    cdm_identity_records = ""
    model_identity_records = ""
    for line in namelist.splitlines():
        if "=" not in line:
            continue
        name, value = (item.strip() for item in line.split("=", 1))
        if name.startswith("cdm_zoom_") and name.endswith("_sha256"):
            cdm_identity_records += f"{name} = {value.strip(chr(39) + chr(34))}\n"
        if name.startswith("model_zoom_"):
            model_identity_records += f"{name} = {value.strip(chr(39) + chr(34))}\n"
    (directory / "COMPLETE").write_text(label + "\n", encoding="utf-8")
    (directory / f"dm_run_provenance_{label}.txt").write_text(
        "# dm_run_provenance_v1\n"
        "dark_matter_model = cdm\n"
        "pic_enabled = .true.\n"
        "sidm_enabled = .false.\n"
        "fdm_enabled = .false.\n"
        f"nstep_coarse = {number}\n"
        f"time_code = {time_code:.7f}d0\n"
        "aexp = 5.0d-1\n"
        f"build_git_hash = {build_git_hash}\n"
        "namelist_copy = namelist.txt\n"
        "compilation_copy = compilation.txt\n"
        "smbh_capture_ledger_enabled = .true.\n"
        f"smbh_capture_ledger_file = {ledger_file}\n"
        "smbh_merge_radius_cells = 0.0d0\n"
        "smbh_compaction_mode = no_finite_radius_rmerge_zero\n"
        "dm_transport = collisionless_nbody\n",
        encoding="utf-8",
    )
    provenance = directory / f"dm_run_provenance_{label}.txt"
    identity_records = ""
    if cdm_identity_records:
        identity_records += "cdm_zoom_execution_identity_status = available\n" + cdm_identity_records
    if model_identity_records:
        identity_records += "model_zoom_execution_identity_status = available\n" + model_identity_records
    provenance.write_text(provenance.read_text(encoding="utf-8") + identity_records, encoding="utf-8")
    (directory / "namelist.txt").write_text(namelist, encoding="utf-8")
    (directory / "compilation.txt").write_text(compilation_text, encoding="utf-8")
    (directory / f"info_{label}.txt").write_text(
        f"time = {time_code:.7f}d0\n"
        "aexp = 5.0d-1\n"
        "unit_t = 3.15576d13\n",
        encoding="utf-8",
    )
    return directory


def test_materializes_exact_cdm_case_capture_and_noncompacting_namelist(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    original_namelist = Path(arguments["run_namelist_path"]).read_text(encoding="utf-8")
    record = materialize_cdm_noncompacting_zoom_run_contract(
        **arguments,
        output_directory=tmp_path / "contract",
    )
    written = json.loads(
        (tmp_path / "contract" / "cdm_noncompacting_zoom_run_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert record == written
    assert record["status"] == "ready_for_operator_submission"
    assert record["case"]["case_id"] == arguments["case_id"]
    assert record["capture_binding"]["capture_event_uid"] == "capture-1-2"
    assert record["capture_binding"]["primary_sink_id"] == 1
    assert record["plan"]["manifest_sha256"]
    assert record["sampling_requirements"]["minimum_complete_outputs"] == 15
    controls = (tmp_path / "contract" / "required_smbh_controls.nml").read_text(
        encoding="utf-8"
    )
    assert "rmerge=0.0d0" in controls
    assert "smbh_capture_ledger_file='zoom_capture.jsonl'" in controls
    assert "model_zoom_case_id='" + str(arguments["case_id"]) + "'" in controls
    assert "&PHYSICS_PARAMS" in controls
    assert record["case_input_identity"]["execution_identity"]
    assert Path(arguments["run_namelist_path"]).read_text(encoding="utf-8") == original_namelist


def test_requires_solver_consumed_physics_group_for_noncompacting_controls(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    namelist = Path(arguments["run_namelist_path"])
    namelist.write_text(
        namelist.read_text(encoding="utf-8").replace("&PHYSICS_PARAMS", "&SINK_PARAMS"),
        encoding="utf-8",
    )
    decision, _ = assess_cdm_noncompacting_zoom_run_inputs(**arguments)
    assert decision.status == "not_ready_for_operator_submission"
    assert any("exactly one &PHYSICS_PARAMS group" in reason for reason in decision.reasons)


def test_marks_mismatched_namelist_not_ready_but_preserves_auditable_contract(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    Path(arguments["run_namelist_path"]).write_text(
        Path(arguments["run_namelist_path"]).read_text(encoding="utf-8").replace(
            "rmerge=0.0d0", "rmerge=1.0d0"
        ),
        encoding="utf-8",
    )
    decision, _ = assess_cdm_noncompacting_zoom_run_inputs(**arguments)
    assert decision.status == "not_ready_for_operator_submission"
    assert any("rmerge must be exactly" in reason for reason in decision.reasons)
    record = materialize_cdm_noncompacting_zoom_run_contract(
        **arguments,
        output_directory=tmp_path / "not-ready-contract",
    )
    assert record["status"] == "not_ready_for_operator_submission"

    arguments = _arguments(tmp_path / "different-level")
    Path(arguments["run_namelist_path"]).write_text(
        Path(arguments["run_namelist_path"]).read_text(encoding="utf-8").replace(
            "levelmax=21", "levelmax=22"
        ),
        encoding="utf-8",
    )
    decision, _ = assess_cdm_noncompacting_zoom_run_inputs(**arguments)
    assert decision.status == "not_ready_for_operator_submission"
    assert "levelmax differs from the selected CDM zoom case" in decision.reasons


def test_rejects_reusing_original_capture_ledger_as_zoom_output(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    original_ledger = tmp_path / "original_capture_ledger.jsonl"
    with pytest.raises(ValueError, match="must not overwrite the original capture ledger"):
        assess_cdm_noncompacting_zoom_run_inputs(
            **(arguments | {"capture_ledger_file": str(original_ledger)})
        )


def test_materializes_auditable_not_ready_contract_for_missing_namelist(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    missing = tmp_path / "missing.nml"
    arguments["run_namelist_path"] = missing
    record = materialize_cdm_noncompacting_zoom_run_contract(
        **arguments,
        output_directory=tmp_path / "missing-namelist-contract",
    )
    assert record["status"] == "not_ready_for_operator_submission"
    assert record["run_inputs"]["namelist"]["sha256"] is None
    assert any("cannot read lagRamses run namelist" in reason for reason in record["reasons"])


def test_runtime_identity_requires_completed_output_namelist_copy_to_match_contract(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    contract_directory = tmp_path / "contract"
    materialize_cdm_noncompacting_zoom_run_contract(
        **arguments,
        output_directory=contract_directory,
    )
    output = _write_runtime_output(
        tmp_path,
        number=1,
        namelist=Path(arguments["run_namelist_path"]).read_text(encoding="utf-8"),
    )
    decision = assess_cdm_noncompacting_zoom_runtime_identity(
        contract_directory / "cdm_noncompacting_zoom_run_contract.json", [output]
    )
    record = decision.as_dict()
    assert decision.verified
    assert record["complete_outputs"][0]["output_number"] == "00001"
    assert record["listed_output_count"] == {
        "count": 1,
        "minimum_complete_outputs": 15,
        "status": "below_minimum_planned_output_count",
    }

    mismatched = _write_runtime_output(
        tmp_path,
        number=2,
        namelist=_namelist(
            rmerge="1.0d0",
            execution_identity=dict(
                assess_cdm_noncompacting_zoom_run_inputs(**arguments)[0].execution_identity
                | assess_cdm_noncompacting_zoom_run_inputs(**arguments)[0].model_execution_identity
            ),
        ),
    )
    rejected = assess_cdm_noncompacting_zoom_runtime_identity(
        contract_directory / "cdm_noncompacting_zoom_run_contract.json", [mismatched]
    )
    assert not rejected.verified
    assert any("namelist copy SHA-256 differs" in reason for reason in rejected.reasons)


def test_runtime_identity_rejects_changed_output_ledger_setting(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    contract_directory = tmp_path / "contract"
    materialize_cdm_noncompacting_zoom_run_contract(
        **arguments,
        output_directory=contract_directory,
    )
    output = _write_runtime_output(
        tmp_path,
        number=1,
        namelist=Path(arguments["run_namelist_path"]).read_text(encoding="utf-8"),
        ledger_file="other_zoom_capture.jsonl",
    )
    decision = assess_cdm_noncompacting_zoom_runtime_identity(
        contract_directory / "cdm_noncompacting_zoom_run_contract.json", [output]
    )
    assert not decision.verified
    assert any("capture-ledger setting differs" in reason for reason in decision.reasons)


def test_runtime_identity_rejects_output_with_different_executed_ic_attestation(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    contract_directory = tmp_path / "contract"
    materialize_cdm_noncompacting_zoom_run_contract(
        **arguments,
        output_directory=contract_directory,
    )
    output = _write_runtime_output(
        tmp_path,
        number=1,
        namelist=Path(arguments["run_namelist_path"]).read_text(encoding="utf-8"),
    )
    provenance = output / "dm_run_provenance_00001.txt"
    expected = assess_cdm_noncompacting_zoom_run_inputs(**arguments)[0].execution_identity[
        "cdm_zoom_initial_conditions_sha256"
    ]
    provenance.write_text(
        provenance.read_text(encoding="utf-8").replace(
            f"cdm_zoom_initial_conditions_sha256 = {expected}\n",
            "cdm_zoom_initial_conditions_sha256 = " + "f" * 64 + "\n",
        ),
        encoding="utf-8",
    )
    decision = assess_cdm_noncompacting_zoom_runtime_identity(
        contract_directory / "cdm_noncompacting_zoom_run_contract.json", [output]
    )
    assert not decision.verified
    assert any("cdm_zoom_initial_conditions_sha256 differs" in reason for reason in decision.reasons)


def test_runtime_identity_refuses_a_contract_with_changed_case_input_artifact(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    contract_directory = tmp_path / "contract"
    materialize_cdm_noncompacting_zoom_run_contract(
        **arguments,
        output_directory=contract_directory,
    )
    artifact = Path(arguments["case_input_artifact_paths"]["initial_conditions"])
    artifact.write_text("changed collisionless initial conditions\n", encoding="utf-8")
    output = _write_runtime_output(
        tmp_path,
        number=1,
        namelist=Path(arguments["run_namelist_path"]).read_text(encoding="utf-8"),
    )
    with pytest.raises(ValueError, match="input artifact initial_conditions SHA-256 no longer matches"):
        assess_cdm_noncompacting_zoom_runtime_identity(
            contract_directory / "cdm_noncompacting_zoom_run_contract.json", [output]
        )


def test_runtime_identity_rejects_mixed_builds_and_cadence(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    contract_directory = tmp_path / "contract"
    materialize_cdm_noncompacting_zoom_run_contract(
        **arguments,
        output_directory=contract_directory,
    )
    namelist = Path(arguments["run_namelist_path"]).read_text(encoding="utf-8")
    first = _write_runtime_output(tmp_path, number=1, namelist=namelist, time_code=1.0)
    second = _write_runtime_output(
        tmp_path,
        number=2,
        namelist=namelist,
        time_code=1.0003,
    )
    changed_build = _write_runtime_output(
        tmp_path,
        number=3,
        namelist=namelist,
        time_code=1.0004,
        build_git_hash="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )
    changed_compilation = _write_runtime_output(
        tmp_path,
        number=4,
        namelist=namelist,
        time_code=1.0005,
        compilation_text="different build provenance\n",
    )
    decision = assess_cdm_noncompacting_zoom_runtime_identity(
        contract_directory / "cdm_noncompacting_zoom_run_contract.json",
        [first, second, changed_build, changed_compilation],
    )
    assert not decision.verified
    assert any("build_git_hash differs" in reason for reason in decision.reasons)
    assert any("compilation copy SHA-256 differs" in reason for reason in decision.reasons)
    assert any("cadence exceeds" in reason for reason in decision.reasons)
