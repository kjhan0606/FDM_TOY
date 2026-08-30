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


def _namelist(*, rmerge: str = "0.0d0", ledger_file: str = "zoom_capture.jsonl") -> str:
    return "\n".join(
        (
            "&SINK_PARAMS",
            "smbh=.true.",
            f"rmerge={rmerge}",
            "smbh_capture_ledger=.true.",
            f"smbh_capture_ledger_file='{ledger_file}'",
            "/",
            "",
        )
    )


def _arguments(tmp_path: Path) -> dict[str, object]:
    binding = _capture_binding(tmp_path)
    namelist = tmp_path / "zoom.nml"
    namelist.write_text(_namelist(), encoding="utf-8")
    plan = load_cdm_noncompacting_zoom_plan("configs/cdm_noncompacting_zoom_grid.yaml")
    return {
        "specification_path": "configs/cdm_noncompacting_zoom_grid.yaml",
        "case_id": plan.grid.cases[0].case_id,
        "capture_binding_path": binding,
        "capture_event_uid": "capture-1-2",
        "primary_sink_id": 1,
        "secondary_sink_id": 2,
        "run_namelist_path": namelist,
        "capture_ledger_file": "zoom_capture.jsonl",
    }


def _write_runtime_output(
    root: Path,
    *,
    number: int,
    namelist: str,
    ledger_file: str = "zoom_capture.jsonl",
) -> Path:
    label = f"{number:05d}"
    directory = root / f"output_{label}"
    directory.mkdir()
    (directory / "COMPLETE").write_text(label + "\n", encoding="utf-8")
    (directory / f"dm_run_provenance_{label}.txt").write_text(
        "# dm_run_provenance_v1\n"
        "dark_matter_model = cdm\n"
        "pic_enabled = .true.\n"
        "sidm_enabled = .false.\n"
        "fdm_enabled = .false.\n"
        f"nstep_coarse = {number}\n"
        "time_code = 1.0d0\n"
        "aexp = 5.0d-1\n"
        "build_git_hash = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "namelist_copy = namelist.txt\n"
        "compilation_copy = compilation.txt\n"
        "smbh_capture_ledger_enabled = .true.\n"
        f"smbh_capture_ledger_file = {ledger_file}\n"
        "smbh_merge_radius_cells = 0.0d0\n"
        "smbh_compaction_mode = no_finite_radius_rmerge_zero\n"
        "dm_transport = collisionless_nbody\n",
        encoding="utf-8",
    )
    (directory / "namelist.txt").write_text(namelist, encoding="utf-8")
    (directory / "compilation.txt").write_text("build provenance\n", encoding="utf-8")
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
    assert Path(arguments["run_namelist_path"]).read_text(encoding="utf-8") == original_namelist


def test_marks_mismatched_namelist_not_ready_but_preserves_auditable_contract(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    Path(arguments["run_namelist_path"]).write_text(_namelist(rmerge="1.0d0"), encoding="utf-8")
    decision, _ = assess_cdm_noncompacting_zoom_run_inputs(**arguments)
    assert decision.status == "not_ready_for_operator_submission"
    assert any("rmerge must be exactly" in reason for reason in decision.reasons)
    record = materialize_cdm_noncompacting_zoom_run_contract(
        **arguments,
        output_directory=tmp_path / "not-ready-contract",
    )
    assert record["status"] == "not_ready_for_operator_submission"


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
    assert record["secular_sampling"] == {
        "complete_output_count": 1,
        "minimum_complete_outputs": 15,
        "status": "insufficient_complete_outputs",
    }

    mismatched = _write_runtime_output(
        tmp_path,
        number=2,
        namelist=_namelist(ledger_file="wrong_zoom_capture.jsonl"),
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
