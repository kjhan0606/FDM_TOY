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
from fdm_smbh_delay.lagramses_cdm_orbit import extract_lagramses_cdm_pair_orbit_track


def _write_output(
    root: Path,
    number: int,
    *,
    time_code: float,
    primary_x: float,
    secondary_x: float,
    merge_radius: float = 0.0,
    namelist: str,
    unit_m: float | None = None,
    unit_d: float | None = None,
) -> Path:
    label = f"{number:05d}"
    directory = root / f"output_{label}"
    directory.mkdir(parents=True)
    mode = "no_finite_radius_rmerge_zero" if merge_radius == 0.0 else "enabled"
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
        "build_git_hash = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "namelist_copy = namelist.txt\n"
        "compilation_copy = compilation.txt\n"
        "smbh_capture_ledger_enabled = .true.\n"
        "smbh_capture_ledger_file = zoom_capture.jsonl\n"
        f"smbh_merge_radius_cells = {merge_radius:.1f}d0\n"
        f"smbh_compaction_mode = {mode}\n"
        "dm_transport = collisionless_nbody\n"
        "force_accounting = resolved_collisionless_only\n",
        encoding="utf-8",
    )
    info = (
        f"time = {time_code:.7f}d0\n"
        "aexp = 5.0d-1\n"
        "unit_l = 3.0856775814913673d18\n"
        "unit_t = 3.15576d13\n"
        "boxlen = 100.0d0\n"
    )
    if unit_m is not None:
        info += f"unit_m = {unit_m:.16e}\n"
    if unit_d is not None:
        info += f"unit_d = {unit_d:.16e}\n"
    (directory / f"info_{label}.txt").write_text(info, encoding="utf-8")
    (directory / f"sink_{label}.csv").write_text(
        f"1,1.0d8,{primary_x:.6f},0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
        f"2,5.0d7,{secondary_x:.6f},0.0,0.0,0.0,0.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (directory / "namelist.txt").write_text(namelist, encoding="utf-8")
    (directory / "compilation.txt").write_text("build provenance\n", encoding="utf-8")
    return directory


def _runtime_identity(root: Path, binding: Path, outputs: list[Path]) -> Path:
    namelist = root / "zoom.nml"
    base_namelist = (
        "&PHYSICS_PARAMS\n"
        "levelmax=21\n"
        "smbh=.true.\n"
        "rmerge=0.0d0\n"
        "smbh_capture_ledger=.true.\n"
        "smbh_capture_ledger_file='zoom_capture.jsonl'\n"
        "/\n"
    )
    namelist.write_text(base_namelist, encoding="utf-8")
    (root / "compilation_reference.txt").write_text("build provenance\n", encoding="utf-8")
    for name in (
        "host_orbit_initial_conditions",
        "initial_conditions",
        "baryon_configuration",
        "sink_initial_conditions",
    ):
        (root / f"{name}.dat").write_text(name + "\n", encoding="utf-8")
    plan = load_cdm_noncompacting_zoom_plan("configs/cdm_noncompacting_zoom_grid.yaml")
    contract_directory = root / "contract"
    arguments = dict(
        specification_path="configs/cdm_noncompacting_zoom_grid.yaml",
        case_id=plan.grid.cases[0].case_id,
        capture_binding_path=binding,
        capture_event_uid="capture-1-2",
        primary_sink_id=1,
        secondary_sink_id=2,
        run_namelist_path=namelist,
        capture_ledger_file="zoom_capture.jsonl",
        expected_build_git_hash="a" * 40,
        expected_compilation_path=root / "compilation_reference.txt",
        case_input_artifact_paths={
            name: root / f"{name}.dat"
            for name in (
                "host_orbit_initial_conditions",
                "initial_conditions",
                "baryon_configuration",
                "sink_initial_conditions",
            )
        },
    )
    provisional, _ = assess_cdm_noncompacting_zoom_run_inputs(**arguments)
    finalized_namelist = base_namelist.replace(
        "/\n",
        "".join(
            f"{name}='{value}'\n"
            for name, value in sorted(
                {
                    **provisional.execution_identity,
                    **provisional.model_execution_identity,
                }.items()
            )
        )
        + "/\n",
    )
    namelist.write_text(finalized_namelist, encoding="utf-8")
    for output in outputs:
        (output / "namelist.txt").write_text(finalized_namelist, encoding="utf-8")
        label = output.name.removeprefix("output_")
        provenance = output / f"dm_run_provenance_{label}.txt"
        provenance.write_text(
            provenance.read_text(encoding="utf-8")
            + "cdm_zoom_execution_identity_status = available\n"
            + "model_zoom_execution_identity_status = available\n"
            + "model_zoom_levelmax = 21\n"
            + "".join(
                f"{name} = {value}\n"
                for name, value in sorted(
                    {
                        **provisional.execution_identity,
                        **provisional.model_execution_identity,
                    }.items()
                )
            ),
            encoding="utf-8",
        )
    materialize_cdm_noncompacting_zoom_run_contract(
        **arguments,
        output_directory=contract_directory,
    )
    decision = assess_cdm_noncompacting_zoom_runtime_identity(
        contract_directory / "cdm_noncompacting_zoom_run_contract.json", outputs
    )
    identity = root / "runtime_identity.json"
    identity.write_text(json.dumps(decision.as_dict(), indent=2, sort_keys=True) + "\n")
    return identity


def _write_capture_binding(root: Path) -> Path:
    ledger_path = root / "smbh_capture_ledger_v1.jsonl"
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
    ledger_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    event = read_capture_ledger(ledger_path).events[0]
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
                    "ledger_path": str(ledger_path),
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


def test_extracts_periodic_comoving_relative_orbit_with_complete_provenance(
    tmp_path: Path,
) -> None:
    namelist = (
        "&PHYSICS_PARAMS\n"
        "levelmax=21\n"
        "smbh=.true.\n"
        "rmerge=0.0d0\n"
        "smbh_capture_ledger=.true.\n"
        "smbh_capture_ledger_file='zoom_capture.jsonl'\n"
        "/\n"
    )
    outputs = [
        _write_output(
            tmp_path,
            3,
            time_code=1.0002,
            primary_x=99.75,
            secondary_x=0.25,
            namelist=namelist,
            unit_d=1.98847e33 / (3.0856775814913673e18**3),
        ),
        _write_output(
            tmp_path,
            1,
            time_code=1.0,
            primary_x=99.0,
            secondary_x=1.0,
            namelist=namelist,
            unit_d=1.98847e33 / (3.0856775814913673e18**3),
        ),
        _write_output(
            tmp_path,
            2,
            time_code=1.0001,
            primary_x=99.5,
            secondary_x=0.5,
            namelist=namelist,
            unit_d=1.98847e33 / (3.0856775814913673e18**3),
        ),
    ]
    binding = _write_capture_binding(tmp_path)
    runtime_identity = _runtime_identity(tmp_path, binding, outputs)
    track = extract_lagramses_cdm_pair_orbit_track(runtime_identity)
    record = track.as_dict()
    assert [sample["output_number"] for sample in record["samples"]] == [
        "00001",
        "00002",
        "00003",
    ]
    assert [sample["separation_pc"] for sample in record["samples"]] == pytest.approx(
        [1.0, 0.5, 0.25]
    )
    assert record["samples"][0]["relative_position_pc"] == pytest.approx([1.0, 0.0, 0.0])
    assert len(record["source_outputs"]) == 3
    assert record["capture_binding"]["capture_event_sha256"]
    assert record["status"] == "raw_relative_orbit_track"
    assert all(sample["orbital_period_myr"] > 0.0 for sample in record["samples"])
    assert all(
        sample["orbital_period_method"]
        == "instantaneous_two_body_kepler_estimate"
        for sample in record["samples"]
    )


def test_refuses_compacting_or_incomplete_lagramses_outputs(tmp_path: Path) -> None:
    namelist = (
        "&PHYSICS_PARAMS\n"
        "levelmax=21\n"
        "smbh=.true.\n"
        "rmerge=0.0d0\n"
        "smbh_capture_ledger=.true.\n"
        "smbh_capture_ledger_file='zoom_capture.jsonl'\n"
        "/\n"
    )
    outputs = [
        _write_output(
            tmp_path, 1, time_code=1.0, primary_x=99.0, secondary_x=1.0, namelist=namelist
        ),
        _write_output(
            tmp_path, 2, time_code=1.0001, primary_x=99.5, secondary_x=0.5, namelist=namelist
        ),
        _write_output(
            tmp_path,
            3,
            time_code=1.0002,
            primary_x=99.75,
            secondary_x=0.25,
            merge_radius=1.0,
            namelist=namelist,
        ),
    ]
    binding = _write_capture_binding(tmp_path)
    runtime_identity = _runtime_identity(tmp_path, binding, outputs)
    with pytest.raises(ValueError, match="is not verified"):
        extract_lagramses_cdm_pair_orbit_track(runtime_identity)


def test_extractor_rereads_runtime_identity_outputs_before_using_them(tmp_path: Path) -> None:
    namelist = (
        "&PHYSICS_PARAMS\n"
        "levelmax=21\n"
        "smbh=.true.\n"
        "rmerge=0.0d0\n"
        "smbh_capture_ledger=.true.\n"
        "smbh_capture_ledger_file='zoom_capture.jsonl'\n"
        "/\n"
    )
    outputs = [
        _write_output(
            tmp_path, 1, time_code=1.0, primary_x=99.0, secondary_x=1.0, namelist=namelist
        ),
        _write_output(
            tmp_path, 2, time_code=1.0001, primary_x=99.5, secondary_x=0.5, namelist=namelist
        ),
        _write_output(
            tmp_path, 3, time_code=1.0002, primary_x=99.75, secondary_x=0.25, namelist=namelist
        ),
    ]
    runtime_identity = _runtime_identity(tmp_path, _write_capture_binding(tmp_path), outputs)
    (outputs[1] / "namelist.txt").write_text(
        namelist.replace("rmerge=0.0d0", "rmerge=1.0d0"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="namelist copy SHA-256 differs"):
        extract_lagramses_cdm_pair_orbit_track(runtime_identity)
