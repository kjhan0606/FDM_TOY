from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import yaml

from fdm_smbh_delay.capture_ledger import read_capture_ledger
from fdm_smbh_delay.model_zoom_materialization import (
    materialize_model_zoom_execution_contract,
    read_verified_model_zoom_execution_contract,
)
from fdm_smbh_delay.zoom_calibration import load_zoom_grid


def _capture_ledger(path: Path) -> str:
    rows = [
        {
            "schema_version": 1,
            "record_type": "event_begin",
            "event_uid": "capture-1-2",
            "classification": "BINARY",
            "nstep_coarse": 4,
            "ilevel": 18,
            "nmember": 2,
            "expected_pairs": 1,
            "aexp": 0.5,
            "redshift": 1.0,
            "t_code": 0.1,
            "texp": 0.2,
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
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return read_capture_ledger(path).events[0].event_sha256


def _specification(model: str) -> dict[str, object]:
    baseline: dict[str, object] = {
        "host_stellar_mass_msun": 1.0e10,
        "host_scale_radius_pc": 500.0,
        "host_inner_slope": 1.0,
        "binary_total_mass_msun": 2.0e7,
        "mass_ratio_q": 0.3,
        "gas_fraction": 0.2,
        "gas_rotation_fraction": 0.5,
        "initial_orbit_eccentricity": 0.4,
        "initial_separation_pc": 2000.0,
        "nuclear_envelope_to_secondary_bh_mass": 5.0,
        "dark_matter_model": model,
    }
    if model == "sidm":
        baseline.update(
            sidm_cross_section_cm2_g=1.0,
            sidm_v0_km_s=100.0,
            sidm_power=-4.0,
            sidm_type="constant",
            sidm_angular="isotropic",
            sidm_inelastic=False,
        )
    if model == "fdm":
        baseline.update(
            fdm_particle_mass_ev=1.0e-22,
            fdm_core_radius_pc=200.0,
            fdm_soliton_mass_msun=1.0e10,
        )
    return {
        "schema_version": 1,
        "replicates": 1,
        "baseline": baseline,
        "variations": [],
        "numerics": [
            {
                "levelmax": 19,
                "finest_cell_size_pc": 1.0,
                "collisionless_particle_mass_msun": 8.0e3,
                "minimum_softening_pc": 0.5,
            },
            {
                "levelmax": 20,
                "finest_cell_size_pc": 0.5,
                "collisionless_particle_mass_msun": 1.0e3,
                "minimum_softening_pc": 0.25,
            },
        ],
    }


def _inputs(root: Path) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for name in ("initial_conditions", "baryon_configuration", "smbh_seed_catalog"):
        path = root / f"{name}.dat"
        path.write_text(name + "\n", encoding="utf-8")
        inputs[name] = path
    return inputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _namelist(
    *, grid_manifest_sha256: str, case_id: str, event_sha256: str, inputs: dict[str, Path]
) -> str:
    identity = {
        "model_zoom_manifest_sha256": grid_manifest_sha256,
        "model_zoom_case_id": case_id,
        "model_zoom_capture_event_sha256": event_sha256,
        "model_zoom_initial_conditions_sha256": _sha256(inputs["initial_conditions"]),
        "model_zoom_baryon_configuration_sha256": _sha256(inputs["baryon_configuration"]),
        "model_zoom_sink_initial_conditions_sha256": _sha256(inputs["smbh_seed_catalog"]),
    }
    return "\n".join(
        (
            "&PHYSICS_PARAMS",
            "smbh=.true.",
            *(f"{name}='{value}'" for name, value in sorted(identity.items())),
            "/",
            "",
        )
    )


@pytest.mark.parametrize("model", ("cdm", "sidm", "fdm"))
def test_materializes_common_identity_for_each_dark_matter_model(
    tmp_path: Path, model: str
) -> None:
    specification = tmp_path / f"{model}.yaml"
    specification.write_text(yaml.safe_dump(_specification(model)), encoding="utf-8")
    grid = load_zoom_grid(specification)
    ledger = tmp_path / "capture.jsonl"
    event_sha256 = _capture_ledger(ledger)
    inputs = _inputs(tmp_path)
    namelist = tmp_path / "run.nml"
    namelist.write_text(
        _namelist(
            grid_manifest_sha256=grid.manifest_sha256,
            case_id=grid.cases[0].case_id,
            event_sha256=event_sha256,
            inputs=inputs,
        ),
        encoding="utf-8",
    )
    record = materialize_model_zoom_execution_contract(
        specification_path=specification,
        case_id=grid.cases[0].case_id,
        capture_ledger_path=ledger,
        capture_event_uid="capture-1-2",
        shared_input_paths=inputs,
        run_namelist_path=namelist,
        output_directory=tmp_path / "contract",
    )
    assert record["status"] == "declared_namelist_identity_verified"
    assert record["dark_matter_model"] == model
    assert record["capture_event"]["event_sha256"] == event_sha256
    identity = record["model_zoom_execution_identity"]
    assert identity["model_zoom_case_id"] == grid.cases[0].case_id
    assert identity["model_zoom_baryon_configuration_sha256"]
    controls = (tmp_path / "contract" / "required_model_zoom_identity.assignments").read_text(
        encoding="utf-8"
    )
    assert not any(line.lstrip().startswith("&") for line in controls.splitlines())
    assert "/\n" not in controls
    assert identity["model_zoom_capture_event_sha256"] in controls
    verified = read_verified_model_zoom_execution_contract(
        tmp_path / "contract" / "model_zoom_execution_contract.json"
    )
    assert verified.case == grid.cases[0]
    namelist.write_text(namelist.read_text(encoding="utf-8") + "! later edit\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run namelist no longer matches"):
        read_verified_model_zoom_execution_contract(
            tmp_path / "contract" / "model_zoom_execution_contract.json"
        )


def test_rejects_missing_shared_input_and_existing_destination(tmp_path: Path) -> None:
    specification = tmp_path / "cdm.yaml"
    specification.write_text(yaml.safe_dump(_specification("cdm")), encoding="utf-8")
    grid = load_zoom_grid(specification)
    ledger = tmp_path / "capture.jsonl"
    event_sha256 = _capture_ledger(ledger)
    inputs = _inputs(tmp_path)
    namelist = tmp_path / "run.nml"
    namelist.write_text(
        _namelist(
            grid_manifest_sha256=grid.manifest_sha256,
            case_id=grid.cases[0].case_id,
            event_sha256=event_sha256,
            inputs=inputs,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shared inputs must name"):
        materialize_model_zoom_execution_contract(
            specification_path=specification,
            case_id=grid.cases[0].case_id,
            capture_ledger_path=ledger,
            capture_event_uid="capture-1-2",
            shared_input_paths={"initial_conditions": inputs["initial_conditions"]},
            run_namelist_path=namelist,
            output_directory=tmp_path / "contract",
        )
    materialize_model_zoom_execution_contract(
        specification_path=specification,
        case_id=grid.cases[0].case_id,
        capture_ledger_path=ledger,
        capture_event_uid="capture-1-2",
        shared_input_paths=inputs,
        run_namelist_path=namelist,
        output_directory=tmp_path / "contract",
    )
    with pytest.raises(ValueError, match="must not already exist"):
        materialize_model_zoom_execution_contract(
            specification_path=specification,
            case_id=grid.cases[0].case_id,
            capture_ledger_path=ledger,
            capture_event_uid="capture-1-2",
            shared_input_paths=inputs,
            run_namelist_path=namelist,
            output_directory=tmp_path / "contract",
        )


def test_rejects_a_second_physics_params_group(tmp_path: Path) -> None:
    specification = tmp_path / "cdm.yaml"
    specification.write_text(yaml.safe_dump(_specification("cdm")), encoding="utf-8")
    grid = load_zoom_grid(specification)
    ledger = tmp_path / "capture.jsonl"
    event_sha256 = _capture_ledger(ledger)
    inputs = _inputs(tmp_path)
    namelist = tmp_path / "run.nml"
    content = _namelist(
        grid_manifest_sha256=grid.manifest_sha256,
        case_id=grid.cases[0].case_id,
        event_sha256=event_sha256,
        inputs=inputs,
    )
    namelist.write_text(content + content, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one &PHYSICS_PARAMS group"):
        materialize_model_zoom_execution_contract(
            specification_path=specification,
            case_id=grid.cases[0].case_id,
            capture_ledger_path=ledger,
            capture_event_uid="capture-1-2",
            shared_input_paths=inputs,
            run_namelist_path=namelist,
            output_directory=tmp_path / "contract",
        )
