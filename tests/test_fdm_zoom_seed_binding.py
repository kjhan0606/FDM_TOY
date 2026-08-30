from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from fdm_smbh_delay.capture_fdm_seed import (
    CaptureFDMSeedAssignment,
    CaptureFDMSeedFrame,
    CaptureFDMSeedFrameSpecification,
    CaptureSMBHMassProjection,
    derive_dual_smbh_sink_pair_from_capture,
    materialize_capture_derived_sink_pair_record,
)
from fdm_smbh_delay.capture_ledger import read_capture_ledger
from fdm_smbh_delay.capture_seed_binding import (
    assess_capture_seed_materialization_binding,
)
from fdm_smbh_delay.dual_soliton_seed import (
    DualSolitonComponent,
    PureFDMDualSolitonSeed,
    materialize_pure_fdm_dual_soliton_seed,
)
from fdm_smbh_delay.dual_soliton_preflight import (
    preflight_pure_fdm_dual_soliton_run,
)
from fdm_smbh_delay.fdm_zoom_seed_binding import (
    materialize_fdm_capture_seed_zoom_binding,
    materialize_fdm_declared_run_input_binding,
    read_verified_fdm_capture_seed_zoom_binding,
    read_verified_fdm_declared_run_input_binding,
)
from fdm_smbh_delay.model_zoom_materialization import (
    materialize_model_zoom_execution_contract,
)
from fdm_smbh_delay.zoom_calibration import load_zoom_grid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture_event(tmp_path: Path):
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
            "merge_radius_code": 2.0,
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
    path = tmp_path / "capture.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return read_capture_ledger(path).events[0], path


def _specification() -> dict[str, object]:
    return {
        "schema_version": 1,
        "replicates": 1,
        "baseline": {
            "host_stellar_mass_msun": 0.0,
            "host_scale_radius_pc": 500.0,
            "host_inner_slope": 1.0,
            "binary_total_mass_msun": 2.0e7,
            "mass_ratio_q": 0.3,
            "gas_fraction": 0.0,
            "gas_rotation_fraction": 0.0,
            "initial_orbit_eccentricity": 0.4,
            "initial_separation_pc": 2000.0,
            "nuclear_envelope_to_secondary_bh_mass": 0.0,
            "dark_matter_model": "fdm",
            "fdm_particle_mass_ev": 1.0e-21,
            "fdm_core_radius_pc": 200.0,
            "fdm_soliton_mass_msun": 1.0e10,
        },
        "variations": [],
        "numerics": [
            {
                "levelmax": 19,
                "finest_cell_size_pc": 1.0,
                "collisionless_particle_mass_msun": 8.0e3,
                "minimum_softening_pc": 0.5,
                "fdm_use_hjm": False,
                "fdm_first_wave_level": 1,
            },
            {
                "levelmax": 20,
                "finest_cell_size_pc": 0.5,
                "collisionless_particle_mass_msun": 1.0e3,
                "minimum_softening_pc": 0.25,
                "fdm_use_hjm": False,
                "fdm_first_wave_level": 1,
            }
        ],
    }


def _prepare(
    tmp_path: Path,
    *,
    seed_axion_mass_ev: float = 1.0e-21,
    include_first_wave_level: bool = True,
) -> tuple[Path, Path]:
    event, ledger = _capture_event(tmp_path)
    catalog = tmp_path / "smbh_catalog.json"
    catalog.write_text('{"masses":[90000000.0,80000000.0]}\n', encoding="utf-8")
    frame_specification = CaptureFDMSeedFrameSpecification(
        event_uid=event.event_uid,
        frame=CaptureFDMSeedFrame(
            box_length_pc=10.0,
            mass_unit_msun=1.0e8,
            velocity_unit_pc_myr=10.0,
            origin_position_pc=(0.0, 0.0, 0.0),
            frame_velocity_pc_myr=(0.0, 0.0, 0.0),
        ),
        assignment=CaptureFDMSeedAssignment(
            primary_member_id=1,
            secondary_member_id=2,
            angular_momentum_code=((0.0, 0.0, 1.0e-5), (0.0, 0.0, -1.0e-5)),
        ),
        mass_projection=CaptureSMBHMassProjection(
            event_uid=event.event_uid,
            member_ids=(1, 2),
            smbh_masses_msun=(0.9e8, 0.8e8),
            source_case_id="capture-source",
            source_sha256=_sha256(catalog),
            source_path=str(catalog),
        ),
    )
    derived = derive_dual_smbh_sink_pair_from_capture(
        event,
        frame=frame_specification.frame,
        assignment=frame_specification.assignment,
        mass_projection=frame_specification.mass_projection,
    )
    frame_path = tmp_path / "capture_frame.json"
    frame_path.write_text(json.dumps(frame_specification.as_dict()), encoding="utf-8")
    pair_record = materialize_capture_derived_sink_pair_record(
        derived, frame_specification_path=frame_path
    )
    pair_path = tmp_path / "capture_sink_pair.json"
    pair_path.write_text(json.dumps(pair_record), encoding="utf-8")
    seed = PureFDMDualSolitonSeed(
        case_id="capture-seed",
        dark_matter_model="fdm",
        stellar_status="absent",
        gas_status="absent",
        box_length_code=1.0,
        m_axion_ev=seed_axion_mass_ev,
        profile_c=0.091,
        solitons=(
            DualSolitonComponent(4.0, 0.1, derived.sinks[0].center_box, (0.0, 0.0, 0.0), 0.0),
            DualSolitonComponent(3.0, 0.1, derived.sinks[1].center_box, (0.0, 0.0, 0.0), 1.0),
        ),
        sinks=derived.sinks,
    )
    materialize_pure_fdm_dual_soliton_seed(seed, tmp_path / "seed")
    seed_manifest = tmp_path / "seed" / "dual_soliton_seed_manifest.json"
    capture_binding = assess_capture_seed_materialization_binding(
        capture_sink_pair_path=pair_path, seed_manifest_path=seed_manifest
    )
    assert capture_binding.verified
    capture_binding_path = tmp_path / "capture_seed_binding.json"
    capture_binding_path.write_text(json.dumps(capture_binding.as_dict()), encoding="utf-8")

    specification = tmp_path / "fdm_zoom.yaml"
    specification.write_text(yaml.safe_dump(_specification()), encoding="utf-8")
    grid = load_zoom_grid(specification)
    inputs = {}
    for name in ("initial_conditions", "baryon_configuration"):
        artifact = tmp_path / f"{name}.dat"
        artifact.write_text(name + "\n", encoding="utf-8")
        inputs[name] = artifact
    inputs["smbh_seed_catalog"] = seed_manifest
    case = grid.cases[0]
    identity = {
        "model_zoom_manifest_sha256": grid.manifest_sha256,
        "model_zoom_case_id": case.case_id,
        "model_zoom_capture_event_sha256": event.event_sha256,
        "model_zoom_initial_conditions_sha256": _sha256(inputs["initial_conditions"]),
        "model_zoom_baryon_configuration_sha256": _sha256(inputs["baryon_configuration"]),
        "model_zoom_sink_initial_conditions_sha256": _sha256(seed_manifest),
    }
    namelist = tmp_path / "run.nml"
    namelist.write_text(
        "\n".join(
            (
                "&PHYSICS_PARAMS",
                *(f"{key}='{value}'" for key, value in sorted(identity.items())),
                "/",
                "&RUN_PARAMS",
                "use_fdm=.true.",
                "poisson=.true.",
                "sink=.true.",
                "hydro=.false.",
                "/",
                "&AMR_PARAMS",
                f"boxlen={seed.box_length_code:.16e}",
                "/",
                "&FDM_PARAMS",
                "fdm_dual_soliton_ic=.true.",
                "fdm_use_hjm=.false.",
                "fdm_outer_ledger=.true.",
                *(("fdm_first_wave_level=1",) if include_first_wave_level else ()),
                f"m_axion={seed.m_axion_ev:.16e}",
                f"fdm_dual_soliton_profile_c={seed.profile_c:.16e}",
                *(
                    assignment
                    for index, soliton in enumerate(seed.solitons, start=1)
                    for assignment in (
                        f"fdm_dual_soliton_rho0({index})={soliton.rho0_code:.16e}",
                        f"fdm_dual_soliton_rc_box({index})={soliton.core_radius_box:.16e}",
                        f"fdm_dual_soliton_phase({index})={soliton.phase_radians:.16e}",
                        *(
                            f"fdm_dual_soliton_center_box({index},{dimension})="
                            f"{soliton.center_box[dimension - 1]:.16e}"
                            for dimension in range(1, 4)
                        ),
                        *(
                            f"fdm_dual_soliton_velocity({index},{dimension})="
                            f"{soliton.velocity_code[dimension - 1]:.16e}"
                            for dimension in range(1, 4)
                        ),
                    )
                ),
                "/",
                "",
            )
        ),
        encoding="utf-8",
    )
    materialize_model_zoom_execution_contract(
        specification_path=specification,
        case_id=case.case_id,
        capture_ledger_path=ledger,
        capture_event_uid=event.event_uid,
        shared_input_paths=inputs,
        run_namelist_path=namelist,
        output_directory=tmp_path / "zoom_contract",
    )
    return (
        tmp_path / "zoom_contract" / "model_zoom_execution_contract.json",
        capture_binding_path,
    )


def test_materializes_and_rechecks_fdm_capture_seed_zoom_identity(tmp_path: Path) -> None:
    contract, capture_binding = _prepare(tmp_path)
    record = materialize_fdm_capture_seed_zoom_binding(
        model_zoom_contract_path=contract,
        capture_seed_binding_path=capture_binding,
        output_directory=tmp_path / "binding",
    )
    assert record["status"] == "fdm_capture_seed_zoom_identity_verified"
    verified = read_verified_fdm_capture_seed_zoom_binding(
        tmp_path / "binding" / "fdm_capture_seed_zoom_binding.json"
    )
    assert verified.verified
    assert verified.seed_case_id == "capture-seed"

    preflight = preflight_pure_fdm_dual_soliton_run(
        seed_manifest_path=tmp_path / "seed" / "dual_soliton_seed_manifest.json",
        run_namelist_path=tmp_path / "run.nml",
        run_ic_sink_path=tmp_path / "seed" / "ic_sink",
    )
    assert preflight.ready
    preflight_path = tmp_path / "dual_soliton_preflight.json"
    preflight_path.write_text(json.dumps(preflight.as_dict()), encoding="utf-8")
    declared = materialize_fdm_declared_run_input_binding(
        fdm_capture_seed_zoom_binding_path=tmp_path / "binding" / "fdm_capture_seed_zoom_binding.json",
        dual_soliton_preflight_path=preflight_path,
        output_directory=tmp_path / "declared-run",
    )
    assert declared["status"] == "fdm_declared_run_input_identity_verified"
    assert read_verified_fdm_declared_run_input_binding(
        tmp_path / "declared-run" / "fdm_declared_run_input_binding.json"
    ).verified

    record["seed_case_id"] = "tampered"
    (tmp_path / "binding" / "fdm_capture_seed_zoom_binding.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no longer matches"):
        read_verified_fdm_capture_seed_zoom_binding(
            tmp_path / "binding" / "fdm_capture_seed_zoom_binding.json"
        )


def test_censors_a_seed_with_the_wrong_axion_mass(tmp_path: Path) -> None:
    contract, capture_binding = _prepare(tmp_path, seed_axion_mass_ev=2.0e-21)
    record = materialize_fdm_capture_seed_zoom_binding(
        model_zoom_contract_path=contract,
        capture_seed_binding_path=capture_binding,
        output_directory=tmp_path / "binding",
    )
    assert record["status"] == "fdm_capture_seed_zoom_identity_not_verified"
    assert any("axion mass" in reason for reason in record["reasons"])


def test_declared_run_binding_requires_case_wave_level_in_fdm_params(tmp_path: Path) -> None:
    contract, capture_binding = _prepare(tmp_path, include_first_wave_level=False)
    materialize_fdm_capture_seed_zoom_binding(
        model_zoom_contract_path=contract,
        capture_seed_binding_path=capture_binding,
        output_directory=tmp_path / "binding",
    )
    preflight = preflight_pure_fdm_dual_soliton_run(
        seed_manifest_path=tmp_path / "seed" / "dual_soliton_seed_manifest.json",
        run_namelist_path=tmp_path / "run.nml",
        run_ic_sink_path=tmp_path / "seed" / "ic_sink",
    )
    assert preflight.ready
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight.as_dict()), encoding="utf-8")
    record = materialize_fdm_declared_run_input_binding(
        fdm_capture_seed_zoom_binding_path=tmp_path / "binding" / "fdm_capture_seed_zoom_binding.json",
        dual_soliton_preflight_path=preflight_path,
        output_directory=tmp_path / "declared-run",
    )
    assert record["status"] == "fdm_declared_run_input_identity_not_verified"
    assert any("fdm_first_wave_level" in reason for reason in record["reasons"])


def test_materializes_a_missing_source_as_a_nonverified_record(tmp_path: Path) -> None:
    output = tmp_path / "binding"
    record = materialize_fdm_capture_seed_zoom_binding(
        model_zoom_contract_path=tmp_path / "missing-contract.json",
        capture_seed_binding_path=tmp_path / "missing-capture-binding.json",
        output_directory=output,
    )
    assert record["status"] == "fdm_capture_seed_zoom_identity_not_verified"
    assert (output / "fdm_capture_seed_zoom_binding.json").is_file()
    assert record["sources"]["seed_manifest"]["sha256"] is None
