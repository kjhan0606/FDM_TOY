from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

from astropy import units as u
import numpy as np
import pytest
import yaml

from fdm_smbh_delay.capture_fdm_seed import (
    CaptureFDMSeedAssignment,
    CaptureFDMSeedFrame,
    CaptureFDMSeedFrameSpecification,
    CaptureSMBHMassProjection,
    derive_dual_smbh_sink_pair_from_capture,
    verify_mass_projection_source,
    verify_pure_fdm_seed_matches_capture_sink_pair,
)
from fdm_smbh_delay.capture_seed_binding import (
    assess_capture_seed_materialization_binding,
)
from fdm_smbh_delay.capture_seed_assembly import (
    assemble_capture_derived_pure_fdm_seed,
    capture_derived_seed_mapping,
    capture_soliton_configuration_from_mapping,
)
from fdm_smbh_delay.capture_ledger import read_capture_ledger
from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.dual_soliton_seed import (
    DualSolitonComponent,
    PureFDMDualSolitonSeed,
    load_pure_fdm_dual_soliton_seed,
    materialize_pure_fdm_dual_soliton_seed,
)


def _rows() -> list[dict[str, object]]:
    unit_length = (1.0 * u.pc).to_value(u.cm)
    unit_velocity = (1.0 * u.pc / u.Myr).to_value(u.cm / u.s)
    unit_mass = (1.0 * u.Msun).to_value(u.g)
    relative_speed = np.sqrt(G_INTERNAL * 2.0e8)
    uid = "capture-7-9"
    rows: list[dict[str, object]] = [
        {
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
    ]
    for index, (sink_id, x, vy) in enumerate(
        ((7, 0.5, 0.5 * relative_speed), (9, -0.5, -0.5 * relative_speed)),
        start=1,
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


def _event(tmp_path: Path, rows: list[dict[str, object]] | None = None):
    ledger = tmp_path / "capture.jsonl"
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows or _rows()),
        encoding="utf-8",
    )
    return read_capture_ledger(ledger).events[0], ledger


def _frame() -> CaptureFDMSeedFrame:
    return CaptureFDMSeedFrame(
        box_length_pc=10.0,
        mass_unit_msun=1.0e8,
        velocity_unit_pc_myr=10.0,
        origin_position_pc=(0.0, 0.0, 0.0),
        frame_velocity_pc_myr=(0.0, 0.0, 0.0),
    )


def _assignment() -> CaptureFDMSeedAssignment:
    return CaptureFDMSeedAssignment(
        primary_member_id=7,
        secondary_member_id=9,
        angular_momentum_code=((0.0, 0.0, 1.0e-5), (0.0, 0.0, -1.0e-5)),
    )


def _projection(
    *, source_path: str = "catalogs/smbh_masses.json", source_sha256: str = "a" * 64
) -> CaptureSMBHMassProjection:
    return CaptureSMBHMassProjection(
        event_uid="capture-7-9",
        member_ids=(7, 9),
        smbh_masses_msun=(0.9e8, 0.8e8),
        source_case_id="hr5-smbh-catalog-case",
        source_sha256=source_sha256,
        source_path=source_path,
    )


def _soliton_configuration(center_one, center_two) -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": "capture-derived-seed",
        "dark_matter_model": "fdm",
        "stellar_status": "absent",
        "gas_status": "absent",
        "box_length_code": 1.0,
        "m_axion_ev": 1.0e-21,
        "profile_c": 0.091,
        "solitons": [
            {
                "rho0_code": 4.0,
                "core_radius_box": 0.1,
                "center_box": list(center_one),
                "velocity_code": [0.0, 0.0, 0.0],
                "phase_radians": 0.0,
            },
            {
                "rho0_code": 3.0,
                "core_radius_box": 0.1,
                "center_box": list(center_two),
                "velocity_code": [0.0, 0.0, 0.0],
                "phase_radians": 1.0,
            },
        ],
    }


def test_capture_binary_derives_from_ledger_kinematics_and_explicit_mass_projection(
    tmp_path: Path,
) -> None:
    event, _ = _event(tmp_path)
    derived = derive_dual_smbh_sink_pair_from_capture(
        event,
        frame=_frame(),
        assignment=_assignment(),
        mass_projection=_projection(),
    )
    assert derived.member_ids == (7, 9)
    assert derived.sinks[0].sink_mass_code == pytest.approx(1.0)
    assert derived.sinks[0].smbh_mass_code == pytest.approx(0.9)
    assert derived.sinks[1].smbh_mass_code == pytest.approx(0.8)
    assert derived.sinks[0].center_box == pytest.approx((0.55, 0.5, 0.5))
    assert derived.sinks[0].dark_matter_fraction == 0.0
    assert "smbh_mass_code" not in _rows()[1]
    assert "sink_dark_matter_fraction" not in _rows()[1]
    assert derived.as_dict()["requirements"]["soliton_components_required"] == 2
    assert derived.as_dict()["mass_projection"]["source_case_id"] == "hr5-smbh-catalog-case"


def test_capture_pair_rejects_mass_projection_for_another_event_or_member_order(
    tmp_path: Path,
) -> None:
    event, _ = _event(tmp_path)
    with pytest.raises(ValueError, match="event_uid does not match"):
        derive_dual_smbh_sink_pair_from_capture(
            event,
            frame=_frame(),
            assignment=_assignment(),
            mass_projection=CaptureSMBHMassProjection(
                event_uid="another-event",
                member_ids=(7, 9),
                smbh_masses_msun=(0.9e8, 0.8e8),
                source_case_id="catalog",
                source_sha256="b" * 64,
                source_path="catalog.json",
            ),
        )
    with pytest.raises(ValueError, match="member_ids do not match"):
        derive_dual_smbh_sink_pair_from_capture(
            event,
            frame=_frame(),
            assignment=_assignment(),
            mass_projection=CaptureSMBHMassProjection(
                event_uid="capture-7-9",
                member_ids=(9, 7),
                smbh_masses_msun=(0.8e8, 0.9e8),
                source_case_id="catalog",
                source_sha256="b" * 64,
                source_path="catalog.json",
            ),
        )


def test_mass_projection_source_requires_the_recorded_hash(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"masses":[90000000.0,80000000.0]}\n', encoding="utf-8")
    digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
    assert verify_mass_projection_source(
        _projection(source_path="catalog.json", source_sha256=digest),
        reference_directory=tmp_path,
    ) == catalog.resolve()
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        verify_mass_projection_source(
            _projection(source_path="catalog.json", source_sha256="a" * 64),
            reference_directory=tmp_path,
        )


def test_frame_specification_binds_projection_to_event_and_member_order() -> None:
    with pytest.raises(ValueError, match="event_uid must match"):
        CaptureFDMSeedFrameSpecification(
            event_uid="capture-7-9",
            frame=_frame(),
            assignment=_assignment(),
            mass_projection=CaptureSMBHMassProjection(
                event_uid="different-event",
                member_ids=(7, 9),
                smbh_masses_msun=(0.9e8, 0.8e8),
                source_case_id="catalog",
                source_sha256="b" * 64,
                source_path="catalog.json",
            ),
        )
    with pytest.raises(ValueError, match="member_ids must match"):
        CaptureFDMSeedFrameSpecification(
            event_uid="capture-7-9",
            frame=_frame(),
            assignment=_assignment(),
            mass_projection=CaptureSMBHMassProjection(
                event_uid="capture-7-9",
                member_ids=(9, 7),
                smbh_masses_msun=(0.8e8, 0.9e8),
                source_case_id="catalog",
                source_sha256="b" * 64,
                source_path="catalog.json",
            ),
        )


def test_completed_dual_soliton_seed_must_preserve_capture_derived_sink_rows(
    tmp_path: Path,
) -> None:
    event, _ = _event(tmp_path)
    derived = derive_dual_smbh_sink_pair_from_capture(
        event,
        frame=_frame(),
        assignment=_assignment(),
        mass_projection=_projection(),
    )
    seed = PureFDMDualSolitonSeed(
        case_id="capture-seed",
        dark_matter_model="fdm",
        stellar_status="absent",
        gas_status="absent",
        box_length_code=1.0,
        m_axion_ev=1.0e-21,
        profile_c=0.091,
        solitons=(
            DualSolitonComponent(4.0, 0.1, derived.sinks[0].center_box, (0.0, 0.0, 0.0), 0.0),
            DualSolitonComponent(3.0, 0.1, derived.sinks[1].center_box, (0.0, 0.0, 0.0), 1.0),
        ),
        sinks=derived.sinks,
    )
    assert verify_pure_fdm_seed_matches_capture_sink_pair(seed, derived)["status"] == "capture_seed_identity_verified"


def test_capture_seed_assembly_accepts_only_explicit_solitons_and_injects_pair_rows(
    tmp_path: Path,
) -> None:
    event, _ = _event(tmp_path)
    derived = derive_dual_smbh_sink_pair_from_capture(
        event,
        frame=_frame(),
        assignment=_assignment(),
        mass_projection=_projection(),
    )
    configuration_record = _soliton_configuration(
        derived.sinks[0].center_box,
        derived.sinks[1].center_box,
    )
    configuration = capture_soliton_configuration_from_mapping(configuration_record)
    seed = assemble_capture_derived_pure_fdm_seed(derived, configuration)
    assert seed.sinks == derived.sinks
    assert "solver_policy" not in capture_derived_seed_mapping(seed)
    configuration_record["sinks"] = []
    with pytest.raises(ValueError, match="fields are invalid"):
        capture_soliton_configuration_from_mapping(configuration_record)


def test_materialized_seed_binding_rechecks_capture_pair_and_catalog_provenance(
    tmp_path: Path,
) -> None:
    event, _ = _event(tmp_path)
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"masses":[90000000.0,80000000.0]}\n', encoding="utf-8")
    derived = derive_dual_smbh_sink_pair_from_capture(
        event,
        frame=_frame(),
        assignment=_assignment(),
        mass_projection=_projection(
            source_path="catalog.json",
            source_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        ),
    )
    pair_record = derived.as_dict()
    pair_record["mass_projection_validation"] = {
        "status": "source_sha256_verified",
        "resolved_source_path": str(catalog.resolve()),
    }
    pair_path = tmp_path / "capture-sinks.json"
    pair_path.write_text(json.dumps(pair_record), encoding="utf-8")
    seed = PureFDMDualSolitonSeed(
        case_id="capture-seed",
        dark_matter_model="fdm",
        stellar_status="absent",
        gas_status="absent",
        box_length_code=1.0,
        m_axion_ev=1.0e-21,
        profile_c=0.091,
        solitons=(
            DualSolitonComponent(4.0, 0.1, derived.sinks[0].center_box, (0.0, 0.0, 0.0), 0.0),
            DualSolitonComponent(3.0, 0.1, derived.sinks[1].center_box, (0.0, 0.0, 0.0), 1.0),
        ),
        sinks=derived.sinks,
    )
    materialize_pure_fdm_dual_soliton_seed(seed, tmp_path / "seed")
    binding = assess_capture_seed_materialization_binding(
        capture_sink_pair_path=pair_path,
        seed_manifest_path=tmp_path / "seed" / "dual_soliton_seed_manifest.json",
    )
    assert binding.status == "capture_seed_materialization_identity_verified"

    sink_path = tmp_path / "seed" / "ic_sink"
    sink_path.write_text("0.0\n", encoding="utf-8")
    assert (
        assess_capture_seed_materialization_binding(
            capture_sink_pair_path=pair_path,
            seed_manifest_path=tmp_path / "seed" / "dual_soliton_seed_manifest.json",
        ).status
        == "capture_seed_materialization_identity_not_verified"
    )

    materialize_pure_fdm_dual_soliton_seed(seed, tmp_path / "seed-again")
    pair_record["sinks"][0]["smbh_mass_code"] = 0.7
    pair_path.write_text(json.dumps(pair_record), encoding="utf-8")
    assert (
        assess_capture_seed_materialization_binding(
            capture_sink_pair_path=pair_path,
            seed_manifest_path=tmp_path / "seed-again" / "dual_soliton_seed_manifest.json",
        ).status
        == "capture_seed_materialization_identity_not_verified"
    )


def test_capture_sink_pair_cli_writes_atomic_provenance_bound_output(tmp_path: Path) -> None:
    _, ledger = _event(tmp_path)
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"masses":[90000000.0,80000000.0]}\n', encoding="utf-8")
    specification = CaptureFDMSeedFrameSpecification(
        event_uid="capture-7-9",
        frame=_frame(),
        assignment=_assignment(),
        mass_projection=_projection(
            source_path="catalog.json",
            source_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        ),
    )
    specification_path = tmp_path / "frame.json"
    specification_path.write_text(json.dumps(specification.as_dict()), encoding="utf-8")
    output = tmp_path / "output" / "capture-sinks.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_capture_fdm_sink_pair.py",
            str(ledger),
            str(specification_path),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "capture_binary_dual_smbh_sink_pair_derived"
    assert record["mass_projection_validation"]["status"] == "source_sha256_verified"


def test_capture_seed_assembly_cli_writes_loadable_seed_and_pair_evidence(
    tmp_path: Path,
) -> None:
    _, ledger = _event(tmp_path)
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"masses":[90000000.0,80000000.0]}\n', encoding="utf-8")
    specification = CaptureFDMSeedFrameSpecification(
        event_uid="capture-7-9",
        frame=_frame(),
        assignment=_assignment(),
        mass_projection=_projection(
            source_path="catalog.json",
            source_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        ),
    )
    specification_path = tmp_path / "frame.json"
    specification_path.write_text(json.dumps(specification.as_dict()), encoding="utf-8")
    configuration_path = tmp_path / "solitons.yaml"
    configuration_path.write_text(
        yaml.safe_dump(_soliton_configuration((0.55, 0.5, 0.5), (0.45, 0.5, 0.5))),
        encoding="utf-8",
    )
    seed_path = tmp_path / "seed.yaml"
    pair_path = tmp_path / "pair.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/assemble_capture_dual_soliton_seed.py",
            str(ledger),
            str(specification_path),
            str(configuration_path),
            str(seed_path),
            str(pair_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert load_pure_fdm_dual_soliton_seed(seed_path).sinks[0].smbh_mass_code == pytest.approx(0.9)
    assert json.loads(pair_path.read_text(encoding="utf-8"))["mass_projection_validation"]["status"] == "source_sha256_verified"
