from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from astropy import units as u
import numpy as np
import pytest

from fdm_smbh_delay.capture_ledger import read_capture_ledger
from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.dm_run_provenance import (
    bind_capture_event_to_dark_matter_run,
    read_dark_matter_run_provenance,
)


def _capture_rows() -> list[dict[str, object]]:
    unit_length = (1.0 * u.pc).to_value(u.cm)
    unit_velocity = (1.0 * u.pc / u.Myr).to_value(u.cm / u.s)
    unit_mass = (1.0 * u.Msun).to_value(u.g)
    speed = np.sqrt(G_INTERNAL * 2.0e8)
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
    for index, (sink_id, x, vy) in enumerate(((7, 0.5, 0.5 * speed), (9, -0.5, -0.5 * speed)), start=1):
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


def _event(tmp_path: Path):
    path = tmp_path / "smbh_capture_ledger_v1.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _capture_rows()), encoding="utf-8")
    return read_capture_ledger(path).events[0], path


def _records(model: str, *, nstep: int = 12, ledger_name: str = "smbh_capture_ledger_v1.jsonl") -> dict[str, str]:
    records = {
        "dark_matter_model": model,
        "pic_enabled": ".true." if model in {"cdm", "sidm"} else ".false.",
        "sidm_enabled": ".true." if model == "sidm" else ".false.",
        "fdm_enabled": ".true." if model == "fdm" else ".false.",
        "nstep_coarse": str(nstep),
        "time_code": "1.0d0",
        "aexp": "5.0d-1",
        "build_git_hash": "a" * 40,
        "namelist_copy": "namelist.txt",
        "compilation_copy": "compilation.txt",
        "smbh_capture_ledger_enabled": ".true.",
        "smbh_capture_ledger_file": ledger_name,
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
    elif model == "fdm":
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


def _cdm_zoom_execution_identity() -> dict[str, str]:
    return {
        "cdm_zoom_execution_identity_status": "available",
        "cdm_zoom_plan_manifest_sha256": "1" * 64,
        "cdm_zoom_capture_event_sha256": "2" * 64,
        "cdm_zoom_host_orbit_initial_conditions_sha256": "3" * 64,
        "cdm_zoom_initial_conditions_sha256": "4" * 64,
        "cdm_zoom_sink_initial_conditions_sha256": "5" * 64,
    }


def _model_zoom_execution_identity(model: str) -> dict[str, str]:
    return {
        "model_zoom_execution_identity_status": "available",
        "model_zoom_manifest_sha256": "6" * 64,
        "model_zoom_case_id": f"{model}-fine-replicate-0",
        "model_zoom_levelmax": "20",
        "model_zoom_capture_event_sha256": "7" * 64,
        "model_zoom_initial_conditions_sha256": "8" * 64,
        "model_zoom_baryon_configuration_sha256": "a" * 64,
        "model_zoom_sink_initial_conditions_sha256": "9" * 64,
    }


def _write_provenance(path: Path, records: dict[str, str]) -> None:
    path.write_text(
        "# dm_run_provenance_v1\n" + "".join(f"{key} = {value}\n" for key, value in records.items()),
        encoding="utf-8",
    )


@pytest.mark.parametrize("model", ("cdm", "sidm", "fdm"))
def test_reads_each_dark_matter_realization_without_mixing_models(tmp_path: Path, model: str) -> None:
    path = tmp_path / f"{model}.txt"
    _write_provenance(path, _records(model))
    provenance = read_dark_matter_run_provenance(path)
    assert provenance.dark_matter_model == model
    if model == "fdm":
        assert provenance.parameter("fdm_force_accounting") == "resolved_wave_only"
    if model == "sidm":
        assert provenance.parameter("sidm_cross_section_cm2_g") == pytest.approx(1.0)


def test_cdm_force_accounting_is_explicitly_required(tmp_path: Path) -> None:
    records = _records("cdm")
    del records["force_accounting"]
    path = tmp_path / "cdm.txt"
    _write_provenance(path, records)
    with pytest.raises(ValueError, match="CDM run-provenance flags are inconsistent"):
        read_dark_matter_run_provenance(path)

    records["force_accounting"] = "analytic_drag"
    _write_provenance(path, records)
    with pytest.raises(ValueError, match="CDM run-provenance flags are inconsistent"):
        read_dark_matter_run_provenance(path)


def test_sidm_scatter_probability_gate_is_fail_closed(tmp_path: Path) -> None:
    records = _records("sidm")
    records["sidm_max_scatter_probability"] = "1.01d-1"
    path = tmp_path / "sidm.txt"
    _write_provenance(path, records)
    with pytest.raises(ValueError, match="exceeds the conservative 0.1 gate"):
        read_dark_matter_run_provenance(path)


def test_reads_complete_cdm_zoom_execution_identity_only_for_cdm(tmp_path: Path) -> None:
    path = tmp_path / "cdm.txt"
    _write_provenance(path, _records("cdm") | _cdm_zoom_execution_identity())
    provenance = read_dark_matter_run_provenance(path)
    assert provenance.parameter("cdm_zoom_execution_identity_status") == "available"
    assert provenance.parameter("cdm_zoom_initial_conditions_sha256") == "4" * 64

    missing = _records("cdm") | _cdm_zoom_execution_identity()
    del missing["cdm_zoom_sink_initial_conditions_sha256"]
    _write_provenance(path, missing)
    with pytest.raises(ValueError, match="requires all digests"):
        read_dark_matter_run_provenance(path)

    non_cdm = _records("fdm") | _cdm_zoom_execution_identity()
    _write_provenance(path, non_cdm)
    with pytest.raises(ValueError, match="cannot appear in a non-CDM run"):
        read_dark_matter_run_provenance(path)


def test_rejects_cdm_zoom_execution_digest_without_status(tmp_path: Path) -> None:
    records = _records("cdm") | _cdm_zoom_execution_identity()
    del records["cdm_zoom_execution_identity_status"]
    path = tmp_path / "cdm.txt"
    _write_provenance(path, records)
    with pytest.raises(ValueError, match="digests require a status"):
        read_dark_matter_run_provenance(path)


@pytest.mark.parametrize("model", ("cdm", "sidm", "fdm"))
def test_reads_common_execution_identity_for_each_model(tmp_path: Path, model: str) -> None:
    path = tmp_path / f"{model}.txt"
    _write_provenance(path, _records(model) | _model_zoom_execution_identity(model))
    provenance = read_dark_matter_run_provenance(path)
    assert provenance.parameter("model_zoom_execution_identity_status") == "available"
    assert provenance.parameter("model_zoom_case_id") == f"{model}-fine-replicate-0"

    incomplete = _records(model) | _model_zoom_execution_identity(model)
    del incomplete["model_zoom_capture_event_sha256"]
    _write_provenance(path, incomplete)
    with pytest.raises(ValueError, match="requires all fields"):
        read_dark_matter_run_provenance(path)


def test_capture_binds_only_to_a_later_output_with_the_same_ledger_name(tmp_path: Path) -> None:
    event, _ = _event(tmp_path)
    path = tmp_path / "cdm.txt"
    _write_provenance(path, _records("cdm"))
    decision = bind_capture_event_to_dark_matter_run(event, read_dark_matter_run_provenance(path))
    assert decision.bound
    assert decision.dark_matter_model == "cdm"

    _write_provenance(path, _records("sidm", nstep=9, ledger_name="other.jsonl"))
    rejected = bind_capture_event_to_dark_matter_run(event, read_dark_matter_run_provenance(path))
    assert not rejected.bound
    assert len(rejected.reasons) == 2

    other_directory = tmp_path / "other"
    other_directory.mkdir()
    _write_provenance(other_directory / "cdm.txt", _records("cdm"))
    same_name_elsewhere = bind_capture_event_to_dark_matter_run(
        event,
        read_dark_matter_run_provenance(other_directory / "cdm.txt"),
    )
    assert not same_name_elsewhere.bound
    assert "path does not resolve" in same_name_elsewhere.reasons[0]


def test_cli_writes_atomic_capture_run_binding(tmp_path: Path) -> None:
    _, ledger = _event(tmp_path)
    provenance = tmp_path / "fdm.txt"
    _write_provenance(provenance, _records("fdm"))
    output = tmp_path / "result" / "binding.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/bind_capture_dm_run.py",
            str(ledger),
            "capture-7-9",
            str(provenance),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "capture_dm_run_provenance_bound"
