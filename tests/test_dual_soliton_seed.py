from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from fdm_smbh_delay.dual_soliton_seed import (
    load_pure_fdm_dual_soliton_seed,
    materialize_pure_fdm_dual_soliton_seed,
)
from fdm_smbh_delay.dual_soliton_preflight import (
    preflight_pure_fdm_dual_soliton_run,
    read_verified_pure_fdm_dual_soliton_runtime_identity,
    validate_pure_fdm_dual_soliton_runtime_identity,
)


def _record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": "pure-fdm-dual-seed-test",
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
                "center_box": [0.35, 0.5, 0.5],
                "velocity_code": [0.0, 0.02, 0.0],
                "phase_radians": 0.0,
            },
            {
                "rho0_code": 3.0,
                "core_radius_box": 0.1,
                "center_box": [0.65, 0.5, 0.5],
                "velocity_code": [0.0, -0.02, 0.0],
                "phase_radians": 1.0,
            },
        ],
        "sinks": [
            {
                "sink_mass_code": 1.0e-4,
                "smbh_mass_code": 5.0e-5,
                "center_box": [0.35, 0.5, 0.5],
                "velocity_code": [0.0, 0.02, 0.0],
                "angular_momentum_code": [0.0, 0.0, 1.0e-5],
                "dark_matter_fraction": 0.0,
            },
            {
                "sink_mass_code": 8.0e-5,
                "smbh_mass_code": 2.5e-5,
                "center_box": [0.65, 0.5, 0.5],
                "velocity_code": [0.0, -0.02, 0.0],
                "angular_momentum_code": [0.0, 0.0, -1.0e-5],
                "dark_matter_fraction": 0.0,
            },
        ],
    }


def _write_seed(path: Path, record: dict[str, object] | None = None) -> None:
    path.write_text(yaml.safe_dump(record or _record()), encoding="utf-8")


def test_materializer_writes_matched_all_wave_namelist_and_two_sink_rows(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    seed = load_pure_fdm_dual_soliton_seed(seed_path)
    manifest = materialize_pure_fdm_dual_soliton_seed(seed, tmp_path / "materialized")

    namelist = Path(manifest["files"]["fdm_namelist_fragment"]["path"]).read_text(
        encoding="utf-8"
    )
    sink_rows = Path(manifest["files"]["ic_sink"]["path"]).read_text(
        encoding="utf-8"
    ).splitlines()
    saved_manifest = json.loads(
        (tmp_path / "materialized" / "dual_soliton_seed_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "fdm_use_hjm=.false." in namelist
    assert "fdm_outer_ledger=.true." in namelist
    assert "fdm_dual_soliton_ic=.true." in namelist
    assert len(sink_rows) == 2
    assert all(len(row.split()) == 12 for row in sink_rows)
    assert float(sink_rows[0].split()[1].replace("d", "e")) == pytest.approx(-0.15)
    assert saved_manifest == manifest
    assert manifest["seed"]["solver_policy"]["analytic_fdm_drag"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value.update({"dark_matter_model": "cdm"}),
            "dark_matter_model=fdm",
        ),
        (
            lambda value: value["sinks"][0].update({"dark_matter_fraction": 0.1}),
            "CDM mass fraction",
        ),
        (
            lambda value: value["sinks"][0].update({"center_box": [0.8, 0.5, 0.5]}),
            "outside its assigned soliton",
        ),
    ],
)
def test_seed_rejects_cdm_or_unbound_smbh_component(
    tmp_path: Path, mutator, message: str
) -> None:
    record = _record()
    mutator(record)
    path = tmp_path / "invalid.yaml"
    _write_seed(path, record)
    with pytest.raises(ValueError, match=message):
        load_pure_fdm_dual_soliton_seed(path)


def test_cli_materializes_new_directory_without_submission(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    output = tmp_path / "seed-output"
    completed = subprocess.run(
        [sys.executable, "scripts/materialize_dual_soliton_ic.py", str(seed_path), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(completed.stdout)
    assert manifest["status"] == "pure_fdm_dual_soliton_seed_materialized"
    assert (output / "ic_sink").is_file()
    assert (output / "fdm_dual_soliton_params.nml").is_file()


def _run_namelist(materialized: Path) -> str:
    return """&PHYSICS_PARAMS
  smbh=.true.
  rmerge=0.0d0
  smbh_capture_ledger=.true.
  smbh_capture_ledger_file='smbh_capture_ledger_v1.jsonl'
/
&RUN_PARAMS
  use_fdm=.true.
  poisson=.true.
  sink=.true.
  hydro=.false.
/
&AMR_PARAMS
  boxlen=1.0d0
/
""" + (materialized / "fdm_dual_soliton_params.nml").read_text(encoding="utf-8")


def _fortran(value: float) -> str:
    return f"{value:.16e}".replace("e", "d")


def _write_runtime_provenance(
    path: Path, seed, *, second_phase: float | None = None
) -> None:
    first, second = seed.solitons
    phase_2 = second.phase_radians if second_phase is None else second_phase
    values = {
        "time_code": "1.0d0",
        "aexp": "5.0d-1",
        "nstep_coarse": "42",
        "m_axion_ev": _fortran(seed.m_axion_ev),
        "hbar_code": "2.0d-3",
        "fdm_use_hjm": "F",
        "fdm_first_wave_level": "0",
        "fdm_dual_soliton_ic": "T",
        "fdm_dual_soliton_profile_c": _fortran(seed.profile_c),
        "fdm_dual_soliton_rho0": " ".join(
            _fortran(value) for value in (first.rho0_code, second.rho0_code)
        ),
        "fdm_dual_soliton_rc_box": " ".join(
            _fortran(value)
            for value in (first.core_radius_box, second.core_radius_box)
        ),
        "fdm_dual_soliton_center_box_1": " ".join(
            _fortran(value) for value in first.center_box
        ),
        "fdm_dual_soliton_center_box_2": " ".join(
            _fortran(value) for value in second.center_box
        ),
        "fdm_dual_soliton_velocity_1": " ".join(
            _fortran(value) for value in first.velocity_code
        ),
        "fdm_dual_soliton_velocity_2": " ".join(
            _fortran(value) for value in second.velocity_code
        ),
        "fdm_dual_soliton_phase": " ".join(
            _fortran(value) for value in (first.phase_radians, phase_2)
        ),
        "analytic_fdm_drag_enabled": "F",
        "force_accounting": "resolved_wave_only",
        "leaf_mass_code": "3.0d0",
        "integrated_current_code": "1.0d-2 -2.0d-2 3.0d-2",
        "leaf_cell_count": "100.0d0",
        "complete_current_stencil_cell_count": "98.0d0",
        "complete_current_stencil_fraction": "9.8d-1",
        "psi_snapshot_prefix": "fdm_00042.out",
    }
    path.write_text(
        "# fdm_outer_wave_provenance_v2\n"
        + "\n".join(f"{key} = {value}" for key, value in values.items())
        + "\n",
        encoding="utf-8",
    )


def test_run_preflight_requires_exact_seed_namelist_and_sink_input(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    seed = load_pure_fdm_dual_soliton_seed(seed_path)
    materialized = tmp_path / "materialized"
    materialize_pure_fdm_dual_soliton_seed(seed, materialized)
    namelist = tmp_path / "run.nml"
    namelist.write_text(_run_namelist(materialized), encoding="utf-8")
    decision = preflight_pure_fdm_dual_soliton_run(
        seed_manifest_path=materialized / "dual_soliton_seed_manifest.json",
        run_namelist_path=namelist,
        run_ic_sink_path=materialized / "ic_sink",
    )
    assert decision.ready
    assert decision.as_dict()["status"] == "ready_for_operator_submission"


def test_run_preflight_blocks_mismatch_without_censoring_a_physical_result(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    seed = load_pure_fdm_dual_soliton_seed(seed_path)
    materialized = tmp_path / "materialized"
    materialize_pure_fdm_dual_soliton_seed(seed, materialized)
    namelist = tmp_path / "run.nml"
    namelist.write_text(
        _run_namelist(materialized).replace("fdm_use_hjm=.false.", "fdm_use_hjm=.true."),
        encoding="utf-8",
    )
    changed_sink = tmp_path / "wrong_ic_sink"
    changed_sink.write_text(
        (materialized / "ic_sink").read_text(encoding="utf-8").replace(
            "1.0000000000000000d-04", "2.0000000000000000d-04"
        ),
        encoding="utf-8",
    )
    decision = preflight_pure_fdm_dual_soliton_run(
        seed_manifest_path=materialized / "dual_soliton_seed_manifest.json",
        run_namelist_path=namelist,
        run_ic_sink_path=changed_sink,
    )
    assert not decision.ready
    assert decision.status == "not_ready_for_operator_submission"
    assert any("fdm_use_hjm" in reason for reason in decision.reasons)
    assert any("run ic_sink" in reason for reason in decision.reasons)


def test_run_preflight_requires_solver_groups_and_byte_identical_sink_input(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    seed = load_pure_fdm_dual_soliton_seed(seed_path)
    materialized = tmp_path / "materialized"
    materialize_pure_fdm_dual_soliton_seed(seed, materialized)
    namelist = tmp_path / "wrong-group.nml"
    namelist.write_text(
        _run_namelist(materialized)
        .replace("  use_fdm=.true.\n", "")
        .replace("&FDM_PARAMS\n", "&FDM_PARAMS\n  use_fdm=.true.\n"),
        encoding="utf-8",
    )
    nearly_identical_sink = tmp_path / "nearly-identical-ic_sink"
    nearly_identical_sink.write_text(
        (materialized / "ic_sink")
        .read_text(encoding="utf-8")
        .replace("1.0000000000000000d-04", "1.0000000000001000d-04"),
        encoding="utf-8",
    )
    decision = preflight_pure_fdm_dual_soliton_run(
        seed_manifest_path=materialized / "dual_soliton_seed_manifest.json",
        run_namelist_path=namelist,
        run_ic_sink_path=nearly_identical_sink,
    )
    assert not decision.ready
    assert any("use_fdm in &RUN_PARAMS" in reason for reason in decision.reasons)
    assert any("SHA-256" in reason for reason in decision.reasons)


def test_run_preflight_requires_hydro_for_a_gas_available_seed(tmp_path: Path) -> None:
    record = _record()
    record["gas_status"] = "available"
    seed_path = tmp_path / "gas-seed.yaml"
    _write_seed(seed_path, record)
    seed = load_pure_fdm_dual_soliton_seed(seed_path)
    materialized = tmp_path / "materialized"
    materialize_pure_fdm_dual_soliton_seed(seed, materialized)
    namelist = tmp_path / "run.nml"
    namelist.write_text(_run_namelist(materialized), encoding="utf-8")
    decision = preflight_pure_fdm_dual_soliton_run(
        seed_manifest_path=materialized / "dual_soliton_seed_manifest.json",
        run_namelist_path=namelist,
        run_ic_sink_path=materialized / "ic_sink",
    )
    assert not decision.ready
    assert any("hydro in &RUN_PARAMS must be true" in reason for reason in decision.reasons)


def test_run_preflight_rejects_scalar_and_array_overrides_on_one_line(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    seed = load_pure_fdm_dual_soliton_seed(seed_path)
    materialized = tmp_path / "materialized"
    materialize_pure_fdm_dual_soliton_seed(seed, materialized)
    namelist = tmp_path / "run.nml"
    namelist.write_text(
        _run_namelist(materialized)
        .replace("fdm_use_hjm=.false.", "fdm_use_hjm=.false., fdm_use_hjm=.true.")
        .replace(
            "fdm_outer_ledger=.true.\n",
            "fdm_outer_ledger=.true.\n  fdm_dual_soliton_rho0=4.0d0,3.0d0\n",
        ),
        encoding="utf-8",
    )
    decision = preflight_pure_fdm_dual_soliton_run(
        seed_manifest_path=materialized / "dual_soliton_seed_manifest.json",
        run_namelist_path=namelist,
        run_ic_sink_path=materialized / "ic_sink",
    )
    assert not decision.ready
    assert any("fdm_use_hjm is assigned more than once" in reason for reason in decision.reasons)
    assert any("aliases an elementwise checked FDM array" in reason for reason in decision.reasons)


def test_run_preflight_accepts_ordinary_inline_namelist_comments(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    seed = load_pure_fdm_dual_soliton_seed(seed_path)
    materialized = tmp_path / "materialized"
    materialize_pure_fdm_dual_soliton_seed(seed, materialized)
    namelist = tmp_path / "run.nml"
    namelist.write_text(
        _run_namelist(materialized).replace(
            "fdm_use_hjm=.false.", "fdm_use_hjm=.false. ! all-wave seed"
        ),
        encoding="utf-8",
    )
    assert preflight_pure_fdm_dual_soliton_run(
        seed_manifest_path=materialized / "dual_soliton_seed_manifest.json",
        run_namelist_path=namelist,
        run_ic_sink_path=materialized / "ic_sink",
    ).ready


def test_preflight_cli_writes_atomic_configuration_decision(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    seed = load_pure_fdm_dual_soliton_seed(seed_path)
    materialized = tmp_path / "materialized"
    materialize_pure_fdm_dual_soliton_seed(seed, materialized)
    namelist = tmp_path / "run.nml"
    namelist.write_text(_run_namelist(materialized), encoding="utf-8")
    output = tmp_path / "result" / "preflight.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/preflight_dual_soliton_run.py",
            str(materialized / "dual_soliton_seed_manifest.json"),
            str(namelist),
            str(materialized / "ic_sink"),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ready_for_operator_submission"


def test_runtime_identity_binds_v2_raw_output_to_materialized_seed(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    seed = load_pure_fdm_dual_soliton_seed(seed_path)
    materialized = tmp_path / "materialized"
    materialize_pure_fdm_dual_soliton_seed(seed, materialized)
    provenance = tmp_path / "fdm_outer_wave_provenance.txt"
    _write_runtime_provenance(provenance, seed)
    identity = validate_pure_fdm_dual_soliton_runtime_identity(
        seed_manifest_path=materialized / "dual_soliton_seed_manifest.json",
        provenance_path=provenance,
    )
    assert identity.verified
    assert identity.as_dict()["status"] == "runtime_seed_identity_verified"

    saved = tmp_path / "runtime-identity.json"
    saved.write_text(json.dumps(identity.as_dict()), encoding="utf-8")
    assert read_verified_pure_fdm_dual_soliton_runtime_identity(saved).verified


def test_saved_runtime_identity_rejects_changed_raw_provenance(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    seed = load_pure_fdm_dual_soliton_seed(seed_path)
    materialized = tmp_path / "materialized"
    materialize_pure_fdm_dual_soliton_seed(seed, materialized)
    provenance = tmp_path / "fdm_outer_wave_provenance.txt"
    _write_runtime_provenance(provenance, seed)
    identity = validate_pure_fdm_dual_soliton_runtime_identity(
        seed_manifest_path=materialized / "dual_soliton_seed_manifest.json",
        provenance_path=provenance,
    )
    saved = tmp_path / "runtime-identity.json"
    saved.write_text(json.dumps(identity.as_dict()), encoding="utf-8")
    _write_runtime_provenance(provenance, seed, second_phase=1.2)
    with pytest.raises(ValueError, match="SHA-256 no longer matches"):
        read_verified_pure_fdm_dual_soliton_runtime_identity(saved)


def test_runtime_identity_refuses_changed_dual_soliton_phase(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    seed = load_pure_fdm_dual_soliton_seed(seed_path)
    materialized = tmp_path / "materialized"
    materialize_pure_fdm_dual_soliton_seed(seed, materialized)
    provenance = tmp_path / "fdm_outer_wave_provenance.txt"
    _write_runtime_provenance(provenance, seed, second_phase=1.2)
    identity = validate_pure_fdm_dual_soliton_runtime_identity(
        seed_manifest_path=materialized / "dual_soliton_seed_manifest.json",
        provenance_path=provenance,
    )
    assert not identity.verified
    assert any("phases" in reason for reason in identity.reasons)
