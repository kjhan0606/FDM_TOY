from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from astropy import units as u
import numpy as np
import pytest
import yaml

from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.capture_ledger import read_capture_ledger
from fdm_smbh_delay.model_specific_physics import (
    assess_model_specific_phase_ensemble,
    compare_model_specific_resolution_pair,
    read_resolved_model_physics_result,
)
from fdm_smbh_delay.dm_comparison import (
    assess_dm_comparison_physics_inputs,
    assess_dm_comparison_smoke_outputs,
    preflight_dm_comparison_family,
    read_dm_comparison_capture_registration,
    read_dm_comparison_family_manifest,
    read_dm_comparison_physics_input,
    read_verified_dm_comparison_capture_ensemble,
    register_dm_comparison_capture_ensemble,
)
from fdm_smbh_delay.fdm_outer_wave_ledger import FDMOuterWaveLedger
from fdm_smbh_delay.resolved_physics_inventory import (
    assess_lagramses_resolved_physics_inventory,
    read_lagramses_resolved_physics_inventory,
)
from fdm_smbh_delay.zoom_calibration import (
    GalaxyMergerZoomCase,
    ZoomNumerics,
    ZoomPhysicsPoint,
    build_zoom_grid,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _capture_rows() -> list[dict[str, object]]:
    """One valid numerical-capture transaction for binding tests."""

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


def _write_capture_ledger(path: Path) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _capture_rows()),
        encoding="utf-8",
    )


def _physics(model: str) -> ZoomPhysicsPoint:
    common = {
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
    if model == "fdm":
        common.update(
            fdm_particle_mass_ev=1.0e-22,
            fdm_core_radius_pc=200.0,
            fdm_soliton_mass_msun=1.0e10,
        )
    elif model == "sidm":
        common.update(
            sidm_cross_section_cm2_g=1.0,
            sidm_v0_km_s=100.0,
            sidm_power=-4.0,
            sidm_type="constant",
            sidm_angular="isotropic",
            sidm_inelastic=False,
        )
    return ZoomPhysicsPoint(**common)


def _case(model: str, *, finest_cell_size_pc: float, replicate: int = 0) -> GalaxyMergerZoomCase:
    return GalaxyMergerZoomCase(
        _physics(model),
        ZoomNumerics(
            levelmax=20 if finest_cell_size_pc < 1.0 else 19,
            finest_cell_size_pc=finest_cell_size_pc,
            collisionless_particle_mass_msun=1.0e3,
            minimum_softening_pc=finest_cell_size_pc / 2.0,
        ),
        replicate=replicate,
    )


def _write_ready_inventory_assessment(
    root: Path,
    model: str,
    *,
    model_zoom_identity: dict[str, str],
) -> tuple[Path, dict[str, Path]]:
    """Create a v2 inventory fixture with real, hash-bound ledger files."""

    label = "00042"
    output = root / "normal_outputs" / model / f"output_{label}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETE").write_text(label + "\n", encoding="utf-8")
    _write_capture_ledger(output / "smbh_capture_ledger_v1.jsonl")
    capture_event_sha256 = read_capture_ledger(
        output / "smbh_capture_ledger_v1.jsonl"
    ).events[0].event_sha256
    (output / "POISSON_PHI_VALID").write_text(
        "LAGRAMSES_POISSON_PHI_VALID_V1\n42 21 1.25 0.5\n", encoding="utf-8"
    )
    for name in ("stars_00042.out", "hydro_00042.out", "grav_00042.out", "sink_00042.info"):
        (output / name).touch()
    if model in {"cdm", "sidm"}:
        (output / "part_00042.out").touch()
    if model == "fdm":
        (output / "fdm_00042.out").touch()

    def artifact(name: str) -> tuple[Path, str]:
        path = output / name
        path.write_text(name + "\n", encoding="utf-8")
        return path, _sha256(path)

    force, force_sha = artifact("force_source_ledger_00042.json")
    conservation, conservation_sha = artifact("conservation_ledger_00042.json")
    ledgers = {"force_ledger": force, "conservation_ledger": conservation}
    values = {
        "output_number": label,
        "nstep_coarse": "42",
        "time_code": "1.25d0",
        "aexp": "5.0d-1",
        "dark_matter_model": model,
        "raw_snapshot_directory": f"output_{label}/",
        "completion_marker": "COMPLETE",
        "star_formation_enabled": ".true.",
        "stars_channel_status": "available",
        "stars_particle_snapshot_prefix": "stars_00042.out",
        "gas_channel_status": "available",
        "gas_snapshot_prefix": "hydro_00042.out",
        "dark_matter_channel_status": "available",
        "particle_snapshot_prefix": "part_00042.out" if model in {"cdm", "sidm"} else "none",
        "potential_snapshot_prefix": "grav_00042.out",
        "potential_checkpoint_status": "validated",
        "sink_info_file": "sink_00042.info",
        "force_source_ledger_status": "available",
        "force_source_ledger_reason": "measured_source_work",
        "force_source_ledger_path": force.name,
        "force_source_ledger_sha256": force_sha,
        "conservation_ledger_status": "available",
        "conservation_ledger_reason": "measured_time_series",
        "conservation_ledger_path": conservation.name,
        "conservation_ledger_sha256": conservation_sha,
    }
    if model == "sidm":
        scattering, scattering_sha = artifact("sidm_scattering_ledger_00042.json")
        ledgers["scattering_ledger"] = scattering
        values.update(
            {
                "sidm_scattering_ledger_status": "available",
                "sidm_scattering_ledger_reason": "measured_scatter_history",
                "sidm_scattering_ledger_path": scattering.name,
                "sidm_scattering_ledger_sha256": scattering_sha,
            }
        )
    if model == "fdm":
        wave = output / f"fdm_outer_wave_provenance_{label}.txt"
        wave.write_text(
            "# fdm_outer_wave_provenance_v2\n"
            "time_code = 1.25d0\n"
            "aexp = 5.0d-1\n"
            "nstep_coarse = 42\n"
            "m_axion_ev = 1.0d-22\n"
            "hbar_code = 2.0d-3\n"
            "fdm_use_hjm = F\n"
            "fdm_first_wave_level = 12\n"
            "analytic_fdm_drag_enabled = .false.\n"
            "force_accounting = resolved_wave_only\n"
            "leaf_mass_code = 3.0d0\n"
            "integrated_current_code = 1.0d-2 -2.0d-2 3.0d-2\n"
            "leaf_cell_count = 100.0\n"
            "complete_current_stencil_cell_count = 98.0\n"
            "complete_current_stencil_fraction = 0.98\n"
            "psi_snapshot_prefix = fdm_00042.out\n"
            "fdm_dual_soliton_ic = F\n"
            "fdm_dual_soliton_profile_c = 0.0\n"
            "fdm_dual_soliton_rho0 = 0.0 0.0\n"
            "fdm_dual_soliton_rc_box = 0.0 0.0\n"
            "fdm_dual_soliton_center_box_1 = 0.0 0.0 0.0\n"
            "fdm_dual_soliton_center_box_2 = 0.0 0.0 0.0\n"
            "fdm_dual_soliton_velocity_1 = 0.0 0.0 0.0\n"
            "fdm_dual_soliton_velocity_2 = 0.0 0.0 0.0\n"
            "fdm_dual_soliton_phase = 0.0 0.0\n",
            encoding="utf-8",
        )
        ledgers["wave_provenance"] = wave
        values.update(
            {
                "fdm_field_snapshot_status": "available",
                "fdm_field_snapshot_prefix": "fdm_00042.out",
                "fdm_wave_provenance_status": "available",
                "fdm_wave_provenance_path": f"output_{label}/" + wave.name,
                "fdm_force_accounting": "resolved_wave_only",
            }
        )
    inventory_path = output / f"resolved_physics_inventory_{label}.txt"
    inventory_path.write_text(
        "# lagramses_resolved_physics_inventory_v2\n"
        + "".join(f"{key} = {value}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    assessment = assess_lagramses_resolved_physics_inventory(
        read_lagramses_resolved_physics_inventory(inventory_path),
        stars_required=True,
        gas_required=True,
    )
    assert assessment.ready_for_registered_analysis
    assessment_path = output / "resolved_physics_inventory_assessment.json"
    _write_json(assessment_path, assessment.as_dict())
    provenance = output / f"dm_run_provenance_{label}.txt"
    common = (
        "# dm_run_provenance_v1\n"
        f"dark_matter_model = {model}\n"
        f"pic_enabled = {'.true.' if model in {'cdm', 'sidm'} else '.false.'}\n"
        f"sidm_enabled = {'.true.' if model == 'sidm' else '.false.'}\n"
        f"fdm_enabled = {'.true.' if model == 'fdm' else '.false.'}\n"
        "nstep_coarse = 42\n"
        "time_code = 1.25d0\n"
        "aexp = 5.0d-1\n"
        "build_git_hash = 1111111111111111111111111111111111111111\n"
        "namelist_copy = namelist.txt\n"
        "compilation_copy = compilation.txt\n"
        "smbh_capture_ledger_enabled = .true.\n"
        "smbh_capture_ledger_file = smbh_capture_ledger_v1.jsonl\n"
        "model_zoom_execution_identity_status = available\n"
        f"model_zoom_manifest_sha256 = {model_zoom_identity['manifest_sha256']}\n"
        f"model_zoom_case_id = {model_zoom_identity['case_id']}\n"
        f"model_zoom_capture_event_sha256 = {capture_event_sha256}\n"
        f"model_zoom_initial_conditions_sha256 = {model_zoom_identity['initial_conditions_sha256']}\n"
        f"model_zoom_baryon_configuration_sha256 = {model_zoom_identity['baryon_configuration_sha256']}\n"
        f"model_zoom_sink_initial_conditions_sha256 = {model_zoom_identity['sink_initial_conditions_sha256']}\n"
    )
    if model == "cdm":
        common += "dm_transport = collisionless_nbody\n"
    elif model == "sidm":
        common += (
            "sidm_cross_section_cm2_g = 1.0d0\n"
            "sidm_type = constant\n"
            "sidm_v0_km_s = 100.0d0\n"
            "sidm_power = -4.0d0\n"
            "sidm_angular = isotropic\n"
            "sidm_inelastic = .false.\n"
            "sidm_max_scatter_probability = 1.0d-2\n"
        )
    else:
        common += (
            "m_axion_ev = 1.0d-22\n"
            "fdm_use_hjm = .false.\n"
            "fdm_first_wave_level = 12\n"
            "fdm_outer_ledger_enabled = .true.\n"
            "fdm_force_accounting = resolved_wave_only\n"
        )
    provenance.write_text(common, encoding="utf-8")
    ledgers["run_provenance"] = provenance
    ledgers["raw_inventory"] = inventory_path
    return assessment_path, ledgers


def _physics_input(
    tmp_path: Path,
    *,
    case: GalaxyMergerZoomCase,
    manifest_sha256: str,
    selected_case_id_override: str | None = None,
) -> tuple[Path, dict[str, dict[str, str]]]:
    ensemble = tmp_path / "capture_ensemble.json"
    shared_inputs: dict[str, dict[str, str]] = {}
    for name in ("initial_conditions", "baryon_configuration", "smbh_seed_catalog"):
        path = tmp_path / "shared" / f"{name}.dat"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n", encoding="utf-8")
        shared_inputs[name] = {
            "path": str(path.relative_to(tmp_path)),
            "sha256": _sha256(path),
        }
    artifact_names = {
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
            "wave_provenance",
            "field_snapshot_index",
        ),
    }
    artifacts: dict[str, dict[str, dict[str, str]]] = {}
    hashes: dict[str, dict[str, str]] = {}
    inventory_assessments: dict[str, dict[str, str]] = {}
    provenance_paths: dict[str, Path] = {}
    for model, names in artifact_names.items():
        artifacts[model] = {}
        hashes[model] = {}
        assessment_path, inventory_ledgers = _write_ready_inventory_assessment(
            tmp_path,
            model,
            model_zoom_identity={
                "manifest_sha256": manifest_sha256,
                "case_id": (
                    (
                        selected_case_id_override
                        if selected_case_id_override is not None
                        else case.case_id
                    )
                    if model == case.physics.dark_matter_model
                    else f"{model}-comparison-only"
                ),
                "initial_conditions_sha256": shared_inputs["initial_conditions"]["sha256"],
                "baryon_configuration_sha256": shared_inputs["baryon_configuration"][
                    "sha256"
                ],
                "sink_initial_conditions_sha256": shared_inputs["smbh_seed_catalog"]["sha256"],
            },
        )
        inventory_assessments[model] = {
            "path": str(assessment_path.relative_to(tmp_path)),
            "sha256": _sha256(assessment_path),
        }
        provenance_paths[model] = inventory_ledgers["run_provenance"]
        for name in names:
            if name == "wave_ledger":
                continue
            path = inventory_ledgers.get(name)
            if path is None:
                path = tmp_path / "artifacts" / model / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{model} {name}\n", encoding="utf-8")
            digest = _sha256(path)
            hashes[model][name] = digest
            artifacts[model][name] = {
                "path": str(path.relative_to(tmp_path)),
                "sha256": digest,
            }
        if model == "fdm":
            wave_source = tmp_path / "artifacts" / model / "wave_source_index.json"
            wave_source.write_text("resolved FDM wave source index\n", encoding="utf-8")
            ledger_path = tmp_path / "artifacts" / model / "outer_wave_ledger.json"
            ledger = FDMOuterWaveLedger(
                source_path=wave_source.name,
                source_sha256=_sha256(wave_source),
                force_ledger_sha256=hashes[model]["force_ledger"],
                field_snapshot_index_sha256=hashes[model]["field_snapshot_index"],
                profile_snapshot_index_sha256=hashes[model]["environment_profile"],
                sample_times_myr=[0.0, 0.1],
                core_centres_pc=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
                fdm_current_msun_pc2_myr=[[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
                coherence_time_myr=[0.05, 0.05],
                de_broglie_wavelength_pc=[10.0, 10.0],
                granule_power=[0.1, 0.1],
                dipole_l1_real_imag=[[0.1, 0.0], [0.1, 0.0]],
                quadrupole_l2_real_imag=[[0.1, 0.0], [0.1, 0.0]],
                wake_extent_pc=[20.0, 20.0],
                hjm_wave_seam_clearance_pc=[30.0, 30.0],
                boundary_clearance_pc=[30.0, 30.0],
                maximum_relative_hamiltonian_error=1.0e-5,
                maximum_relative_mass_error=1.0e-5,
                maximum_relative_angular_momentum_error=1.0e-5,
                force_accounting="live_wave_only",
            )
            _write_json(ledger_path, ledger.as_dict())
            hashes[model]["wave_ledger"] = _sha256(ledger_path)
            artifacts[model]["wave_ledger"] = {
                "path": str(ledger_path.relative_to(tmp_path)),
                "sha256": hashes[model]["wave_ledger"],
            }
    manifest_path = tmp_path / "comparison_family.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "family_id": "model-specific-fixture",
            "shared_inputs": shared_inputs,
            "run_provenance": {
                model: str(path.relative_to(tmp_path))
                for model, path in provenance_paths.items()
            },
        },
    )
    registration_path = tmp_path / "capture_registration.json"
    _write_json(
        registration_path,
        {
            "schema_version": 1,
            "family_manifest_path": str(manifest_path.relative_to(tmp_path)),
            "family_manifest_sha256": _sha256(manifest_path),
            "captures": {
                model: {
                    "ledger_path": str(
                        (path.parent / "smbh_capture_ledger_v1.jsonl").relative_to(tmp_path)
                    ),
                    "event_uid": "capture-7-9",
                }
                for model, path in provenance_paths.items()
            },
        },
    )
    family = read_dm_comparison_family_manifest(manifest_path)
    assert preflight_dm_comparison_family(family).ready
    assert assess_dm_comparison_smoke_outputs(
        preflight_dm_comparison_family(family)
    ).verified
    capture_ensemble = register_dm_comparison_capture_ensemble(
        read_dm_comparison_capture_registration(registration_path)
    )
    assert capture_ensemble.registered
    _write_json(ensemble, capture_ensemble.as_dict())
    physics_input = tmp_path / "physics_input.json"
    _write_json(
        physics_input,
        {
            "schema_version": 2,
            "capture_ensemble_path": "capture_ensemble.json",
            "capture_ensemble_sha256": _sha256(ensemble),
            "artifacts": artifacts,
            "normal_output_inventory_assessments": inventory_assessments,
        },
    )
    return physics_input, hashes


def _channels() -> dict[str, dict[str, str]]:
    return {
        name: {
            "status": "available",
            "profile_sha256": "b" * 64,
            "force_ledger_sha256": "c" * 64,
        }
        for name in ("stars", "gas", "dark_matter")
    }


def _result_record(
    *,
    case: GalaxyMergerZoomCase,
    manifest_sha256: str,
    physics_input: Path,
    artifacts: dict[str, str],
    power_factor: float = 1.0,
    force_accounting: str = "live_wave_only",
) -> dict[str, object]:
    model = case.physics.dark_matter_model
    evidence: dict[str, object] = {"artifact_sha256s": artifacts}
    if model == "sidm":
        evidence["maximum_scatter_probability"] = 0.02
    elif model == "fdm":
        evidence.update(
            force_accounting=force_accounting,
            minimum_de_broglie_resolution_cells=8.0,
            minimum_wake_resolution_cells=8.0,
        )
    result: dict[str, object] = {
        "schema_version": 3,
        "status": "complete",
        "case_id": case.case_id,
        "case": case.as_dict(),
        "dark_matter_model": model,
        "zoom_manifest_sha256": manifest_sha256,
        "capture_event_uid": "capture-7-9",
        "physics_input_path": str(physics_input),
        "physics_input_sha256": _sha256(physics_input),
        "environment_channels": _channels(),
        "rate_points": [
            {
                "separation_pc": separation,
                "orbital_power_pc2_myr3": -power_factor * separation,
                "orbital_torque_msun_pc2_myr": -2.0 * power_factor * separation,
                "eccentricity": 0.4,
            }
            for separation in (1.0, 2.0, 4.0)
        ],
        "diagnostics": {
            "maximum_relative_energy_error": 1.0e-5,
            "maximum_relative_angular_momentum_error": 2.0e-5,
            "minimum_orbital_resolution_cells": 8.0,
        },
        "model_evidence": evidence,
    }
    physics_record = json.loads(physics_input.read_text(encoding="utf-8"))
    assessment_artifact = physics_record["normal_output_inventory_assessments"][model]
    assessment_path = physics_input.parent / assessment_artifact["path"]
    assessment_record = json.loads(assessment_path.read_text(encoding="utf-8"))
    inventory_artifact = assessment_record["inventory"]["source"]
    ensemble_path = physics_input.parent / physics_record["capture_ensemble_path"]
    ensemble_record = json.loads(ensemble_path.read_text(encoding="utf-8"))
    result["capture_event_sha256"] = ensemble_record["capture_bindings"][model][
        "capture_event"
    ]["event_sha256"]
    run_artifact = ensemble_record["capture_bindings"][model]["run_provenance"]["source"]

    def absolute_artifact(artifact: dict[str, str]) -> dict[str, str]:
        path = Path(artifact["path"])
        return {
            "path": str((path if path.is_absolute() else physics_input.parent / path).resolve()),
            "sha256": artifact["sha256"],
        }

    rate_ledger_path = physics_input.parent / "rate_ledgers" / (
        f"{model}-r{case.replicate}-l{case.numerics.levelmax}.json"
    )
    rate_ledger = {
        "schema_version": 2,
        "status": "diagnosed",
        "case_id": result["case_id"],
        "case": result["case"],
        "dark_matter_model": result["dark_matter_model"],
        "zoom_manifest_sha256": result["zoom_manifest_sha256"],
        "capture_event_uid": result["capture_event_uid"],
        "capture_event_sha256": result["capture_event_sha256"],
        "normal_output_inventory": absolute_artifact(inventory_artifact),
        "run_provenance": absolute_artifact(run_artifact),
        "environment_channels": result["environment_channels"],
        "rate_points": result["rate_points"],
        "diagnostics": result["diagnostics"],
        "model_evidence": result["model_evidence"],
    }
    _write_json(rate_ledger_path, rate_ledger)
    result["schema_version"] = 3
    result["rate_ledger_path"] = str(rate_ledger_path)
    result["rate_ledger_sha256"] = _sha256(rate_ledger_path)
    return result


@pytest.mark.parametrize("model", ("cdm", "sidm", "fdm"))
def test_each_model_keeps_its_own_required_evidence(tmp_path: Path, model: str) -> None:
    case = _case(model, finest_cell_size_pc=0.5)
    physics_input, hashes = _physics_input(
        tmp_path, case=case, manifest_sha256="a" * 64
    )
    path = tmp_path / f"{model}.json"
    _write_json(
        path,
        _result_record(
            case=case,
            manifest_sha256="a" * 64,
            physics_input=physics_input,
            artifacts=hashes[model],
        ),
    )
    result = read_resolved_model_physics_result(
        path,
        case=case,
        zoom_manifest_sha256="a" * 64,
    )
    assert result.dark_matter_model == model
    assert result.channel_map["stars"].status == "available"
    if model == "fdm":
        assert result.model_evidence.force_accounting == "live_wave_only"
    if model == "sidm":
        assert result.model_evidence.maximum_scatter_probability == pytest.approx(0.02)


def test_fdm_resolution_and_phase_acceptance_never_adds_an_analytic_drag(tmp_path: Path) -> None:
    manifest = "a" * 64
    members = []
    for replicate in (0, 1):
        fine = _case("fdm", finest_cell_size_pc=0.5, replicate=replicate)
        coarse = _case("fdm", finest_cell_size_pc=1.0, replicate=replicate)
        fine_input, fine_hashes = _physics_input(
            tmp_path / f"fine-input-r{replicate}", case=fine, manifest_sha256=manifest
        )
        coarse_input, coarse_hashes = _physics_input(
            tmp_path / f"coarse-input-r{replicate}", case=coarse, manifest_sha256=manifest
        )
        fine_path = tmp_path / f"fine-r{replicate}.json"
        coarse_path = tmp_path / f"coarse-r{replicate}.json"
        _write_json(
            fine_path,
            _result_record(
                case=fine,
                manifest_sha256=manifest,
                physics_input=fine_input,
                artifacts=fine_hashes["fdm"],
            ),
        )
        _write_json(
            coarse_path,
            _result_record(
                case=coarse,
                manifest_sha256=manifest,
                physics_input=coarse_input,
                artifacts=coarse_hashes["fdm"],
                power_factor=1.05,
            ),
        )
        members.append(
            compare_model_specific_resolution_pair(
                read_resolved_model_physics_result(
                    fine_path, case=fine, zoom_manifest_sha256=manifest
                ),
                read_resolved_model_physics_result(
                    coarse_path, case=coarse, zoom_manifest_sha256=manifest
                ),
            )
        )
    assert all(member.accepted for member in members)
    phase = assess_model_specific_phase_ensemble(tuple(members))
    assert phase.ready_for_separate_model_interpretation
    assert phase.as_dict()["interpretation"].find("coalescence") >= 0


def test_rejects_fdm_analytic_force_or_input_hash_mismatch(tmp_path: Path) -> None:
    case = _case("fdm", finest_cell_size_pc=0.5)
    physics_input, hashes = _physics_input(
        tmp_path, case=case, manifest_sha256="a" * 64
    )
    record = _result_record(
        case=case,
        manifest_sha256="a" * 64,
        physics_input=physics_input,
        artifacts=hashes["fdm"],
        force_accounting="analytic_drag",
    )
    path = tmp_path / "invalid-force.json"
    _write_json(path, record)
    with pytest.raises(ValueError, match="resolved-wave force accounting"):
        read_resolved_model_physics_result(path, case=case, zoom_manifest_sha256="a" * 64)

    record = _result_record(
        case=case,
        manifest_sha256="a" * 64,
        physics_input=physics_input,
        artifacts=hashes["fdm"],
    )
    record["model_evidence"]["artifact_sha256s"]["wave_ledger"] = "0" * 64
    _write_json(path, record)
    with pytest.raises(ValueError, match="differs from accepted physics input"):
        read_resolved_model_physics_result(path, case=case, zoom_manifest_sha256="a" * 64)


def test_result_must_reuse_its_registered_capture_event(tmp_path: Path) -> None:
    case = _case("cdm", finest_cell_size_pc=0.5)
    physics_input, hashes = _physics_input(
        tmp_path, case=case, manifest_sha256="a" * 64
    )
    record = _result_record(
        case=case,
        manifest_sha256="a" * 64,
        physics_input=physics_input,
        artifacts=hashes["cdm"],
    )
    record["capture_event_uid"] = "another-capture"
    path = tmp_path / "wrong-capture.json"
    _write_json(path, record)
    with pytest.raises(ValueError, match="differs from its registered capture ensemble"):
        read_resolved_model_physics_result(path, case=case, zoom_manifest_sha256="a" * 64)

    record = _result_record(
        case=case,
        manifest_sha256="a" * 64,
        physics_input=physics_input,
        artifacts=hashes["cdm"],
    )
    record["capture_event_sha256"] = "f" * 64
    path = tmp_path / "wrong-capture-sha.json"
    _write_json(path, record)
    with pytest.raises(ValueError, match="capture_event_sha256 differs"):
        read_resolved_model_physics_result(path, case=case, zoom_manifest_sha256="a" * 64)


def test_result_requires_the_baryon_configuration_attested_by_its_family(
    tmp_path: Path,
) -> None:
    case = _case("fdm", finest_cell_size_pc=0.5)
    physics_input, hashes = _physics_input(
        tmp_path, case=case, manifest_sha256="a" * 64
    )
    input_record = json.loads(physics_input.read_text(encoding="utf-8"))
    ensemble_path = physics_input.parent / input_record["capture_ensemble_path"]
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    provenance_path = Path(ensemble["capture_bindings"]["fdm"]["run_provenance"]["source"]["path"])
    provenance_path.write_text(
        provenance_path.read_text(encoding="utf-8").replace(
            "model_zoom_baryon_configuration_sha256 = "
            + _sha256(tmp_path / "shared" / "baryon_configuration.dat"),
            "model_zoom_baryon_configuration_sha256 = " + "f" * 64,
        ),
        encoding="utf-8",
    )
    refreshed_ensemble = register_dm_comparison_capture_ensemble(
        read_dm_comparison_capture_registration(tmp_path / "capture_registration.json")
    )
    assert refreshed_ensemble.registered
    _write_json(ensemble_path, refreshed_ensemble.as_dict())
    input_record["capture_ensemble_sha256"] = _sha256(ensemble_path)
    _write_json(physics_input, input_record)
    record = _result_record(
        case=case,
        manifest_sha256="a" * 64,
        physics_input=physics_input,
        artifacts=hashes["fdm"],
    )
    path = tmp_path / "wrong-baryon-attestation.json"
    _write_json(path, record)
    with pytest.raises(ValueError, match="model_zoom_baryon_configuration_sha256 differs"):
        read_resolved_model_physics_result(path, case=case, zoom_manifest_sha256="a" * 64)


def test_resolution_pair_requires_the_same_capture_event_sha256(tmp_path: Path) -> None:
    manifest = "a" * 64
    fine = _case("fdm", finest_cell_size_pc=0.5)
    coarse = _case("fdm", finest_cell_size_pc=1.0)
    fine_input, fine_hashes = _physics_input(
        tmp_path / "fine", case=fine, manifest_sha256=manifest
    )
    coarse_input, coarse_hashes = _physics_input(
        tmp_path / "coarse", case=coarse, manifest_sha256=manifest
    )
    fine_path = tmp_path / "fine.json"
    coarse_path = tmp_path / "coarse.json"
    _write_json(
        fine_path,
        _result_record(
            case=fine,
            manifest_sha256=manifest,
            physics_input=fine_input,
            artifacts=fine_hashes["fdm"],
        ),
    )
    _write_json(
        coarse_path,
        _result_record(
            case=coarse,
            manifest_sha256=manifest,
            physics_input=coarse_input,
            artifacts=coarse_hashes["fdm"],
        ),
    )
    fine_result = read_resolved_model_physics_result(
        fine_path, case=fine, zoom_manifest_sha256=manifest
    )
    coarse_result = read_resolved_model_physics_result(
        coarse_path, case=coarse, zoom_manifest_sha256=manifest
    )
    with pytest.raises(ValueError, match="does not share one capture"):
        compare_model_specific_resolution_pair(
            fine_result,
            replace(coarse_result, capture_event_sha256="f" * 64),
        )


def test_capture_ensemble_is_rebuilt_from_its_family_and_capture_event(tmp_path: Path) -> None:
    case = _case("fdm", finest_cell_size_pc=0.5)
    physics_input, _ = _physics_input(tmp_path, case=case, manifest_sha256="a" * 64)
    input_record = json.loads(physics_input.read_text(encoding="utf-8"))
    ensemble_path = physics_input.parent / input_record["capture_ensemble_path"]
    assert read_verified_dm_comparison_capture_ensemble(ensemble_path).registered
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    ensemble["capture_bindings"]["fdm"]["capture_event"]["event_sha256"] = "0" * 64
    _write_json(ensemble_path, ensemble)
    with pytest.raises(ValueError, match="capture event differs"):
        read_verified_dm_comparison_capture_ensemble(ensemble_path)


def test_result_rate_values_and_output_identity_must_match_the_rate_ledger(
    tmp_path: Path,
) -> None:
    case = _case("cdm", finest_cell_size_pc=0.5)
    physics_input, hashes = _physics_input(
        tmp_path, case=case, manifest_sha256="a" * 64
    )
    record = _result_record(
        case=case,
        manifest_sha256="a" * 64,
        physics_input=physics_input,
        artifacts=hashes["cdm"],
    )
    path = tmp_path / "rate-tampered.json"
    record["rate_points"][0]["eccentricity"] = 0.1
    _write_json(path, record)
    with pytest.raises(ValueError, match="rate ledger rate_points differs"):
        read_resolved_model_physics_result(path, case=case, zoom_manifest_sha256="a" * 64)

    record = _result_record(
        case=case,
        manifest_sha256="a" * 64,
        physics_input=physics_input,
        artifacts=hashes["cdm"],
    )
    ledger_path = Path(record["rate_ledger_path"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["capture_event_uid"] = "other-capture"
    _write_json(ledger_path, ledger)
    record["rate_ledger_sha256"] = _sha256(ledger_path)
    _write_json(path, record)
    with pytest.raises(ValueError, match="rate ledger provenance differs"):
        read_resolved_model_physics_result(path, case=case, zoom_manifest_sha256="a" * 64)


def test_result_rejects_a_normal_output_attested_for_another_case(tmp_path: Path) -> None:
    case = _case("fdm", finest_cell_size_pc=0.5)
    physics_input, hashes = _physics_input(
        tmp_path,
        case=case,
        manifest_sha256="a" * 64,
        selected_case_id_override="fdm-other-resolution-replicate",
    )
    record = _result_record(
        case=case,
        manifest_sha256="a" * 64,
        physics_input=physics_input,
        artifacts=hashes["fdm"],
    )
    path = tmp_path / "wrong-executed-case.json"
    _write_json(path, record)
    with pytest.raises(ValueError, match="model zoom case differs"):
        read_resolved_model_physics_result(path, case=case, zoom_manifest_sha256="a" * 64)


def test_fdm_full_wave_ledger_is_distinct_from_raw_provenance(tmp_path: Path) -> None:
    case = _case("fdm", finest_cell_size_pc=0.5)
    physics_input, _ = _physics_input(tmp_path, case=case, manifest_sha256="a" * 64)
    record = json.loads(physics_input.read_text(encoding="utf-8"))
    ledger_path = tmp_path / record["artifacts"]["fdm"]["wave_ledger"]["path"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["force_accounting"] = "resolved_wake_plus_measured_residual"
    _write_json(ledger_path, ledger)
    record["artifacts"]["fdm"]["wave_ledger"]["sha256"] = _sha256(ledger_path)
    _write_json(physics_input, record)
    assessment = assess_dm_comparison_physics_inputs(
        read_dm_comparison_physics_input(physics_input)
    )
    assert not assessment.ready_for_model_specific_analysis
    assert any("live_wave_only until a residual ledger is attested" in reason for reason in assessment.reasons)

    record["artifacts"]["fdm"]["wave_provenance"]["sha256"] = "0" * 64
    _write_json(physics_input, record)
    assessment = assess_dm_comparison_physics_inputs(
        read_dm_comparison_physics_input(physics_input)
    )
    assert any("fdm wave_provenance" in reason for reason in assessment.reasons)


def test_sidm_controls_are_required_and_cdm_rejects_them() -> None:
    sidm = _physics("sidm")
    assert sidm.sidm_cross_section_cm2_g == pytest.approx(1.0)
    with pytest.raises(ValueError, match="CDM zooms cannot carry"):
        ZoomPhysicsPoint(
            **{
                **_physics("cdm").__dict__,
                "sidm_cross_section_cm2_g": 1.0,
                "sidm_v0_km_s": 100.0,
                "sidm_power": -4.0,
                "sidm_type": "constant",
                "sidm_angular": "isotropic",
                "sidm_inelastic": False,
            }
        )


def test_non_submitting_cli_validates_resolution_and_phase_records(tmp_path: Path) -> None:
    specification = {
        "schema_version": 1,
        "replicates": 2,
        "baseline": asdict(_physics("fdm")),
        "variations": [],
        "numerics": [
            {
                "levelmax": 19,
                "finest_cell_size_pc": 1.0,
                "collisionless_particle_mass_msun": 1.0e3,
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
    config = tmp_path / "grid.yaml"
    config.write_text(yaml.safe_dump(specification), encoding="utf-8")
    grid = build_zoom_grid(specification)
    pairs: list[tuple[Path, Path]] = []
    for replicate in (0, 1):
        cases = [case for case in grid.cases if case.replicate == replicate]
        coarse, fine = sorted(cases, key=lambda case: case.numerics.finest_cell_size_pc, reverse=True)
        fine_input, fine_hashes = _physics_input(
            tmp_path / f"cli-fine-input-r{replicate}",
            case=fine,
            manifest_sha256=grid.manifest_sha256,
        )
        coarse_input, coarse_hashes = _physics_input(
            tmp_path / f"cli-coarse-input-r{replicate}",
            case=coarse,
            manifest_sha256=grid.manifest_sha256,
        )
        fine_path = tmp_path / f"cli-fine-r{replicate}.json"
        coarse_path = tmp_path / f"cli-coarse-r{replicate}.json"
        _write_json(
            fine_path,
            _result_record(
                case=fine,
                manifest_sha256=grid.manifest_sha256,
                physics_input=fine_input,
                artifacts=fine_hashes["fdm"],
            ),
        )
        _write_json(
            coarse_path,
            _result_record(
                case=coarse,
                manifest_sha256=grid.manifest_sha256,
                physics_input=coarse_input,
                artifacts=coarse_hashes["fdm"],
                power_factor=1.05,
            ),
        )
        pairs.append((fine_path, coarse_path))
    checked = tmp_path / "checked.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_resolved_model_physics_run.py",
            str(config),
            str(pairs[0][0]),
            str(checked),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(checked.read_text(encoding="utf-8"))["status"] == "resolved_model_physics_ready"
    resolution = tmp_path / "resolution.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_model_specific_resolution.py",
            str(config),
            str(pairs[0][0]),
            str(pairs[0][1]),
            str(resolution),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(resolution.read_text(encoding="utf-8"))["status"] == "accepted_model_specific_rates"
    phase = tmp_path / "phase.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/assess_model_specific_phase_ensemble.py",
            str(config),
            "--pair",
            str(pairs[0][0]),
            str(pairs[0][1]),
            "--pair",
            str(pairs[1][0]),
            str(pairs[1][1]),
            str(phase),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(phase.read_text(encoding="utf-8"))["status"] == "accepted_model_specific_phase_ensemble"
