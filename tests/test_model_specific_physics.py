from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from fdm_smbh_delay.model_specific_physics import (
    assess_model_specific_phase_ensemble,
    compare_model_specific_resolution_pair,
    read_resolved_model_physics_result,
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


def _physics_input(tmp_path: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    ensemble = tmp_path / "capture_ensemble.json"
    _write_json(
        ensemble,
        {
            "schema_version": 1,
            "status": "dm_comparison_capture_ensemble_registered",
        },
    )
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
            "field_snapshot_index",
        ),
    }
    artifacts: dict[str, dict[str, dict[str, str]]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for model, names in artifact_names.items():
        artifacts[model] = {}
        hashes[model] = {}
        for name in names:
            path = tmp_path / "artifacts" / model / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{model} {name}\n", encoding="utf-8")
            digest = _sha256(path)
            hashes[model][name] = digest
            artifacts[model][name] = {
                "path": str(path.relative_to(tmp_path)),
                "sha256": digest,
            }
    physics_input = tmp_path / "physics_input.json"
    _write_json(
        physics_input,
        {
            "schema_version": 1,
            "capture_ensemble_path": "capture_ensemble.json",
            "capture_ensemble_sha256": _sha256(ensemble),
            "artifacts": artifacts,
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
    return {
        "schema_version": 1,
        "status": "complete",
        "case_id": case.case_id,
        "case": case.as_dict(),
        "dark_matter_model": model,
        "zoom_manifest_sha256": manifest_sha256,
        "capture_event_uid": "capture-7-9",
        "physics_input_path": str(physics_input.name),
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


@pytest.mark.parametrize("model", ("cdm", "sidm", "fdm"))
def test_each_model_keeps_its_own_required_evidence(tmp_path: Path, model: str) -> None:
    physics_input, hashes = _physics_input(tmp_path)
    case = _case(model, finest_cell_size_pc=0.5)
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
    physics_input, hashes = _physics_input(tmp_path)
    manifest = "a" * 64
    members = []
    for replicate in (0, 1):
        fine = _case("fdm", finest_cell_size_pc=0.5, replicate=replicate)
        coarse = _case("fdm", finest_cell_size_pc=1.0, replicate=replicate)
        fine_path = tmp_path / f"fine-r{replicate}.json"
        coarse_path = tmp_path / f"coarse-r{replicate}.json"
        _write_json(
            fine_path,
            _result_record(
                case=fine,
                manifest_sha256=manifest,
                physics_input=physics_input,
                artifacts=hashes["fdm"],
            ),
        )
        _write_json(
            coarse_path,
            _result_record(
                case=coarse,
                manifest_sha256=manifest,
                physics_input=physics_input,
                artifacts=hashes["fdm"],
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
    physics_input, hashes = _physics_input(tmp_path)
    case = _case("fdm", finest_cell_size_pc=0.5)
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
    physics_input, hashes = _physics_input(tmp_path)
    pairs: list[tuple[Path, Path]] = []
    for replicate in (0, 1):
        cases = [case for case in grid.cases if case.replicate == replicate]
        coarse, fine = sorted(cases, key=lambda case: case.numerics.finest_cell_size_pc, reverse=True)
        fine_path = tmp_path / f"cli-fine-r{replicate}.json"
        coarse_path = tmp_path / f"cli-coarse-r{replicate}.json"
        _write_json(
            fine_path,
            _result_record(
                case=fine,
                manifest_sha256=grid.manifest_sha256,
                physics_input=physics_input,
                artifacts=hashes["fdm"],
            ),
        )
        _write_json(
            coarse_path,
            _result_record(
                case=coarse,
                manifest_sha256=grid.manifest_sha256,
                physics_input=physics_input,
                artifacts=hashes["fdm"],
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
