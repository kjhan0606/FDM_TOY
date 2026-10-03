"""Disk restart contract for the numerical outer-FDM candidate orbit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import numpy as np

from .fdm_outer_halo import FDMOuterHaloClosure
from .fdm_outer_response_family import FDMOuterResponseFamily
from .fdm_response_sources import VerifiedFDMResponseSourceBundle
from .kpc_checkpoint import (
    _digest, _physics_value, _state_from_record as _phase_state_from_record,
    _state_record as _phase_state_record, kpc_implementation_sha256,
)
from .kpc_inspiral import (
    DualNucleusState, KpcInspiralModel, KpcIntegrationConfig,
    KpcToHardConfig, KpcToHardState,
)


_CANDIDATE_MODULES = (
    "fdm_candidate_checkpoint.py", "fdm_candidate_orbit.py",
    "fdm_candidate_phase.py",
    "fdm_orbital_response.py", "fdm_outer_response.py",
    "fdm_outer_response_family.py", "fdm_response_kick.py",
    "fdm_response_sources.py", "host_orbit.py",
)


def candidate_implementation_sha256() -> str:
    directory = Path(__file__).resolve().parent
    return _digest({
        "base_kpc_implementation_sha256": kpc_implementation_sha256(),
        "candidate_sources": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in _CANDIDATE_MODULES
        },
    })


def candidate_physics_sha256(
    *,
    model: KpcInspiralModel,
    config: KpcIntegrationConfig,
    response_family: FDMOuterResponseFamily,
    closure_by_source_sha256: Mapping[str, FDMOuterHaloClosure],
    radial_support_pc: tuple[float, float],
    random_seed: int,
) -> str:
    if not isinstance(closure_by_source_sha256, Mapping):
        raise ValueError("candidate source closure mapping is required")
    if any(
        not isinstance(key, str) or not isinstance(value, FDMOuterHaloClosure)
        for key, value in closure_by_source_sha256.items()
    ):
        raise ValueError("candidate source closure mapping is invalid")
    return _digest({
        "model": _physics_value(model),
        "config": _physics_value(config),
        "response_family": _physics_value(response_family),
        "closures": {
            key: _physics_value(value)
            for key, value in sorted(closure_by_source_sha256.items())
        },
        "radial_support_pc": _physics_value(radial_support_pc),
        "random_seed": _physics_value(random_seed),
    })


def candidate_phase_physics_sha256(
    *,
    model: KpcInspiralModel,
    outer_config: KpcIntegrationConfig,
    phase_config: KpcToHardConfig,
    response_family: FDMOuterResponseFamily,
    closure_by_source_sha256: Mapping[str, FDMOuterHaloClosure],
    radial_support_pc: tuple[float, float],
    random_seed: int,
    source_manifest_sha256: str | None = None,
) -> str:
    if source_manifest_sha256 is not None and (
        not isinstance(source_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256) is None
    ):
        raise ValueError("candidate source manifest SHA-256 is invalid")
    return _digest({
        "outer_physics_sha256": candidate_physics_sha256(
            model=model, config=outer_config, response_family=response_family,
            closure_by_source_sha256=closure_by_source_sha256,
            radial_support_pc=radial_support_pc, random_seed=random_seed,
        ),
        "phase_config": _physics_value(phase_config),
        "source_manifest_sha256": source_manifest_sha256,
    })


def _state_record(state: DualNucleusState) -> dict[str, object]:
    return {
        "elapsed_myr": state.elapsed_myr,
        "position_pc": state.position_pc.tolist(),
        "velocity_pc_myr": state.velocity_pc_myr.tolist(),
        "envelope_truncation_radius_pc": state.envelope_truncation_radius_pc,
        "completed_steps": state.completed_steps,
    }


def _atomic_json_replace(destination: Path, record: dict[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".partial", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_json(path: str | Path) -> object:
    try:
        return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read candidate checkpoint: {error}") from error


def write_candidate_fdm_checkpoint(
    path: str | Path,
    state: DualNucleusState,
    *,
    model: KpcInspiralModel,
    config: KpcIntegrationConfig,
    response_family: FDMOuterResponseFamily,
    closure_by_source_sha256: Mapping[str, FDMOuterHaloClosure],
    radial_support_pc: tuple[float, float],
    random_seed: int,
) -> None:
    """Atomically replace one explicitly named candidate checkpoint."""

    destination = Path(path).expanduser().resolve()
    if not destination.parent.is_dir():
        raise ValueError("candidate checkpoint parent directory must exist")
    if state.elapsed_myr > config.maximum_time_myr or state.completed_steps > config.maximum_steps:
        raise ValueError("candidate checkpoint state exceeds integration budget")
    state_record = _state_record(state)
    record = {
        "schema_version": 1,
        "physics_sha256": candidate_physics_sha256(
            model=model, config=config, response_family=response_family,
            closure_by_source_sha256=closure_by_source_sha256,
            radial_support_pc=radial_support_pc, random_seed=random_seed,
        ),
        "implementation_sha256": candidate_implementation_sha256(),
        "state_sha256": _digest(state_record),
        "state": state_record,
    }
    _atomic_json_replace(destination, record)


def read_candidate_fdm_checkpoint(
    path: str | Path,
    *,
    model: KpcInspiralModel,
    config: KpcIntegrationConfig,
    response_family: FDMOuterResponseFamily,
    closure_by_source_sha256: Mapping[str, FDMOuterHaloClosure],
    radial_support_pc: tuple[float, float],
    random_seed: int,
) -> DualNucleusState:
    """Reject changed physics/code, corrupted state, or out-of-budget restart."""

    record = _read_json(path)
    if not isinstance(record, dict) or set(record) != {
        "schema_version", "physics_sha256", "implementation_sha256",
        "state_sha256", "state",
    } or record["schema_version"] != 1:
        raise ValueError("unsupported candidate checkpoint schema")
    expected_physics = candidate_physics_sha256(
        model=model, config=config, response_family=response_family,
        closure_by_source_sha256=closure_by_source_sha256,
        radial_support_pc=radial_support_pc, random_seed=random_seed,
    )
    if record["physics_sha256"] != expected_physics:
        raise ValueError("candidate checkpoint physics fingerprint differs")
    if record["implementation_sha256"] != candidate_implementation_sha256():
        raise ValueError("candidate checkpoint implementation fingerprint differs")
    raw = record["state"]
    if not isinstance(raw, dict) or set(raw) != {
        "elapsed_myr", "position_pc", "velocity_pc_myr",
        "envelope_truncation_radius_pc", "completed_steps",
    }:
        raise ValueError("candidate checkpoint state record is invalid")
    if record["state_sha256"] != _digest(raw):
        raise ValueError("candidate checkpoint state checksum differs")
    if type(raw["completed_steps"]) is not int:
        raise ValueError("candidate checkpoint completed_steps must be an integer")
    state = DualNucleusState(
        elapsed_myr=raw["elapsed_myr"],
        position_pc=np.asarray(raw["position_pc"], dtype=float),
        velocity_pc_myr=np.asarray(raw["velocity_pc_myr"], dtype=float),
        envelope_truncation_radius_pc=raw["envelope_truncation_radius_pc"],
        completed_steps=raw["completed_steps"],
    )
    if state.elapsed_myr > config.maximum_time_myr or state.completed_steps > config.maximum_steps:
        raise ValueError("candidate checkpoint state exceeds integration budget")
    return state


def write_candidate_fdm_phase_checkpoint(
    path: str | Path,
    state: KpcToHardState,
    *,
    model: KpcInspiralModel,
    outer_config: KpcIntegrationConfig,
    phase_config: KpcToHardConfig,
    response_family: FDMOuterResponseFamily,
    closure_by_source_sha256: Mapping[str, FDMOuterHaloClosure],
    radial_support_pc: tuple[float, float],
    random_seed: int,
    source_manifest_sha256: str | None = None,
) -> None:
    """Atomically preserve trajectory state and full phase-transition history."""

    destination = Path(path).expanduser().resolve()
    if not destination.parent.is_dir():
        raise ValueError("candidate phase checkpoint parent directory must exist")
    dynamics = state.dynamical_state
    if dynamics.elapsed_myr > outer_config.maximum_time_myr or dynamics.completed_steps > outer_config.maximum_steps:
        raise ValueError("candidate phase checkpoint exceeds outer integration budget")
    record_state = _phase_state_record(state)
    record = {
        "schema_version": 1,
        "kind": "candidate_fdm_phase",
        "physics_sha256": candidate_phase_physics_sha256(
            model=model, outer_config=outer_config, phase_config=phase_config,
            response_family=response_family,
            closure_by_source_sha256=closure_by_source_sha256,
            radial_support_pc=radial_support_pc, random_seed=random_seed,
            source_manifest_sha256=source_manifest_sha256,
        ),
        "implementation_sha256": candidate_implementation_sha256(),
        "state_sha256": _digest(record_state),
        "state": record_state,
    }
    _atomic_json_replace(destination, record)


def read_candidate_fdm_phase_checkpoint(
    path: str | Path,
    *,
    model: KpcInspiralModel,
    outer_config: KpcIntegrationConfig,
    phase_config: KpcToHardConfig,
    response_family: FDMOuterResponseFamily,
    closure_by_source_sha256: Mapping[str, FDMOuterHaloClosure],
    radial_support_pc: tuple[float, float],
    random_seed: int,
    source_manifest_sha256: str | None = None,
) -> KpcToHardState:
    """Reject changed response/phase physics, implementation, or history."""

    record = _read_json(path)
    if not isinstance(record, dict) or set(record) != {
        "schema_version", "kind", "physics_sha256", "implementation_sha256",
        "state_sha256", "state",
    } or record["schema_version"] != 1 or record["kind"] != "candidate_fdm_phase":
        raise ValueError("unsupported candidate phase checkpoint schema")
    expected_physics = candidate_phase_physics_sha256(
        model=model, outer_config=outer_config, phase_config=phase_config,
        response_family=response_family,
        closure_by_source_sha256=closure_by_source_sha256,
        radial_support_pc=radial_support_pc, random_seed=random_seed,
        source_manifest_sha256=source_manifest_sha256,
    )
    if record["physics_sha256"] != expected_physics:
        raise ValueError("candidate phase checkpoint physics fingerprint differs")
    if record["implementation_sha256"] != candidate_implementation_sha256():
        raise ValueError("candidate phase checkpoint implementation fingerprint differs")
    if record["state_sha256"] != _digest(record["state"]):
        raise ValueError("candidate phase checkpoint state checksum differs")
    state = _phase_state_from_record(record["state"])
    dynamics = state.dynamical_state
    if dynamics.elapsed_myr > outer_config.maximum_time_myr or dynamics.completed_steps > outer_config.maximum_steps:
        raise ValueError("candidate phase checkpoint exceeds outer integration budget")
    return state


def write_verified_candidate_fdm_phase_checkpoint(
    path: str | Path,
    state: KpcToHardState,
    *,
    source_bundle: VerifiedFDMResponseSourceBundle,
    model: KpcInspiralModel,
    outer_config: KpcIntegrationConfig,
    phase_config: KpcToHardConfig,
    radial_support_pc: tuple[float, float],
    random_seed: int,
) -> None:
    """Bind a phase checkpoint to an unchanged source manifest and nodes."""

    source_bundle.verify_current_files()
    write_candidate_fdm_phase_checkpoint(
        path, state, model=model, outer_config=outer_config,
        phase_config=phase_config,
        response_family=source_bundle.response_family,
        closure_by_source_sha256=source_bundle.closure_by_source_sha256,
        radial_support_pc=radial_support_pc, random_seed=random_seed,
        source_manifest_sha256=source_bundle.manifest_sha256,
    )
    source_bundle.verify_current_files()


def read_verified_candidate_fdm_phase_checkpoint(
    path: str | Path,
    *,
    source_bundle: VerifiedFDMResponseSourceBundle,
    model: KpcInspiralModel,
    outer_config: KpcIntegrationConfig,
    phase_config: KpcToHardConfig,
    radial_support_pc: tuple[float, float],
    random_seed: int,
) -> KpcToHardState:
    """Re-check current source bytes and the manifest-bound phase state."""

    source_bundle.verify_current_files()
    state = read_candidate_fdm_phase_checkpoint(
        path, model=model, outer_config=outer_config,
        phase_config=phase_config,
        response_family=source_bundle.response_family,
        closure_by_source_sha256=source_bundle.closure_by_source_sha256,
        radial_support_pc=radial_support_pc, random_seed=random_seed,
        source_manifest_sha256=source_bundle.manifest_sha256,
    )
    source_bundle.verify_current_files()
    return state
