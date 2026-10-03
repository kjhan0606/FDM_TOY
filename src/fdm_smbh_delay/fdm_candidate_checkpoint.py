"""Disk restart contract for the numerical outer-FDM candidate orbit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from .fdm_outer_halo import FDMOuterHaloClosure
from .fdm_outer_response_family import FDMOuterResponseFamily
from .kpc_checkpoint import _digest, _physics_value, kpc_implementation_sha256
from .kpc_inspiral import DualNucleusState, KpcInspiralModel, KpcIntegrationConfig


_CANDIDATE_MODULES = (
    "fdm_candidate_checkpoint.py", "fdm_candidate_orbit.py",
    "fdm_orbital_response.py", "fdm_outer_response.py",
    "fdm_outer_response_family.py", "fdm_response_kick.py", "host_orbit.py",
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


def _state_record(state: DualNucleusState) -> dict[str, object]:
    return {
        "elapsed_myr": state.elapsed_myr,
        "position_pc": state.position_pc.tolist(),
        "velocity_pc_myr": state.velocity_pc_myr.tolist(),
        "envelope_truncation_radius_pc": state.envelope_truncation_radius_pc,
        "completed_steps": state.completed_steps,
    }


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

    try:
        record = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read candidate checkpoint: {error}") from error
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
