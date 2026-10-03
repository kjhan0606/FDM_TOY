"""Disk restart contract for the phase-aware static-host kpc integrator."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import astropy
import numpy as np
import scipy

from .kpc_inspiral import (
    DualNucleusState, KpcInspiralModel, KpcToHardConfig, KpcToHardState,
)
from .kpc_to_pc import InspiralPhase, InspiralState


CHECKPOINT_SCHEMA_VERSION = 2

# Explicit physics dependency list, not a recursive package scan. A changed
# integrator, force law, profile interpolation, handoff definition, or unit
# conversion invalidates bitwise restart compatibility.
_PHYSICS_MODULES = (
    "binary_evolution.py", "capture_ledger.py", "constants.py", "delay_budget.py",
    "environmental_friction.py", "fdm_outer_halo.py", "galaxy_environment.py",
    "gw.py", "kpc_checkpoint.py", "kpc_inspiral.py", "kpc_to_pc.py",
    "lagramses.py",
    "orbital_exchange.py", "profile_table.py", "soliton.py", "wave_drag.py",
)


def _canonical(record: Any) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(record: Any) -> str:
    return hashlib.sha256(_canonical(record)).hexdigest()


def _physics_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite kpc physics parameter")
        return number
    if isinstance(value, np.ndarray):
        if np.any(~np.isfinite(value)):
            raise ValueError("non-finite kpc physics array")
        return {"array_shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, (tuple, list)):
        return [_physics_value(item) for item in value]
    if is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _physics_value(getattr(value, field.name))
                for field in fields(value) if field.init
            },
        }
    raise ValueError(f"unsupported kpc physics type: {type(value).__name__}")


def kpc_physics_sha256(model: KpcInspiralModel, config: KpcToHardConfig) -> str:
    """Bind every declared force/background parameter and integration control."""

    return _digest({"model": _physics_value(model), "config": _physics_value(config)})


def kpc_implementation_sha256() -> str:
    """Hash the bounded local physics source set and numerical dependency versions."""

    module_directory = Path(__file__).resolve().parent
    sources = {
        name: hashlib.sha256((module_directory / name).read_bytes()).hexdigest()
        for name in _PHYSICS_MODULES
    }
    return _digest({
        "sources": sources,
        "python": list(sys.version_info[:3]),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "astropy": astropy.__version__,
    })


def _phase_record(state: InspiralState) -> dict[str, Any]:
    return {
        "event_uid": state.event_uid,
        "phase": state.phase.value,
        "elapsed_myr": state.elapsed_myr,
        "separation_pc": state.separation_pc,
        "semimajor_axis_pc": state.semimajor_axis_pc,
        "eccentricity": state.eccentricity,
        "reason": state.reason,
    }


def _state_record(state: KpcToHardState) -> dict[str, Any]:
    dynamics = state.dynamical_state
    return {
        "dynamics": {
            "elapsed_myr": dynamics.elapsed_myr,
            "position_pc": dynamics.position_pc.tolist(),
            "velocity_pc_myr": dynamics.velocity_pc_myr.tolist(),
            "envelope_truncation_radius_pc": dynamics.envelope_truncation_radius_pc,
            "completed_steps": dynamics.completed_steps,
        },
        "transition_history": [_phase_record(phase) for phase in state.transition_history],
    }


def _phase_from_record(record: Any) -> InspiralState:
    if not isinstance(record, dict) or set(record) != {
        "event_uid", "phase", "elapsed_myr", "separation_pc",
        "semimajor_axis_pc", "eccentricity", "reason",
    }:
        raise ValueError("invalid kpc checkpoint phase record")
    if not isinstance(record["event_uid"], str) or not isinstance(record["reason"], str):
        raise ValueError("invalid kpc checkpoint phase text")
    return InspiralState(
        event_uid=record["event_uid"],
        phase=InspiralPhase(record["phase"]),
        elapsed_myr=record["elapsed_myr"],
        separation_pc=record["separation_pc"],
        semimajor_axis_pc=record["semimajor_axis_pc"],
        eccentricity=record["eccentricity"],
        reason=record["reason"],
    )


def _state_from_record(record: Any) -> KpcToHardState:
    if not isinstance(record, dict) or set(record) != {"dynamics", "transition_history"}:
        raise ValueError("invalid kpc checkpoint state record")
    dynamics = record["dynamics"]
    if not isinstance(dynamics, dict) or set(dynamics) != {
        "elapsed_myr", "position_pc", "velocity_pc_myr",
        "envelope_truncation_radius_pc", "completed_steps",
    }:
        raise ValueError("invalid kpc checkpoint dynamics record")
    if isinstance(dynamics["completed_steps"], bool) or not isinstance(dynamics["completed_steps"], int):
        raise ValueError("invalid kpc checkpoint step count")
    history = record["transition_history"]
    if not isinstance(history, list) or not history:
        raise ValueError("invalid kpc checkpoint phase history")
    phases = tuple(_phase_from_record(phase) for phase in history)
    return KpcToHardState(
        dynamical_state=DualNucleusState(
            elapsed_myr=dynamics["elapsed_myr"],
            position_pc=np.asarray(dynamics["position_pc"], dtype=float),
            velocity_pc_myr=np.asarray(dynamics["velocity_pc_myr"], dtype=float),
            envelope_truncation_radius_pc=dynamics["envelope_truncation_radius_pc"],
            completed_steps=dynamics["completed_steps"],
        ),
        inspiral_state=phases[-1],
        transition_history=phases,
    )


def write_kpc_to_hard_checkpoint(
    path: str | Path,
    state: KpcToHardState,
    model: KpcInspiralModel,
    config: KpcToHardConfig,
) -> None:
    """Atomically replace one explicitly named checkpoint after each accepted step."""

    destination = Path(path).expanduser().resolve()
    if not destination.parent.is_dir():
        raise ValueError("kpc checkpoint parent directory must exist")
    state_record = _state_record(state)
    record = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "physics_sha256": kpc_physics_sha256(model, config),
        "implementation_sha256": kpc_implementation_sha256(),
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


def read_kpc_to_hard_checkpoint(
    path: str | Path,
    model: KpcInspiralModel,
    config: KpcToHardConfig,
) -> KpcToHardState:
    """Reject changed physics, corrupted state, or an out-of-budget restart."""

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read kpc checkpoint: {error}") from error
    if not isinstance(record, dict) or set(record) != {
        "schema_version", "physics_sha256", "implementation_sha256",
        "state_sha256", "state",
    } or record["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported kpc checkpoint schema")
    if record["physics_sha256"] != kpc_physics_sha256(model, config):
        raise ValueError("kpc checkpoint physics/configuration identity changed")
    if record["implementation_sha256"] != kpc_implementation_sha256():
        raise ValueError("kpc checkpoint physics implementation identity changed")
    if record["state_sha256"] != _digest(record["state"]):
        raise ValueError("kpc checkpoint state digest disagrees")
    state = _state_from_record(record["state"])
    if state.dynamical_state.elapsed_myr > config.maximum_time_myr:
        raise ValueError("kpc checkpoint exceeds maximum integration time")
    if state.dynamical_state.completed_steps > config.maximum_steps:
        raise ValueError("kpc checkpoint exceeds maximum step count")
    return state
