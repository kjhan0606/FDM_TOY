"""Materialize a common, non-submitting model-zoom identity fragment.

This contract is shared by CDM, SIDM, and FDM zooms.  It attaches one exact
manifest case, one complete binary capture transaction, and the three shared
input files to the lagRamses normal-output sidecar without modifying a run
namelist or launching a calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .capture_ledger import CaptureEvent, read_capture_ledger
from .zoom_calibration import GalaxyMergerZoomCase, ZoomGrid, load_zoom_grid


MODEL_ZOOM_EXECUTION_CONTRACT_SCHEMA_VERSION = 1
_SHARED_INPUTS = (
    "initial_conditions",
    "baryon_configuration",
    "smbh_seed_catalog",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: str | Path, label: str) -> dict[str, str]:
    source = Path(path).expanduser().resolve()
    try:
        digest = _sha256(source)
    except OSError as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    return {"path": str(source), "sha256": digest}


def _shared_inputs(paths: Mapping[str, str | Path]) -> dict[str, dict[str, str]]:
    if set(paths) != set(_SHARED_INPUTS):
        raise ValueError(
            "shared inputs must name initial_conditions, baryon_configuration, and "
            "smbh_seed_catalog"
        )
    return {name: _artifact(paths[name], f"shared input {name}") for name in _SHARED_INPUTS}


def _case(grid: ZoomGrid, case_id: str) -> GalaxyMergerZoomCase:
    matches = [case for case in grid.cases if case.case_id == case_id]
    if len(matches) != 1:
        raise ValueError("case_id does not identify exactly one zoom manifest case")
    return matches[0]


def _capture_event(path: str | Path, event_uid: str) -> CaptureEvent:
    source = Path(path).expanduser().resolve()
    try:
        ledger = read_capture_ledger(source)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read capture ledger: {error}") from error
    matches = [event for event in ledger.events if event.event_uid == event_uid]
    if len(matches) != 1:
        raise ValueError("capture_event_uid does not identify exactly one complete event")
    event = matches[0]
    if event.classification != "BINARY" or event.binary_orbital_state is None:
        raise ValueError("model zoom identity requires one complete BINARY capture event")
    pair = event.pairs[0]
    if not (
        pair.within_numerical_merge_radius
        and pair.source_two_body_bound
        and event.binary_orbital_state.bound
    ):
        raise ValueError("model zoom identity requires a bound numerical binary capture")
    return event


def _identity(
    grid: ZoomGrid,
    case: GalaxyMergerZoomCase,
    event: CaptureEvent,
    shared_inputs: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    return {
        "model_zoom_manifest_sha256": grid.manifest_sha256,
        "model_zoom_case_id": case.case_id,
        "model_zoom_capture_event_sha256": event.event_sha256,
        "model_zoom_initial_conditions_sha256": shared_inputs["initial_conditions"][
            "sha256"
        ],
        "model_zoom_baryon_configuration_sha256": shared_inputs[
            "baryon_configuration"
        ]["sha256"],
        "model_zoom_sink_initial_conditions_sha256": shared_inputs["smbh_seed_catalog"][
            "sha256"
        ],
    }


def _controls(identity: Mapping[str, str]) -> str:
    return "\n".join(
        (
            "! Generated common CDM/SIDM/FDM model-zoom identity controls.",
            "! Insert once in the solver-consumed &PHYSICS_PARAMS group.",
            "&PHYSICS_PARAMS",
            *(f"{name}='{identity[name]}'" for name in sorted(identity)),
            "/",
            "",
        )
    )


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, path)


@dataclass(frozen=True)
class ModelZoomExecutionContract:
    """Exact input identity for a single all-model zoom realization."""

    specification_path: Path
    grid: ZoomGrid
    case: GalaxyMergerZoomCase
    capture_event: CaptureEvent
    shared_inputs: Mapping[str, Mapping[str, str]]

    def as_dict(self, *, controls_path: Path) -> dict[str, Any]:
        identity = _identity(self.grid, self.case, self.capture_event, self.shared_inputs)
        return {
            "schema_version": MODEL_ZOOM_EXECUTION_CONTRACT_SCHEMA_VERSION,
            "status": "ready_for_operator_namelist_insertion",
            "interpretation": (
                "exact model-zoom input identity only; this does not modify a run, "
                "submit a job, validate resolved physics, or estimate a coalescence time"
            ),
            "dark_matter_model": self.case.physics.dark_matter_model,
            "case": self.case.as_dict(),
            "zoom_manifest": {
                "path": str(self.specification_path),
                "sha256": _sha256(self.specification_path),
                "manifest_sha256": self.grid.manifest_sha256,
            },
            "capture_event": {
                "event_uid": self.capture_event.event_uid,
                "event_sha256": self.capture_event.event_sha256,
                "ledger_path": str(self.capture_event.source_path),
                "ledger_sha256": _sha256(self.capture_event.source_path),
            },
            "shared_inputs": {
                name: dict(self.shared_inputs[name]) for name in _SHARED_INPUTS
            },
            "model_zoom_execution_identity": identity,
            "required_physics_params_fragment": {
                "path": str(controls_path),
                "sha256": _sha256(controls_path),
            },
            "execution": "operator namelist insertion and Slurm submission only",
            "reasons": [],
        }


def materialize_model_zoom_execution_contract(
    *,
    specification_path: str | Path,
    case_id: str,
    capture_ledger_path: str | Path,
    capture_event_uid: str,
    shared_input_paths: Mapping[str, str | Path],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write one identity contract and controls fragment without touching a run."""

    specification = Path(specification_path).expanduser().resolve()
    try:
        grid = load_zoom_grid(specification)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load zoom manifest: {error}") from error
    case = _case(grid, case_id)
    event = _capture_event(capture_ledger_path, capture_event_uid)
    inputs = _shared_inputs(shared_input_paths)
    destination = Path(output_directory).expanduser().resolve()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("model-zoom contract output directory must not already exist") from error
    controls_path = destination / "required_model_zoom_identity.nml"
    _write_atomic(controls_path, _controls(_identity(grid, case, event, inputs)))
    contract = ModelZoomExecutionContract(
        specification_path=specification,
        grid=grid,
        case=case,
        capture_event=event,
        shared_inputs=inputs,
    )
    record = contract.as_dict(controls_path=controls_path)
    _write_atomic(
        destination / "model_zoom_execution_contract.json",
        json.dumps(record, indent=2, sort_keys=True) + "\n",
    )
    return record
