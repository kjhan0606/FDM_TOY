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
import re
import tempfile
from typing import Any, Mapping

from .capture_ledger import CaptureEvent, read_capture_ledger
from .zoom_calibration import GalaxyMergerZoomCase, ZoomGrid, load_zoom_grid


MODEL_ZOOM_EXECUTION_CONTRACT_SCHEMA_VERSION = 2
_SHARED_INPUTS = (
    "initial_conditions",
    "baryon_configuration",
    "smbh_seed_catalog",
)
_ASSIGNMENT = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z0-9_]*)[ \t]*=[ \t]*([^!,/\r\n]+)", re.MULTILINE
)
_PHYSICS_PARAMS_GROUP = re.compile(
    r"^[ \t]*&[ \t]*physics_params\b(?P<body>.*?)(?:^[ \t]*/[ \t]*(?:!.*)?$|^[ \t]*&end\b)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
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
            "! Generated common CDM/SIDM/FDM model-zoom identity assignments.",
            "! Merge these assignments exactly once into the existing solver-consumed",
            "! &PHYSICS_PARAMS group; do not append a second namelist group.",
            *(f"{name}='{identity[name]}'" for name in sorted(identity)),
            "",
        )
    )


def _fortran_string(value: str) -> str:
    normalized = value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1]
    if not normalized:
        raise ValueError("model zoom identity assignment cannot be empty")
    return normalized


def _verified_namelist_identity(
    path: str | Path, identity: Mapping[str, str]
) -> dict[str, str]:
    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read lagRamses run namelist: {error}") from error
    groups = list(_PHYSICS_PARAMS_GROUP.finditer(text))
    if len(groups) != 1:
        raise ValueError("run namelist must contain exactly one &PHYSICS_PARAMS group")
    assignments: dict[str, list[str]] = {}
    for name, value in _ASSIGNMENT.findall(groups[0].group("body")):
        assignments.setdefault(name.lower(), []).append(value.strip())
    for name, expected in identity.items():
        values = assignments.get(name, [])
        if len(values) != 1:
            raise ValueError(f"run namelist must assign {name} exactly once")
        actual = _fortran_string(values[0])
        if actual != expected:
            raise ValueError(f"run namelist {name} differs from the materialized identity")
    return {"path": str(source), "sha256": _sha256(source)}


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
    run_namelist: Mapping[str, str]

    def as_dict(self, *, controls_path: Path) -> dict[str, Any]:
        identity = _identity(self.grid, self.case, self.capture_event, self.shared_inputs)
        return {
            "schema_version": MODEL_ZOOM_EXECUTION_CONTRACT_SCHEMA_VERSION,
            "status": "declared_namelist_identity_verified",
            "interpretation": (
                "declared namelist identity only; this does not prove that the capture "
                "belongs to this DM realization or that lagRamses consumed the three "
                "external inputs, submit a job, validate resolved physics, or estimate "
                "a coalescence time"
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
            "required_physics_params_assignments": {
                "path": str(controls_path),
                "sha256": _sha256(controls_path),
            },
            "run_namelist": dict(self.run_namelist),
            "execution": "not authorized for submission by this declaration alone",
            "reasons": [],
        }


@dataclass(frozen=True)
class VerifiedModelZoomExecutionContract:
    """A current, declaration-level contract revalidated from source files."""

    source_path: Path
    source_sha256: str
    grid: ZoomGrid
    case: GalaxyMergerZoomCase
    capture_event: CaptureEvent
    shared_inputs: Mapping[str, Mapping[str, str]]
    run_namelist: Mapping[str, str]


def read_verified_model_zoom_execution_contract(
    path: str | Path,
) -> VerifiedModelZoomExecutionContract:
    """Re-read every declaration input and reject a stale model-zoom contract."""

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read model-zoom execution contract: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "interpretation",
        "dark_matter_model",
        "case",
        "zoom_manifest",
        "capture_event",
        "shared_inputs",
        "model_zoom_execution_identity",
        "required_physics_params_assignments",
        "run_namelist",
        "execution",
        "reasons",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_fields
        or record.get("schema_version") != MODEL_ZOOM_EXECUTION_CONTRACT_SCHEMA_VERSION
        or record.get("status") != "declared_namelist_identity_verified"
        or record.get("reasons") != []
    ):
        raise ValueError("model-zoom execution contract is not a verified declaration")
    manifest = record.get("zoom_manifest")
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "path",
        "sha256",
        "manifest_sha256",
    }:
        raise ValueError("model-zoom execution contract manifest is invalid")
    specification = Path(manifest["path"]).expanduser().resolve()
    if _sha256(specification) != manifest.get("sha256"):
        raise ValueError("model-zoom execution contract manifest SHA-256 no longer matches")
    grid = load_zoom_grid(specification)
    if grid.manifest_sha256 != manifest.get("manifest_sha256"):
        raise ValueError("model-zoom execution contract manifest identity no longer matches")
    case_record = record.get("case")
    if not isinstance(case_record, Mapping) or not isinstance(case_record.get("case_id"), str):
        raise ValueError("model-zoom execution contract case is invalid")
    case = _case(grid, case_record["case_id"])
    if case.as_dict() != dict(case_record) or record.get("dark_matter_model") != case.physics.dark_matter_model:
        raise ValueError("model-zoom execution contract case no longer matches its manifest")
    capture = record.get("capture_event")
    if not isinstance(capture, Mapping) or set(capture) != {
        "event_uid",
        "event_sha256",
        "ledger_path",
        "ledger_sha256",
    }:
        raise ValueError("model-zoom execution contract capture event is invalid")
    ledger_path = Path(capture["ledger_path"]).expanduser().resolve()
    if _sha256(ledger_path) != capture.get("ledger_sha256"):
        raise ValueError("model-zoom execution contract capture ledger SHA-256 no longer matches")
    event = _capture_event(ledger_path, capture["event_uid"])
    if event.event_sha256 != capture.get("event_sha256"):
        raise ValueError("model-zoom execution contract capture event SHA-256 no longer matches")
    shared_record = record.get("shared_inputs")
    if not isinstance(shared_record, Mapping) or set(shared_record) != set(_SHARED_INPUTS):
        raise ValueError("model-zoom execution contract shared inputs are invalid")
    shared_inputs = _shared_inputs(
        {
            name: shared_record[name].get("path")
            if isinstance(shared_record[name], Mapping)
            else None
            for name in _SHARED_INPUTS
        }
    )
    if shared_inputs != {name: dict(shared_record[name]) for name in _SHARED_INPUTS}:
        raise ValueError("model-zoom execution contract shared-input hashes no longer match")
    identity = _identity(grid, case, event, shared_inputs)
    if record.get("model_zoom_execution_identity") != identity:
        raise ValueError("model-zoom execution contract identity no longer matches")
    run_namelist_record = record.get("run_namelist")
    if not isinstance(run_namelist_record, Mapping) or set(run_namelist_record) != {"path", "sha256"}:
        raise ValueError("model-zoom execution contract run namelist is invalid")
    run_namelist = _verified_namelist_identity(run_namelist_record["path"], identity)
    if run_namelist != dict(run_namelist_record):
        raise ValueError("model-zoom execution contract run namelist no longer matches")
    assignments = record.get("required_physics_params_assignments")
    if not isinstance(assignments, Mapping) or set(assignments) != {"path", "sha256"}:
        raise ValueError("model-zoom execution contract identity assignments are invalid")
    assignments_path = Path(assignments["path"]).expanduser().resolve()
    if _sha256(assignments_path) != assignments.get("sha256"):
        raise ValueError("model-zoom execution contract identity assignments SHA-256 no longer matches")
    try:
        if assignments_path.read_text(encoding="utf-8") != _controls(identity):
            raise ValueError("model-zoom execution contract identity assignments differ")
    except OSError as error:
        raise ValueError(f"cannot read model-zoom identity assignments: {error}") from error
    return VerifiedModelZoomExecutionContract(
        source_path=source,
        source_sha256=_sha256(source),
        grid=grid,
        case=case,
        capture_event=event,
        shared_inputs=shared_inputs,
        run_namelist=run_namelist,
    )


def materialize_model_zoom_execution_contract(
    *,
    specification_path: str | Path,
    case_id: str,
    capture_ledger_path: str | Path,
    capture_event_uid: str,
    shared_input_paths: Mapping[str, str | Path],
    run_namelist_path: str | Path,
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
    identity = _identity(grid, case, event, inputs)
    run_namelist = _verified_namelist_identity(run_namelist_path, identity)
    destination = Path(output_directory).expanduser().resolve()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("model-zoom contract output directory must not already exist") from error
    controls_path = destination / "required_model_zoom_identity.assignments"
    _write_atomic(controls_path, _controls(identity))
    contract = ModelZoomExecutionContract(
        specification_path=specification,
        grid=grid,
        case=case,
        capture_event=event,
        shared_inputs=inputs,
        run_namelist=run_namelist,
    )
    record = contract.as_dict(controls_path=controls_path)
    _write_atomic(
        destination / "model_zoom_execution_contract.json",
        json.dumps(record, indent=2, sort_keys=True) + "\n",
    )
    return record
