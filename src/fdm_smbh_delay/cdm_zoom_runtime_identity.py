"""Verify that completed CDM zoom outputs match one materialized run contract.

This module consumes only complete output metadata and copied input files.  It
does not infer an environmental delay, accept a secular rate, or submit work.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .cdm_zoom import assess_noncompacting_cdm_zoom_run
from .cdm_zoom_materialization import assess_cdm_noncompacting_zoom_run_inputs
from .cdm_zoom_plan import CDMNonCompactingZoomPlan, load_cdm_noncompacting_zoom_plan
from .dm_run_provenance import read_dark_matter_run_provenance
from .lagramses_cdm_orbit import read_bound_cdm_capture
from .zoom_calibration import GalaxyMergerZoomCase


CDM_NONCOMPACTING_ZOOM_RUNTIME_IDENTITY_SCHEMA_VERSION = 1
_OUTPUT_DIRECTORY = re.compile(r"output_(\d{5})$")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MYR_SECONDS = 3.15576e13
_CASE_INPUT_ARTIFACTS = {
    "host_orbit_initial_conditions",
    "initial_conditions",
    "sink_initial_conditions",
}
_EXECUTION_IDENTITY_FIELDS = {
    "cdm_zoom_plan_manifest_sha256",
    "cdm_zoom_capture_event_sha256",
    "cdm_zoom_host_orbit_initial_conditions_sha256",
    "cdm_zoom_initial_conditions_sha256",
    "cdm_zoom_sink_initial_conditions_sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_field(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _path_field(record: Mapping[str, Any], field: str, label: str) -> Path:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} {field} is required")
    return Path(value).expanduser().resolve()


def _artifact(record: Mapping[str, Any], field: str, label: str) -> tuple[Path, str]:
    source = record.get(field)
    if not isinstance(source, Mapping):
        raise ValueError(f"{label} {field} is invalid")
    path = _path_field(source, "path", f"{label} {field}")
    digest = _sha256_field(source.get("sha256"), f"{label} {field} SHA-256")
    try:
        actual_digest = _sha256(path)
    except OSError as error:
        raise ValueError(f"cannot read {label} {field}: {error}") from error
    if actual_digest != digest:
        raise ValueError(f"{label} {field} SHA-256 no longer matches")
    return path, digest


def _output_file(directory: Path, reference: str, label: str) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name in {"", ".", ".."}:
        raise ValueError(f"{label} must be a filename inside its output directory")
    return directory / candidate


def _finite_number(value: str, label: str) -> float:
    try:
        result = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _info_records(path: Path) -> dict[str, float]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read output info file: {error}") from error
    expected = {"time", "aexp", "unit_t"}
    records: dict[str, float] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = (item.strip() for item in line.split("=", 1))
        if key not in expected:
            continue
        if key in records:
            raise ValueError(f"output info file duplicates {key}")
        records[key] = _finite_number(value, f"output info {key}")
    if set(records) != expected or records["aexp"] <= 0.0 or records["unit_t"] <= 0.0:
        raise ValueError("output info file must contain finite time/aexp/unit_t")
    return records


@dataclass(frozen=True)
class _VerifiedCDMZoomContract:
    source_path: Path
    source_sha256: str
    plan: CDMNonCompactingZoomPlan
    case: GalaxyMergerZoomCase
    namelist_sha256: str
    capture_ledger_file: str
    capture_binding: dict[str, Any]
    expected_build_git_hash: str
    expected_compilation_sha256: str
    case_input_artifact_sha256s: dict[str, str]
    execution_identity: dict[str, str]


def _read_verified_contract(path: str | Path) -> _VerifiedCDMZoomContract:
    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM zoom run contract: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "interpretation",
        "dark_matter_model",
        "case",
        "plan",
        "capture_binding",
        "case_input_identity",
        "run_inputs",
        "sampling_requirements",
        "execution",
        "reasons",
    }
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise ValueError("CDM zoom run contract fields are invalid")
    if (
        record.get("schema_version") != 1
        or record.get("status") != "ready_for_operator_submission"
        or record.get("dark_matter_model") != "cdm"
        or record.get("reasons") != []
    ):
        raise ValueError("CDM zoom run contract is not a ready CDM input")
    plan_record = record.get("plan")
    if not isinstance(plan_record, Mapping):
        raise ValueError("CDM zoom run contract plan is invalid")
    plan_path = _path_field(plan_record, "path", "CDM zoom run contract plan")
    if _sha256(plan_path) != _sha256_field(plan_record.get("sha256"), "CDM zoom plan SHA-256"):
        raise ValueError("CDM zoom plan SHA-256 no longer matches")
    plan = load_cdm_noncompacting_zoom_plan(plan_path)
    if plan.grid.manifest_sha256 != plan_record.get("manifest_sha256"):
        raise ValueError("CDM zoom plan manifest identity no longer matches")
    case_record = record.get("case")
    if not isinstance(case_record, Mapping) or not isinstance(case_record.get("case_id"), str):
        raise ValueError("CDM zoom run contract case is invalid")
    matches = [case for case in plan.grid.cases if case.case_id == case_record["case_id"]]
    if len(matches) != 1 or matches[0].as_dict() != dict(case_record):
        raise ValueError("CDM zoom contract case differs from the current plan")
    case = matches[0]

    binding = record.get("capture_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("CDM zoom run contract capture binding is invalid")
    binding_path = _path_field(binding, "path", "CDM zoom run contract capture binding")
    capture_uid = binding.get("capture_event_uid")
    primary_id = binding.get("primary_sink_id")
    secondary_id = binding.get("secondary_sink_id")
    if (
        not isinstance(capture_uid, str)
        or isinstance(primary_id, bool)
        or not isinstance(primary_id, int)
        or isinstance(secondary_id, bool)
        or not isinstance(secondary_id, int)
    ):
        raise ValueError("CDM zoom run contract capture-pair identity is invalid")
    if read_bound_cdm_capture(
        binding_path,
        capture_event_uid=capture_uid,
        primary_sink_id=primary_id,
        secondary_sink_id=secondary_id,
    ) != dict(binding):
        raise ValueError("CDM zoom capture binding no longer matches its ledger event")

    case_identity = record.get("case_input_identity")
    if not isinstance(case_identity, Mapping) or set(case_identity) != {
        "expected_build_git_hash",
        "expected_compilation",
        "input_artifacts",
        "execution_identity",
    }:
        raise ValueError("CDM zoom run contract case-input identity is invalid")
    expected_build = case_identity.get("expected_build_git_hash")
    if not isinstance(expected_build, str) or re.fullmatch(r"[0-9a-f]{40}", expected_build) is None:
        raise ValueError("CDM zoom run contract expected build identity is invalid")
    compilation_path, compilation_sha256 = _artifact(
        case_identity, "expected_compilation", "CDM zoom run contract"
    )
    artifact_records = case_identity.get("input_artifacts")
    if not isinstance(artifact_records, Mapping) or set(artifact_records) != _CASE_INPUT_ARTIFACTS:
        raise ValueError("CDM zoom run contract input-artifact fields are invalid")
    artifact_paths: dict[str, Path] = {}
    artifact_sha256s: dict[str, str] = {}
    for name in sorted(_CASE_INPUT_ARTIFACTS):
        artifact_path, artifact_sha = _artifact(
            artifact_records, name, "CDM zoom run contract input artifact"
        )
        artifact_paths[name] = artifact_path
        artifact_sha256s[name] = artifact_sha
    execution_identity = case_identity.get("execution_identity")
    expected_execution_identity = {
        "cdm_zoom_plan_manifest_sha256": plan.grid.manifest_sha256,
        "cdm_zoom_capture_event_sha256": binding["capture_event_sha256"],
        "cdm_zoom_host_orbit_initial_conditions_sha256": artifact_sha256s[
            "host_orbit_initial_conditions"
        ],
        "cdm_zoom_initial_conditions_sha256": artifact_sha256s["initial_conditions"],
        "cdm_zoom_sink_initial_conditions_sha256": artifact_sha256s[
            "sink_initial_conditions"
        ],
    }
    if (
        not isinstance(execution_identity, Mapping)
        or set(execution_identity) != _EXECUTION_IDENTITY_FIELDS
        or any(
            _sha256_field(execution_identity.get(name), name) != expected
            for name, expected in expected_execution_identity.items()
        )
    ):
        raise ValueError("CDM zoom run contract execution identity is invalid")

    run_inputs = record.get("run_inputs")
    if not isinstance(run_inputs, Mapping):
        raise ValueError("CDM zoom run contract run inputs are invalid")
    namelist_path, namelist_sha256 = _artifact(run_inputs, "namelist", "CDM zoom run contract")
    controls_path, _ = _artifact(
        run_inputs, "required_smbh_controls_fragment", "CDM zoom run contract"
    )
    if controls_path.parent != source.parent:
        raise ValueError("CDM zoom controls fragment must remain in its contract directory")
    ledger_file = run_inputs.get("smbh_capture_ledger_file")
    if not isinstance(ledger_file, str) or not ledger_file.strip():
        raise ValueError("CDM zoom run contract capture ledger file is invalid")
    decision, _ = assess_cdm_noncompacting_zoom_run_inputs(
        specification_path=plan_path,
        case_id=case.case_id,
        capture_binding_path=binding_path,
        capture_event_uid=capture_uid,
        primary_sink_id=primary_id,
        secondary_sink_id=secondary_id,
        run_namelist_path=namelist_path,
        capture_ledger_file=ledger_file,
        expected_build_git_hash=expected_build,
        expected_compilation_path=compilation_path,
        case_input_artifact_paths=artifact_paths,
    )
    if not decision.ready:
        raise ValueError("CDM zoom run input no longer passes non-compacting preflight")
    return _VerifiedCDMZoomContract(
        source_path=source,
        source_sha256=_sha256(source),
        plan=plan,
        case=case,
        namelist_sha256=namelist_sha256,
        capture_ledger_file=ledger_file,
        capture_binding=dict(binding),
        expected_build_git_hash=expected_build,
        expected_compilation_sha256=compilation_sha256,
        case_input_artifact_sha256s=artifact_sha256s,
        execution_identity=expected_execution_identity,
    )


@dataclass(frozen=True)
class CDMNonCompactingZoomRuntimeIdentity:
    """Identity decision for completed outputs from one non-compacting zoom."""

    contract: _VerifiedCDMZoomContract
    outputs: tuple[dict[str, Any], ...]
    status: str
    reasons: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return self.status == "runtime_identity_verified"

    def as_dict(self) -> dict[str, Any]:
        complete_count = len(self.outputs)
        return {
            "schema_version": CDM_NONCOMPACTING_ZOOM_RUNTIME_IDENTITY_SCHEMA_VERSION,
            "status": self.status,
            "interpretation": (
                "completed-output identity only; this does not accept resolution, "
                "phase, force/conservation, secular-rate, or coalescence evidence"
            ),
            "dark_matter_model": "cdm",
            "contract": {
                "path": str(self.contract.source_path),
                "sha256": self.contract.source_sha256,
                "case_id": self.contract.case.case_id,
                "manifest_sha256": self.contract.plan.grid.manifest_sha256,
            },
            "complete_outputs": list(self.outputs),
            "listed_output_count": {
                "count": complete_count,
                "minimum_complete_outputs": self.contract.plan.minimum_complete_outputs,
                "status": (
                    "minimum_planned_output_count_reached"
                    if complete_count >= self.contract.plan.minimum_complete_outputs
                    else "below_minimum_planned_output_count"
                ),
            },
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class VerifiedCDMNonCompactingZoomOutputs:
    """Re-read identity-verified output metadata for the orbit extractor."""

    source_path: Path
    source_sha256: str
    contract: _VerifiedCDMZoomContract
    outputs: tuple[dict[str, Any], ...]

    @property
    def output_directories(self) -> tuple[Path, ...]:
        return tuple(Path(record["directory"]).resolve() for record in self.outputs)


def read_verified_cdm_noncompacting_zoom_runtime_identity(
    path: str | Path,
) -> VerifiedCDMNonCompactingZoomOutputs:
    """Require a current verified identity record before orbit extraction.

    The record alone is insufficient: this re-reads the contract and every
    listed output, so a later sidecar, namelist-copy, build, or cadence change
    invalidates the prospective orbit input.
    """

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM zoom runtime identity: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "interpretation",
        "dark_matter_model",
        "contract",
        "complete_outputs",
        "listed_output_count",
        "reasons",
    }
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise ValueError("CDM zoom runtime identity fields are invalid")
    if (
        record.get("schema_version") != 1
        or record.get("status") != "runtime_identity_verified"
        or record.get("dark_matter_model") != "cdm"
        or record.get("reasons") != []
    ):
        raise ValueError("CDM zoom runtime identity is not verified")
    contract_record = record.get("contract")
    if not isinstance(contract_record, Mapping):
        raise ValueError("CDM zoom runtime identity contract is invalid")
    contract_path = _path_field(contract_record, "path", "CDM zoom runtime identity contract")
    if _sha256(contract_path) != _sha256_field(
        contract_record.get("sha256"), "CDM zoom runtime identity contract SHA-256"
    ):
        raise ValueError("CDM zoom runtime identity contract SHA-256 no longer matches")
    contract = _read_verified_contract(contract_path)
    if contract_record != {
        "path": str(contract.source_path),
        "sha256": contract.source_sha256,
        "case_id": contract.case.case_id,
        "manifest_sha256": contract.plan.grid.manifest_sha256,
    }:
        raise ValueError("CDM zoom runtime identity contract differs from its current input")
    output_records = record.get("complete_outputs")
    if not isinstance(output_records, list) or not output_records:
        raise ValueError("CDM zoom runtime identity requires complete outputs")
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for saved in output_records:
        if not isinstance(saved, Mapping) or not isinstance(saved.get("directory"), str):
            raise ValueError("CDM zoom runtime identity output record is invalid")
        actual = _verify_output(Path(saved["directory"]).expanduser().resolve(), contract)
        if actual != dict(saved):
            raise ValueError("CDM zoom runtime identity output metadata no longer matches")
        if actual["output_number"] in seen:
            raise ValueError("CDM zoom runtime identity outputs are duplicated")
        seen.add(actual["output_number"])
        verified.append(actual)
    verified.sort(key=lambda item: item["output_number"])
    reasons = _output_set_reasons(verified, contract)
    if reasons:
        raise ValueError("CDM zoom runtime identity is no longer valid: " + "; ".join(reasons))
    count = record.get("listed_output_count")
    expected_count = {
        "count": len(verified),
        "minimum_complete_outputs": contract.plan.minimum_complete_outputs,
        "status": (
            "minimum_planned_output_count_reached"
            if len(verified) >= contract.plan.minimum_complete_outputs
            else "below_minimum_planned_output_count"
        ),
    }
    if count != expected_count:
        raise ValueError("CDM zoom runtime identity output-count record is invalid")
    return VerifiedCDMNonCompactingZoomOutputs(
        source_path=source,
        source_sha256=_sha256(source),
        contract=contract,
        outputs=tuple(verified),
    )


def _verify_output(
    directory: Path,
    contract: _VerifiedCDMZoomContract,
) -> dict[str, Any]:
    match = _OUTPUT_DIRECTORY.fullmatch(directory.name)
    if match is None:
        raise ValueError("each lagRamses output directory must be output_00000 form")
    output_number = match.group(1)
    complete = directory / "COMPLETE"
    try:
        if complete.read_text(encoding="utf-8").strip() != output_number:
            raise ValueError("output COMPLETE marker is invalid")
    except OSError as error:
        raise ValueError(f"cannot read output COMPLETE marker: {error}") from error
    provenance_path = directory / f"dm_run_provenance_{output_number}.txt"
    provenance = read_dark_matter_run_provenance(provenance_path)
    if not assess_noncompacting_cdm_zoom_run(provenance).ready:
        raise ValueError("output does not prove non-compacting CDM provenance")
    if provenance.smbh_capture_ledger_file != contract.capture_ledger_file:
        raise ValueError("output capture-ledger setting differs from its run contract")
    if provenance.build_git_hash != contract.expected_build_git_hash:
        raise ValueError("output build_git_hash differs from its run contract")
    if provenance.parameter("cdm_zoom_execution_identity_status") != "available":
        raise ValueError("output does not attest the contracted CDM zoom input identity")
    for name, expected in contract.execution_identity.items():
        if provenance.parameter(name) != expected:
            raise ValueError(f"output {name} differs from its run contract")
    namelist_copy = _output_file(directory, provenance.namelist_copy, "output namelist_copy")
    if _sha256(namelist_copy) != contract.namelist_sha256:
        raise ValueError("output namelist copy SHA-256 differs from its run contract")
    compilation_copy = _output_file(
        directory, provenance.compilation_copy, "output compilation_copy"
    )
    if _sha256(compilation_copy) != contract.expected_compilation_sha256:
        raise ValueError("output compilation copy SHA-256 differs from its run contract")
    info_path = directory / f"info_{output_number}.txt"
    info = _info_records(info_path)
    if not math.isclose(
        provenance.time_code, info["time"], rel_tol=1.0e-12, abs_tol=1.0e-14
    ):
        raise ValueError("output provenance time_code differs from info time")
    if not math.isclose(
        provenance.scale_factor, info["aexp"], rel_tol=1.0e-12, abs_tol=1.0e-14
    ):
        raise ValueError("output provenance aexp differs from info aexp")
    time_myr = info["time"] * info["unit_t"] / _MYR_SECONDS
    if not math.isfinite(time_myr):
        raise ValueError("output proper time is invalid")
    return {
        "output_number": output_number,
        "directory": str(directory),
        "complete_marker": {"path": str(complete), "sha256": _sha256(complete)},
        "dm_run_provenance": {
            "path": str(provenance_path),
            "sha256": provenance.source_sha256,
            "build_git_hash": provenance.build_git_hash,
        },
        "namelist_copy": {"path": str(namelist_copy), "sha256": _sha256(namelist_copy)},
        "compilation_copy": {
            "path": str(compilation_copy),
            "sha256": _sha256(compilation_copy),
        },
        "info": {"path": str(info_path), "sha256": _sha256(info_path)},
        "nstep_coarse": provenance.nstep_coarse,
        "time_code": provenance.time_code,
        "aexp": provenance.scale_factor,
        "time_myr": time_myr,
    }


def _output_set_reasons(
    outputs: list[dict[str, Any]], contract: _VerifiedCDMZoomContract
) -> list[str]:
    if not outputs:
        return []
    reasons: list[str] = []
    roots = {str(Path(record["directory"]).parent) for record in outputs}
    if len(roots) != 1:
        reasons.append("outputs do not share one run root")
    build_hashes = {record["dm_run_provenance"]["build_git_hash"] for record in outputs}
    if len(build_hashes) != 1:
        reasons.append("outputs do not share one build_git_hash")
    compilation_hashes = {record["compilation_copy"]["sha256"] for record in outputs}
    if len(compilation_hashes) != 1:
        reasons.append("outputs do not share one compilation-copy SHA-256")
    ordered = sorted(outputs, key=lambda record: record["time_myr"])
    if any(
        left["time_myr"] >= right["time_myr"]
        for left, right in zip(ordered[:-1], ordered[1:])
    ):
        reasons.append("output proper times are not strictly increasing")
    if any(
        left["nstep_coarse"] >= right["nstep_coarse"]
        for left, right in zip(ordered[:-1], ordered[1:])
    ):
        reasons.append("output nstep_coarse values are not strictly increasing")
    if any(
        left["aexp"] > right["aexp"]
        for left, right in zip(ordered[:-1], ordered[1:])
    ):
        reasons.append("output aexp values decrease with proper time")
    if any(
        right["time_myr"] - left["time_myr"]
        > contract.plan.maximum_output_cadence_myr * (1.0 + 1.0e-12)
        for left, right in zip(ordered[:-1], ordered[1:])
    ):
        reasons.append("output proper-time cadence exceeds the contracted maximum")
    return reasons


def assess_cdm_noncompacting_zoom_runtime_identity(
    contract_path: str | Path,
    output_directories: Iterable[str | Path],
) -> CDMNonCompactingZoomRuntimeIdentity:
    """Verify every complete output against its immutable ready run contract."""

    contract = _read_verified_contract(contract_path)
    prepared: list[dict[str, Any]] = []
    reasons: list[str] = []
    seen_numbers: set[str] = set()
    for reference in output_directories:
        directory = Path(reference).expanduser().resolve()
        match = _OUTPUT_DIRECTORY.fullmatch(directory.name)
        output_label = match.group(1) if match is not None else directory.name
        if output_label in seen_numbers:
            reasons.append(f"output {output_label} is duplicated")
            continue
        seen_numbers.add(output_label)
        try:
            prepared.append(_verify_output(directory, contract))
        except (OSError, ValueError) as error:
            reasons.append(f"output {output_label}: {error}")
    if not prepared and not reasons:
        reasons.append("at least one complete lagRamses output is required")
    prepared.sort(key=lambda record: record["output_number"])
    reasons.extend(_output_set_reasons(prepared, contract))
    return CDMNonCompactingZoomRuntimeIdentity(
        contract=contract,
        outputs=tuple(prepared),
        status="runtime_identity_verified" if not reasons else "runtime_identity_not_verified",
        reasons=tuple(reasons),
    )
