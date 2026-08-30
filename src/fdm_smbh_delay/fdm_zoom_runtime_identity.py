"""Bind completed pure-FDM normal outputs to one declared zoom input.

This module consumes the immutable copies and sidecars written by lagRamses.
It deliberately does not infer that every MPI shard exists, that a diagnostic
extractor ran, or that the resulting solitons are relaxed.  Those claims need
the later V3 wave provenance and a separately attested extractor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .dual_soliton_preflight import validate_pure_fdm_dual_soliton_runtime_identity
from .fdm_zoom_seed_binding import (
    FDMCaptureSeedZoomBinding,
    FDMDeclaredRunInputBinding,
    read_verified_fdm_capture_seed_zoom_binding,
    read_verified_fdm_declared_run_input_binding,
)
from .lagramses_fdm_provenance import read_lagramses_fdm_outer_wave_provenance
from .model_zoom_materialization import (
    VerifiedModelZoomExecutionContract,
    read_verified_model_zoom_execution_contract,
)
from .dm_run_provenance import read_dark_matter_run_provenance


FDM_ZOOM_RUNTIME_OUTPUT_IDENTITY_SCHEMA_VERSION = 1
_OUTPUT_DIRECTORY = re.compile(r"output_(\d{5})$")
_GROUP_DIRECTORY = re.compile(r"group_(\d{5})$")
_BUILD_SHA256 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


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


def _artifact(record: Any, label: str) -> tuple[Path, str]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} artifact is invalid")
    reference = record.get("path")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{label} artifact path is invalid")
    path = Path(reference).expanduser().resolve()
    digest = _sha256_field(record.get("sha256"), f"{label} SHA-256")
    try:
        actual = _sha256(path)
    except OSError as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if actual != digest:
        raise ValueError(f"{label} SHA-256 no longer matches")
    return path, digest


def _output_file(directory: Path, reference: str, label: str) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name in {"", ".", ".."}:
        raise ValueError(f"{label} must be a filename inside its metadata directory")
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


def _compilation_last_commit(path: Path) -> str:
    """Read the current lagRamses compilation sidecar's exact revision line."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read output compilation sidecar: {error}") from error
    matches: list[str] = []
    for line in lines:
        if "=" not in line:
            continue
        key, value = (item.strip() for item in line.split("=", 1))
        if key.lower() == "last commit":
            matches.append(value)
    if len(matches) != 1 or _BUILD_SHA256.fullmatch(matches[0]) is None:
        raise ValueError("output compilation sidecar requires one lowercase last commit")
    return matches[0]


@dataclass(frozen=True)
class _VerifiedFDMDeclaredRun:
    source_path: Path
    source_sha256: str
    declared: FDMDeclaredRunInputBinding
    fdm_binding: FDMCaptureSeedZoomBinding
    contract: VerifiedModelZoomExecutionContract
    expected_namelist_sha256: str


def _read_verified_declared_run(path: str | Path) -> _VerifiedFDMDeclaredRun:
    source = Path(path).expanduser().resolve()
    declared = read_verified_fdm_declared_run_input_binding(source)
    fdm_binding = read_verified_fdm_capture_seed_zoom_binding(
        declared.fdm_capture_seed_zoom_binding_path
    )
    contract = read_verified_model_zoom_execution_contract(
        fdm_binding.model_zoom_contract_path
    )
    if contract.case.physics.dark_matter_model != "fdm":
        raise ValueError("declared FDM run does not select an FDM zoom case")
    if declared.zoom_case_id != contract.case.case_id:
        raise ValueError("declared FDM run zoom case differs from its contract")
    if declared.seed_case_id != fdm_binding.seed_case_id:
        raise ValueError("declared FDM run seed case differs from its FDM binding")
    namelist = contract.run_namelist
    if not isinstance(namelist, Mapping) or set(namelist) != {"path", "sha256"}:
        raise ValueError("declared FDM run contract namelist is invalid")
    _, namelist_sha256 = _artifact(namelist, "declared FDM run namelist")
    return _VerifiedFDMDeclaredRun(
        source_path=source,
        source_sha256=_sha256(source),
        declared=declared,
        fdm_binding=fdm_binding,
        contract=contract,
        expected_namelist_sha256=namelist_sha256,
    )


def _expected_model_identity(contract: VerifiedModelZoomExecutionContract) -> dict[str, str]:
    return {
        "model_zoom_manifest_sha256": contract.grid.manifest_sha256,
        "model_zoom_case_id": contract.case.case_id,
        "model_zoom_capture_event_sha256": contract.capture_event.event_sha256,
        "model_zoom_initial_conditions_sha256": contract.shared_inputs[
            "initial_conditions"
        ]["sha256"],
        "model_zoom_baryon_configuration_sha256": contract.shared_inputs[
            "baryon_configuration"
        ]["sha256"],
        "model_zoom_sink_initial_conditions_sha256": contract.shared_inputs[
            "smbh_seed_catalog"
        ]["sha256"],
    }


def _metadata_directory(root: Path, output_number: str) -> Path:
    filename = f"dm_run_provenance_{output_number}.txt"
    candidates = [root]
    try:
        candidates.extend(
            child
            for child in root.iterdir()
            if child.is_dir() and _GROUP_DIRECTORY.fullmatch(child.name) is not None
        )
    except OSError as error:
        raise ValueError(f"cannot enumerate FDM output directory: {error}") from error
    matches = [candidate for candidate in candidates if (candidate / filename).is_file()]
    if len(matches) != 1:
        raise ValueError("FDM output must contain exactly one DM run-provenance sidecar")
    return matches[0].resolve()


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-14)


def _verify_output(
    root: Path,
    declared: _VerifiedFDMDeclaredRun,
) -> dict[str, Any]:
    match = _OUTPUT_DIRECTORY.fullmatch(root.name)
    if match is None:
        raise ValueError("each FDM output directory must be named output_00000")
    output_number = match.group(1)
    complete = root / "COMPLETE"
    try:
        if complete.read_text(encoding="utf-8").strip() != output_number:
            raise ValueError("output COMPLETE marker is invalid")
    except OSError as error:
        raise ValueError(f"cannot read output COMPLETE marker: {error}") from error
    metadata = _metadata_directory(root, output_number)
    run_path = metadata / f"dm_run_provenance_{output_number}.txt"
    run = read_dark_matter_run_provenance(run_path)
    if run.dark_matter_model != "fdm" or not run.fdm_enabled or run.sidm_enabled:
        raise ValueError("output does not record a pure FDM realization")
    if run.parameter("fdm_force_accounting") != "resolved_wave_only":
        raise ValueError("output FDM force accounting is not resolved_wave_only")
    if not run.parameter("fdm_outer_ledger_enabled"):
        raise ValueError("output does not record the required FDM raw-provenance writer")
    if not run.smbh_capture_ledger_enabled:
        raise ValueError("output records the SMBH capture ledger as disabled")
    if (
        run.parameter("smbh_merge_radius_cells") != 0.0
        or run.parameter("smbh_compaction_mode") != "no_finite_radius_rmerge_zero"
    ):
        raise ValueError("output does not record the required non-compacting SMBH mode")
    if _BUILD_SHA256.fullmatch(run.build_git_hash) is None:
        raise ValueError("output build_git_hash must be a 40-character lowercase revision")
    if run.build_git_hash != declared.declared.expected_build_git_hash:
        raise ValueError("output build_git_hash differs from the declared FDM run input")
    if run.parameter("model_zoom_execution_identity_status") != "available":
        raise ValueError("output does not attest the model-zoom execution identity")
    if run.parameter("model_zoom_levelmax") != declared.contract.case.numerics.levelmax:
        raise ValueError("output model zoom level differs from the declared FDM case")
    for name, expected in _expected_model_identity(declared.contract).items():
        if run.parameter(name) != expected:
            raise ValueError(f"output {name} differs from the declared FDM case")
    namelist_copy = _output_file(metadata, run.namelist_copy, "output namelist_copy")
    if _sha256(namelist_copy) != declared.expected_namelist_sha256:
        raise ValueError("output namelist copy differs from the declared FDM run input")
    compilation_copy = _output_file(metadata, run.compilation_copy, "output compilation_copy")
    compilation_hash = _sha256(compilation_copy)
    if _compilation_last_commit(compilation_copy) != run.build_git_hash:
        raise ValueError("output compilation last commit differs from DM run provenance")
    info_path = metadata / f"info_{output_number}.txt"
    info = _info_records(info_path)
    if not _close(run.time_code, info["time"]):
        raise ValueError("output run-provenance time differs from info time")
    if not _close(run.scale_factor, info["aexp"]):
        raise ValueError("output run-provenance scale factor differs from info aexp")
    raw_path = root / f"fdm_outer_wave_provenance_{output_number}.txt"
    raw = read_lagramses_fdm_outer_wave_provenance(raw_path)
    if raw.source_schema_version != 2:
        raise ValueError("FDM output requires dual-soliton raw provenance V2")
    if raw.psi_snapshot_prefix != f"fdm_{output_number}.out":
        raise ValueError("raw FDM psi snapshot prefix differs from its output number")
    seed_identity = validate_pure_fdm_dual_soliton_runtime_identity(
        seed_manifest_path=declared.fdm_binding.seed_manifest_path,
        provenance_path=raw_path,
    )
    if not seed_identity.verified:
        raise ValueError("raw FDM provenance does not match the declared dual-soliton seed")
    if (
        raw.nstep_coarse != run.nstep_coarse
        or not _close(raw.time_code, run.time_code)
        or not _close(raw.aexp, run.scale_factor)
    ):
        raise ValueError("raw FDM provenance time/step differs from DM run provenance")
    physics = declared.contract.case.physics
    numerics = declared.contract.case.numerics
    if physics.fdm_particle_mass_ev is None or numerics.fdm_use_hjm is None:
        raise ValueError("declared FDM case lacks required FDM controls")
    if (
        not _close(raw.m_axion_ev, float(physics.fdm_particle_mass_ev))
        or raw.fdm_use_hjm != numerics.fdm_use_hjm
        or raw.fdm_first_wave_level != numerics.fdm_first_wave_level
    ):
        raise ValueError("raw FDM controls differ from the declared FDM zoom case")
    if (
        not _close(float(run.parameter("m_axion_ev")), raw.m_axion_ev)
        or run.parameter("fdm_use_hjm") != raw.fdm_use_hjm
        or run.parameter("fdm_first_wave_level") != raw.fdm_first_wave_level
    ):
        raise ValueError("DM run-provenance FDM controls differ from raw wave provenance")
    return {
        "output_number": output_number,
        "output_directory": str(root),
        "metadata_directory": str(metadata),
        "complete_marker": {"path": str(complete), "sha256": _sha256(complete)},
        "dm_run_provenance": {
            "path": str(run_path),
            "sha256": run.source_sha256,
            "build_git_hash": run.build_git_hash,
        },
        "raw_fdm_provenance": {"path": str(raw_path), "sha256": raw.source_sha256},
        "namelist_copy": {"path": str(namelist_copy), "sha256": _sha256(namelist_copy)},
        "compilation_copy": {"path": str(compilation_copy), "sha256": compilation_hash},
        "info": {"path": str(info_path), "sha256": _sha256(info_path)},
        "nstep_coarse": run.nstep_coarse,
        "time_code": run.time_code,
        "aexp": run.scale_factor,
    }


def _output_set_reasons(outputs: list[dict[str, Any]]) -> list[str]:
    if not outputs:
        return []
    reasons: list[str] = []
    roots = {str(Path(record["output_directory"]).parent) for record in outputs}
    if len(roots) != 1:
        reasons.append("outputs do not share one run root")
    build_hashes = {record["dm_run_provenance"]["build_git_hash"] for record in outputs}
    if len(build_hashes) != 1:
        reasons.append("outputs do not share one build_git_hash")
    compilation_hashes = {record["compilation_copy"]["sha256"] for record in outputs}
    if len(compilation_hashes) != 1:
        reasons.append("outputs do not share one compilation-copy SHA-256")
    ordered = sorted(outputs, key=lambda record: record["output_number"])
    if any(
        left["time_code"] >= right["time_code"]
        for left, right in zip(ordered[:-1], ordered[1:])
    ):
        reasons.append("output times are not strictly increasing")
    if any(
        left["nstep_coarse"] >= right["nstep_coarse"]
        for left, right in zip(ordered[:-1], ordered[1:])
    ):
        reasons.append("output nstep_coarse values are not strictly increasing")
    if any(
        left["aexp"] > right["aexp"]
        for left, right in zip(ordered[:-1], ordered[1:])
    ):
        reasons.append("output aexp values decrease with time")
    return reasons


@dataclass(frozen=True)
class FDMZoomRuntimeOutputIdentity:
    """A recheckable, output-set identity for one declared pure-FDM zoom."""

    declared_run_path: Path
    declared_run_sha256: str
    zoom_case_id: str | None
    seed_case_id: str | None
    outputs: tuple[dict[str, Any], ...]
    status: str
    reasons: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return self.status == "fdm_runtime_output_identity_verified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FDM_ZOOM_RUNTIME_OUTPUT_IDENTITY_SCHEMA_VERSION,
            "status": self.status,
            "interpretation": (
                "completed-output identity only; this binds the listed normal-output "
                "metadata to one declared FDM seed/zoom input, but does not establish "
                "complete MPI shard coverage, solver input consumption beyond the copied "
                "namelist/model-zoom sidecar, relaxation, conservation, resolution, "
                "extractor execution, or a physical delay"
            ),
            "declared_run_input_binding": {
                "path": str(self.declared_run_path),
                "sha256": self.declared_run_sha256,
                "zoom_case_id": self.zoom_case_id,
                "seed_case_id": self.seed_case_id,
            },
            "complete_outputs": list(self.outputs),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class VerifiedFDMZoomRuntimeOutputs:
    """Current output-set identity available to later, non-estimating consumers."""

    source_path: Path
    source_sha256: str
    declared_run: _VerifiedFDMDeclaredRun
    outputs: tuple[dict[str, Any], ...]

    @property
    def raw_fdm_provenance_paths(self) -> tuple[Path, ...]:
        return tuple(Path(record["raw_fdm_provenance"]["path"]).resolve() for record in self.outputs)


def assess_fdm_declared_zoom_runtime_outputs(
    declared_run_input_binding_path: str | Path,
    output_directories: Iterable[str | Path],
) -> FDMZoomRuntimeOutputIdentity:
    """Verify listed normal outputs against one declared FDM run input.

    No output is discovered implicitly.  The operator must enumerate each
    output intended for downstream use, which prevents a partial directory
    scan from silently changing the temporal source set.
    """

    declared = _read_verified_declared_run(declared_run_input_binding_path)
    prepared: list[dict[str, Any]] = []
    reasons: list[str] = []
    seen: set[str] = set()
    for reference in output_directories:
        root = Path(reference).expanduser().resolve()
        match = _OUTPUT_DIRECTORY.fullmatch(root.name)
        label = match.group(1) if match is not None else root.name
        if label in seen:
            reasons.append(f"output {label} is duplicated")
            continue
        seen.add(label)
        try:
            prepared.append(_verify_output(root, declared))
        except (OSError, ValueError) as error:
            reasons.append(f"output {label}: {error}")
    if not prepared and not reasons:
        reasons.append("at least one completed FDM output is required")
    prepared.sort(key=lambda record: record["output_number"])
    reasons.extend(_output_set_reasons(prepared))
    return FDMZoomRuntimeOutputIdentity(
        declared_run_path=declared.source_path,
        declared_run_sha256=declared.source_sha256,
        zoom_case_id=declared.declared.zoom_case_id,
        seed_case_id=declared.declared.seed_case_id,
        outputs=tuple(prepared),
        status=(
            "fdm_runtime_output_identity_verified"
            if not reasons
            else "fdm_runtime_output_identity_not_verified"
        ),
        reasons=tuple(reasons),
    )


def read_verified_fdm_declared_zoom_runtime_outputs(
    path: str | Path,
) -> VerifiedFDMZoomRuntimeOutputs:
    """Revalidate a saved FDM output-set identity before consuming it."""

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read FDM runtime output identity: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "interpretation",
        "declared_run_input_binding",
        "complete_outputs",
        "reasons",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_fields
        or record.get("schema_version") != FDM_ZOOM_RUNTIME_OUTPUT_IDENTITY_SCHEMA_VERSION
        or record.get("status") != "fdm_runtime_output_identity_verified"
        or record.get("reasons") != []
    ):
        raise ValueError("FDM runtime output identity is not verified")
    declared_record = record.get("declared_run_input_binding")
    if not isinstance(declared_record, Mapping) or set(declared_record) != {
        "path",
        "sha256",
        "zoom_case_id",
        "seed_case_id",
    }:
        raise ValueError("FDM runtime output identity declared-run source is invalid")
    declared_reference = declared_record.get("path")
    if not isinstance(declared_reference, str) or not declared_reference.strip():
        raise ValueError("FDM runtime output identity declared-run path is invalid")
    declared_path = Path(declared_reference).expanduser().resolve()
    declared_hash = _sha256_field(
        declared_record.get("sha256"), "FDM declared-run binding SHA-256"
    )
    try:
        if _sha256(declared_path) != declared_hash:
            raise ValueError("FDM declared-run binding SHA-256 no longer matches")
    except OSError as error:
        raise ValueError(f"cannot read FDM declared-run binding: {error}") from error
    declared = _read_verified_declared_run(declared_path)
    if declared_hash != declared.source_sha256 or declared_record != {
        "path": str(declared.source_path),
        "sha256": declared.source_sha256,
        "zoom_case_id": declared.declared.zoom_case_id,
        "seed_case_id": declared.declared.seed_case_id,
    }:
        raise ValueError("FDM runtime output identity declared-run source differs")
    output_records = record.get("complete_outputs")
    if not isinstance(output_records, list) or not output_records:
        raise ValueError("FDM runtime output identity requires complete outputs")
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for saved in output_records:
        if not isinstance(saved, Mapping) or not isinstance(saved.get("output_directory"), str):
            raise ValueError("FDM runtime output identity output record is invalid")
        actual = _verify_output(Path(saved["output_directory"]).expanduser().resolve(), declared)
        if actual != dict(saved):
            raise ValueError("FDM runtime output identity output metadata no longer matches")
        if actual["output_number"] in seen:
            raise ValueError("FDM runtime output identity duplicates an output")
        seen.add(actual["output_number"])
        verified.append(actual)
    verified.sort(key=lambda record: record["output_number"])
    reasons = _output_set_reasons(verified)
    if reasons:
        raise ValueError("FDM runtime output identity is no longer valid: " + "; ".join(reasons))
    expected = FDMZoomRuntimeOutputIdentity(
        declared_run_path=declared.source_path,
        declared_run_sha256=declared.source_sha256,
        zoom_case_id=declared.declared.zoom_case_id,
        seed_case_id=declared.declared.seed_case_id,
        outputs=tuple(verified),
        status="fdm_runtime_output_identity_verified",
        reasons=(),
    )
    if expected.as_dict() != record:
        raise ValueError("FDM runtime output identity no longer matches its source inputs")
    return VerifiedFDMZoomRuntimeOutputs(
        source_path=source,
        source_sha256=_sha256(source),
        declared_run=declared,
        outputs=tuple(verified),
    )
