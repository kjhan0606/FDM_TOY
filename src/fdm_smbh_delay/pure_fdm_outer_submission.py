"""Non-submitting readiness gate for the pure-FDM outer zoom campaign.

The outer manifest and preflight are declarations, not solver outputs.  This
module re-reads both files, compares them with the canonical specification,
and requires a source scan plus an operator-produced runtime writer test
record before it can return a submission-ready status.  It never constructs a
Slurm command and never submits a job.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .dm_run_provenance import read_dark_matter_run_provenance
from .lagramses_fdm_provenance import read_lagramses_fdm_outer_wave_provenance
from .lagramses_writer_audit import (
    WriterForceAccountingAudit,
    audit_lagramses_writer_force_accounting,
)
from .pure_fdm_zoom import preflight_pure_fdm_outer_zoom
from .zoom_calibration import ZoomGrid, load_zoom_grid


PURE_FDM_OUTER_SUBMISSION_SCHEMA_VERSION = 1
WRITER_RUNTIME_ATTESTATION_SCHEMA_VERSION = 4
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_OUTPUT_DIRECTORY_RE = re.compile(r"output_[0-9]{5}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(record)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _artifact(record: Any, name: str) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} must be an artifact object")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{name}.path is required")
    path = Path(raw_path).expanduser().resolve()
    expected = _sha256(record.get("sha256"), f"{name}.sha256")
    if not path.is_file():
        raise ValueError(f"{name} is not a regular file: {path}")
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 does not match the current file")
    return path, actual


def _sidecar_relative_artifact(
    output_directory: Path, reference: str, name: str
) -> Path:
    """Resolve a sidecar-referenced file without permitting path substitution."""

    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{name} reference is required")
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} reference must be relative to the output directory")
    candidate = (output_directory / relative).resolve()
    if candidate.parent != output_directory:
        raise ValueError(f"{name} reference must name a file in the output directory")
    return candidate


def _read_text(path: Path, name: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read {name}: {error}") from error


def _namelist_assignments(path: Path) -> dict[str, list[str]]:
    """Collect simple scalar assignments after removing Fortran comments.

    The integration test only needs scalar FDM switches.  Splitting comma
    separated assignments also makes duplicate/override lines fail closed.
    """

    assignments: dict[str, list[str]] = {}
    for raw_line in _read_text(path, "copied namelist").splitlines():
        line = raw_line.split("!", 1)[0]
        for fragment in line.split(","):
            if "=" not in fragment:
                continue
            key, value = fragment.split("=", 1)
            key = key.strip().lower()
            value = value.strip().split()[0] if value.strip() else ""
            if key:
                assignments.setdefault(key, []).append(value.lower())
    return assignments


def _fortran_logical(value: str, name: str) -> bool:
    if value in {".true.", "t", "true"}:
        return True
    if value in {".false.", "f", "false"}:
        return False
    raise ValueError(f"{name} must be a Fortran logical")


def _runtime_artifact_context(
    sidecar_path: Path, provenance: Any
) -> dict[str, Path]:
    output_directory = sidecar_path.parent
    if _OUTPUT_DIRECTORY_RE.fullmatch(output_directory.name) is None:
        raise ValueError(
            "FDM writer sidecar must be directly inside an output_NNNNN directory"
        )
    namelist = _sidecar_relative_artifact(
        output_directory, provenance.namelist_copy, "namelist_copy"
    )
    compilation = _sidecar_relative_artifact(
        output_directory, provenance.compilation_copy, "compilation_copy"
    )
    complete_marker = output_directory / "COMPLETE"
    run_log = output_directory.parent / "run.log"
    output_label = output_directory.name.removeprefix("output_")
    expected_sidecar_name = f"dm_run_provenance_{output_label}.txt"
    if sidecar_path.name != expected_sidecar_name:
        raise ValueError(
            "FDM writer sidecar filename does not match its output directory"
        )
    sidecars = sorted(output_directory.glob("dm_run_provenance_*.txt"))
    if sidecars != [sidecar_path]:
        raise ValueError("output directory must contain exactly one DM sidecar")
    outer_wave_provenance = output_directory / (
        f"fdm_outer_wave_provenance_{output_label}.txt"
    )
    wave_records = sorted(output_directory.glob("fdm_outer_wave_provenance_*.txt"))
    if wave_records != [outer_wave_provenance]:
        raise ValueError(
            "output directory must contain exactly one raw FDM outer-wave provenance record"
        )
    for path, name in (
        (complete_marker, "COMPLETE marker"),
        (namelist, "copied namelist"),
        (compilation, "copied compilation"),
        (run_log, "runtime log"),
        (outer_wave_provenance, "raw FDM outer-wave provenance"),
    ):
        if not path.is_file():
            raise ValueError(f"{name} is missing: {path}")
    return {
        "complete_marker": complete_marker,
        "namelist": namelist,
        "compilation": compilation,
        "run_log": run_log,
        "outer_wave_provenance": outer_wave_provenance,
        "output_directory": output_directory,
        "run_directory": output_directory.parent,
    }


def _validate_runtime_supporting_artifacts(
    context: Mapping[str, Path], provenance: Any
) -> tuple[int, int]:
    assignments = _namelist_assignments(context["namelist"])
    for key in (
        "use_fdm",
        "fdm_outer_ledger",
        "fdm_use_hjm",
        "fdm_first_wave_level",
        "levelmax",
    ):
        values = assignments.get(key, [])
        if len(values) != 1:
            raise ValueError(f"copied namelist must contain exactly one {key} assignment")
    if not _fortran_logical(assignments["use_fdm"][0], "use_fdm"):
        raise ValueError("copied namelist does not enable use_fdm")
    if not _fortran_logical(
        assignments["fdm_outer_ledger"][0], "fdm_outer_ledger"
    ):
        raise ValueError("copied namelist does not enable fdm_outer_ledger")
    nml_hjm = _fortran_logical(assignments["fdm_use_hjm"][0], "fdm_use_hjm")
    if nml_hjm != provenance.parameter("fdm_use_hjm"):
        raise ValueError("copied namelist fdm_use_hjm disagrees with the FDM sidecar")
    try:
        first_wave_level = int(assignments["fdm_first_wave_level"][0])
    except ValueError as error:
        raise ValueError("copied namelist fdm_first_wave_level must be an integer") from error
    if first_wave_level < 0:
        raise ValueError("copied namelist fdm_first_wave_level must be non-negative")
    if first_wave_level == 0:
        raise ValueError(
            "copied namelist fdm_first_wave_level must be explicit for attestation"
        )
    try:
        levelmax = int(assignments["levelmax"][0])
    except ValueError as error:
        raise ValueError("copied namelist levelmax must be an integer") from error
    if levelmax < 1:
        raise ValueError("copied namelist levelmax must be positive")
    effective_first_wave_level = min(first_wave_level, levelmax + 1)
    try:
        sidecar_first_wave_level = int(
            provenance.parameter("fdm_first_wave_level")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("FDM sidecar fdm_first_wave_level is invalid") from error
    if sidecar_first_wave_level != effective_first_wave_level:
        raise ValueError(
            "FDM sidecar fdm_first_wave_level disagrees with the effective namelist value"
        )

    build_hash = provenance.build_git_hash.strip()
    if not build_hash:
        raise ValueError("FDM sidecar build_git_hash is required")
    compilation = _read_text(context["compilation"], "copied compilation")
    commits = re.findall(r"(?im)^\s*last\s+commit\s*=\s*(\S+)\s*$", compilation)
    if len(commits) != 1 or commits[0] != build_hash:
        raise ValueError(
            "copied compilation last commit does not match the FDM sidecar build_git_hash"
        )
    runtime_log = _read_text(context["run_log"], "runtime log")
    nproc_matches = [
        int(value)
        for value in re.findall(
            r"(?im)^\s*Working\s+with\s+nproc\s*=\s*(\d+)\s+for\s+ndim\s*=\s*3\s*$",
            runtime_log,
        )
    ]
    if (
        len(nproc_matches) < 2
        or len(set(nproc_matches)) != 1
        or nproc_matches[0] < 2
        or len(nproc_matches) != nproc_matches[0]
    ):
        raise ValueError(
            "runtime log does not attest one consistent multi-rank MPI execution"
        )
    completed = len(re.findall(r"(?im)^\s*Run\s+completed\s*$", runtime_log))
    if completed != nproc_matches[0]:
        raise ValueError("runtime log does not contain the successful completion marker")
    try:
        raw_wave = read_lagramses_fdm_outer_wave_provenance(
            context["outer_wave_provenance"]
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"raw FDM outer-wave provenance is invalid: {error}") from error
    if raw_wave.source_schema_version < 3:
        raise ValueError("raw FDM outer-wave provenance lacks MPI output-set identity")
    raw_decision = raw_wave.decision()
    if raw_decision.get("status") != "available_raw_provenance":
        raise ValueError(
            "raw FDM outer-wave provenance is not available: "
            + str(raw_decision.get("reason", "unknown reason"))
        )
    if raw_wave.nstep_coarse != provenance.nstep_coarse:
        raise ValueError("raw FDM and DM sidecars disagree on nstep_coarse")
    if not math.isclose(raw_wave.time_code, provenance.time_code, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError("raw FDM and DM sidecars disagree on time_code")
    if not math.isclose(raw_wave.aexp, provenance.scale_factor, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError("raw FDM and DM sidecars disagree on aexp")
    if not math.isclose(
        raw_wave.m_axion_ev,
        float(provenance.parameter("m_axion_ev")),
        rel_tol=1.0e-12,
        abs_tol=1.0e-30,
    ):
        raise ValueError("raw FDM and DM sidecars disagree on m_axion_ev")
    if raw_wave.fdm_use_hjm != provenance.parameter("fdm_use_hjm"):
        raise ValueError("raw FDM and DM sidecars disagree on fdm_use_hjm")
    if raw_wave.fdm_first_wave_level != effective_first_wave_level:
        raise ValueError("raw FDM provenance disagrees with the effective namelist value")
    if raw_wave.analytic_fdm_drag_enabled or raw_wave.force_accounting != "resolved_wave_only":
        raise ValueError("raw FDM provenance does not attest resolved-wave-only accounting")
    expected_prefix = f"fdm_{context['output_directory'].name.removeprefix('output_')}.out"
    if raw_wave.psi_snapshot_prefix != expected_prefix:
        raise ValueError("raw FDM provenance snapshot prefix does not match its output")
    return nproc_matches[0], effective_first_wave_level


def _validate_manifest(path: Path, grid: ZoomGrid) -> None:
    record = _read_json(path, "outer manifest")
    expected = grid.as_dict()
    if record != expected:
        raise ValueError(
            "outer manifest is not byte-identified by the requested specification"
        )


def _validate_preflight(path: Path, grid: ZoomGrid) -> None:
    record = _read_json(path, "outer preflight")
    expected = preflight_pure_fdm_outer_zoom(grid).as_dict()
    if record != expected:
        raise ValueError(
            "outer preflight does not exactly match the requested manifest"
        )


def _validate_runtime_attestation(
    path: Path,
    *,
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    """Validate the complete operator record from a writer integration test."""

    record = _read_json(path, "writer runtime attestation")
    expected_keys = {
        "schema_version",
        "status",
        "source",
        "executable",
        "fdm_sidecar",
        "complete_marker",
        "namelist",
        "compilation",
        "run_log",
        "fdm_outer_provenance",
        "execution",
    }
    if set(record) != expected_keys:
        raise ValueError("writer runtime attestation has an unsupported key set")
    if record.get("schema_version") != WRITER_RUNTIME_ATTESTATION_SCHEMA_VERSION:
        raise ValueError("unsupported writer runtime attestation schema")
    if record.get("status") != "runtime_writer_integration_passed":
        raise ValueError("writer runtime attestation is not marked passed")

    attested_source, attested_source_sha = _artifact(record.get("source"), "source")
    if attested_source != source_path or attested_source_sha != source_sha256:
        raise ValueError("writer runtime attestation source does not match the audit")
    executable_path, executable_sha = _artifact(record.get("executable"), "executable")
    sidecar_path, sidecar_sha = _artifact(record.get("fdm_sidecar"), "fdm_sidecar")
    expected = _build_runtime_attestation_record(
        attested_source,
        executable_path,
        sidecar_path,
        source_sha256=attested_source_sha,
    )
    if record != expected:
        raise ValueError(
            "writer runtime attestation supporting artifacts do not match the current run"
        )
    return expected


def _artifact_payload(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _file_sha256(path)}


def _build_runtime_attestation_record(
    source_path: Path,
    executable_path: Path,
    sidecar_path: Path,
    *,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a canonical v4 record from one completed output directory."""

    source_payload = _artifact_payload(source_path)
    if source_sha256 is not None and source_payload["sha256"] != source_sha256:
        raise ValueError("writer source hash does not match the static audit")
    executable_path = executable_path.resolve()
    sidecar_path = sidecar_path.resolve()
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise ValueError(f"executable is not an executable file: {executable_path}")
    if not sidecar_path.is_file():
        raise ValueError(f"fdm_sidecar is not a regular file: {sidecar_path}")
    try:
        provenance = read_dark_matter_run_provenance(sidecar_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"FDM writer sidecar is invalid: {error}") from error
    if provenance.dark_matter_model != "fdm":
        raise ValueError("writer runtime attestation sidecar is not an FDM output")
    if provenance.parameter("fdm_force_accounting") != "resolved_wave_only":
        raise ValueError("FDM sidecar does not declare resolved_wave_only accounting")
    if provenance.parameter("fdm_outer_ledger_enabled") is not True:
        raise ValueError("FDM sidecar does not enable the outer-wave ledger")
    context = _runtime_artifact_context(sidecar_path, provenance)
    nproc, effective_first_wave_level = _validate_runtime_supporting_artifacts(
        context, provenance
    )
    return {
        "schema_version": WRITER_RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "status": "runtime_writer_integration_passed",
        "source": source_payload,
        "executable": _artifact_payload(executable_path),
        "fdm_sidecar": _artifact_payload(sidecar_path),
        "complete_marker": _artifact_payload(context["complete_marker"]),
        "namelist": _artifact_payload(context["namelist"]),
        "compilation": _artifact_payload(context["compilation"]),
        "run_log": _artifact_payload(context["run_log"]),
        "fdm_outer_provenance": _artifact_payload(
            context["outer_wave_provenance"]
        ),
        "execution": {
            "run_directory": str(context["run_directory"]),
            "output_directory": str(context["output_directory"]),
            "output_label": context["output_directory"].name,
            "mpi_nproc": nproc,
            "effective_fdm_first_wave_level": effective_first_wave_level,
        },
    }


def build_fdm_writer_runtime_attestation(
    source: str | Path,
    executable: str | Path,
    fdm_sidecar: str | Path,
    *,
    operator_confirmed: bool = False,
) -> dict[str, Any]:
    """Record artifacts from an operator-completed FDM writer integration test.

    This function does not execute the binary.  The explicit confirmation
    flag documents that the operator has already run the compiled writer test;
    the function then verifies the current source, executable, emitted FDM
    sidecar, completion marker, copied inputs, compilation identity, and run
    log before returning the attestation record.
    """

    if operator_confirmed is not True:
        raise ValueError(
            "operator_confirmed=True is required after completing the runtime test"
        )
    source_path = Path(source).expanduser().resolve()
    executable_path = Path(executable).expanduser().resolve()
    sidecar_path = Path(fdm_sidecar).expanduser().resolve()
    audit = audit_lagramses_writer_force_accounting(
        source_path, required_models=("fdm",)
    )
    if not audit.tokens_present or audit.source_sha256 is None:
        raise ValueError(
            "FDM writer source does not pass the static token prerequisite"
        )
    return _build_runtime_attestation_record(
        source_path,
        executable_path,
        sidecar_path,
        source_sha256=audit.source_sha256,
    )


@dataclass(frozen=True)
class PureFDMOuterSubmissionPreflight:
    """A conservative decision before an operator submits the outer grid."""

    specification_path: Path
    manifest_path: Path
    manifest_sha256: str | None
    preflight_path: Path
    preflight_sha256: str | None
    writer_source_audit: WriterForceAccountingAudit | None
    runtime_attestation: dict[str, Any] | None
    status: str
    reasons: tuple[str, ...]
    outer_case_count: int
    nested_request_count: int

    @property
    def ready(self) -> bool:
        return self.status == "ready_for_operator_submission"

    def as_dict(self) -> dict[str, Any]:
        audit = (
            None if self.writer_source_audit is None else self.writer_source_audit.as_dict()
        )
        return {
            "schema_version": PURE_FDM_OUTER_SUBMISSION_SCHEMA_VERSION,
            "status": self.status,
            "interpretation": (
                "operator Slurm submission gate only; this record submits no job "
                "and accepts no kpc-to-pc delay"
            ),
            "specification": {"path": str(self.specification_path)},
            "manifest": {
                "path": str(self.manifest_path),
                "sha256": self.manifest_sha256,
            },
            "preflight": {
                "path": str(self.preflight_path),
                "sha256": self.preflight_sha256,
            },
            "writer_source_audit": audit,
            "writer_runtime_attestation": self.runtime_attestation,
            "outer_case_count": self.outer_case_count,
            "nested_request_count": self.nested_request_count,
            "execution": {
                "scheduler": "Slurm only",
                "submission": "operator-owned; this command never submits",
                "syn101": "not used",
            },
            "reasons": list(self.reasons),
        }


def assess_pure_fdm_outer_submission(
    specification: str | Path,
    manifest: str | Path,
    preflight: str | Path,
    writer_source: str | Path,
    runtime_attestation: str | Path | None = None,
) -> PureFDMOuterSubmissionPreflight:
    """Revalidate all non-submitting inputs for one pure-FDM outer campaign."""

    specification_path = Path(specification).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    preflight_path = Path(preflight).expanduser().resolve()
    writer_source_path = Path(writer_source).expanduser().resolve()
    reasons: list[str] = []
    structure_failed = False
    grid = None
    manifest_sha: str | None = None
    preflight_sha: str | None = None
    nested_count = 0
    case_count = 0
    try:
        grid = load_zoom_grid(specification_path)
        case_count = len(grid.cases)
        _validate_manifest(manifest_path, grid)
        manifest_sha = _file_sha256(manifest_path)
        expected_preflight = preflight_pure_fdm_outer_zoom(grid)
        _validate_preflight(preflight_path, grid)
        preflight_sha = _file_sha256(preflight_path)
        nested_count = len(expected_preflight.nested_requests)
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        reasons.append(str(error))
        structure_failed = True

    audit: WriterForceAccountingAudit | None = None
    runtime_record: dict[str, Any] | None = None
    try:
        audit = audit_lagramses_writer_force_accounting(
            writer_source_path, required_models=("fdm",)
        )
    except (OSError, TypeError, ValueError) as error:
        reasons.append(f"writer source audit failed: {error}")
    else:
        if not audit.tokens_present:
            reasons.extend(audit.reasons)

    if not reasons and audit is not None and audit.tokens_present:
        if runtime_attestation is None:
            reasons.append(
                "compiled writer runtime attestation is required after the static source scan"
            )
        else:
            try:
                runtime_record = _validate_runtime_attestation(
                    Path(runtime_attestation).expanduser().resolve(),
                    source_path=writer_source_path,
                    source_sha256=audit.source_sha256 or "",
                )
            except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
                reasons.append(str(error))

    if reasons:
        if structure_failed:
            status = "not_ready_outer_manifest_or_preflight"
        elif audit is None or not audit.tokens_present:
            status = "not_ready_writer_source"
        elif runtime_record is None:
            status = "not_ready_writer_runtime_attestation"
        else:
            status = "not_ready_outer_manifest_or_preflight"
    else:
        status = "ready_for_operator_submission"
    return PureFDMOuterSubmissionPreflight(
        specification_path=specification_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        preflight_path=preflight_path,
        preflight_sha256=preflight_sha,
        writer_source_audit=audit,
        runtime_attestation=runtime_record,
        status=status,
        reasons=tuple(reasons),
        outer_case_count=case_count,
        nested_request_count=nested_count,
    )


def read_verified_pure_fdm_outer_submission(
    record_path: str | Path,
    specification: str | Path,
    manifest: str | Path,
    preflight: str | Path,
    writer_source: str | Path,
    runtime_attestation: str | Path,
) -> PureFDMOuterSubmissionPreflight:
    """Rebuild and compare a saved ready decision before operator submission.

    A saved JSON file is not evidence by itself.  All source artifacts are
    re-read and hashed through :func:`assess_pure_fdm_outer_submission`, then
    the complete canonical record is compared.  Non-ready records are
    intentionally rejected by this verified reader.
    """

    saved_path = Path(record_path).expanduser().resolve()
    saved = _read_json(saved_path, "outer submission preflight")
    decision = assess_pure_fdm_outer_submission(
        specification,
        manifest,
        preflight,
        writer_source,
        runtime_attestation,
    )
    if not decision.ready:
        raise ValueError(
            "current outer submission inputs are not ready: "
            + "; ".join(decision.reasons)
        )
    if saved != decision.as_dict():
        raise ValueError("saved outer submission preflight does not match current inputs")
    return decision
