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
from pathlib import Path
import re
from typing import Any, Mapping

from .dm_run_provenance import read_dark_matter_run_provenance
from .lagramses_writer_audit import (
    WriterForceAccountingAudit,
    audit_lagramses_writer_force_accounting,
)
from .pure_fdm_zoom import preflight_pure_fdm_outer_zoom
from .zoom_calibration import ZoomGrid, load_zoom_grid


PURE_FDM_OUTER_SUBMISSION_SCHEMA_VERSION = 1
WRITER_RUNTIME_ATTESTATION_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


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
    """Validate a small operator record from a compiled writer integration test.

    The record is intentionally not a cryptographic proof of execution.  It
    binds the test's source and executable bytes and re-reads the emitted FDM
    sidecar, which is enough to prevent a stale or model-mismatched fixture
    from being used as a submission prerequisite.
    """

    record = _read_json(path, "writer runtime attestation")
    expected_keys = {"schema_version", "status", "source", "executable", "fdm_sidecar"}
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
    return {
        "schema_version": WRITER_RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "status": "runtime_writer_integration_passed",
        "source": {"path": str(attested_source), "sha256": attested_source_sha},
        "executable": {"path": str(executable_path), "sha256": executable_sha},
        "fdm_sidecar": {"path": str(sidecar_path), "sha256": sidecar_sha},
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
    the function then verifies the current source, executable, and emitted FDM
    sidecar bytes before returning the attestation record.
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
    attested_source, source_sha = _artifact(
        {"path": str(source_path), "sha256": audit.source_sha256}, "source"
    )
    attested_executable, executable_sha = _artifact(
        {"path": str(executable_path), "sha256": _file_sha256(executable_path)},
        "executable",
    )
    attested_sidecar, sidecar_sha = _artifact(
        {"path": str(sidecar_path), "sha256": _file_sha256(sidecar_path)},
        "fdm_sidecar",
    )
    try:
        provenance = read_dark_matter_run_provenance(attested_sidecar)
    except (OSError, ValueError) as error:
        raise ValueError(f"FDM writer sidecar is invalid: {error}") from error
    if provenance.dark_matter_model != "fdm":
        raise ValueError("writer runtime attestation sidecar is not an FDM output")
    if provenance.parameter("fdm_force_accounting") != "resolved_wave_only":
        raise ValueError("FDM sidecar does not declare resolved_wave_only accounting")
    if provenance.parameter("fdm_outer_ledger_enabled") is not True:
        raise ValueError("FDM sidecar does not enable the outer-wave ledger")
    return {
        "schema_version": WRITER_RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "status": "runtime_writer_integration_passed",
        "source": {"path": str(attested_source), "sha256": source_sha},
        "executable": {"path": str(attested_executable), "sha256": executable_sha},
        "fdm_sidecar": {"path": str(attested_sidecar), "sha256": sidecar_sha},
    }


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
