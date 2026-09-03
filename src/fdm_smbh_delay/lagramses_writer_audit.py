"""Static, non-submitting audit of the lagRamses force-accounting writer.

The audit is deliberately weaker than a runtime provenance check.  It only
answers whether the selected source file contains the model-specific tokens
and nearby field names that a writer integration test must exercise.  A source
scan never attests that a compiled executable emitted those values or that the
values were derived from the active runtime controls.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


AUDIT_SCHEMA_VERSION = 1

_MODEL_REQUIREMENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cdm": (
        "force_accounting = resolved_collisionless_only",
        ("dm_transport", "collisionless"),
    ),
    "sidm": (
        "force_accounting = resolved_collisionless_plus_scattering",
        ("sidm_max_scatter_probability", "sidm_Pmax"),
    ),
    "fdm": (
        "fdm_force_accounting = resolved_wave_only",
        ("fdm_outer_ledger_enabled", "fdm_first_wave_level", "fdm_use_hjm"),
    ),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fortran_code(text: str) -> str:
    """Remove Fortran comments while preserving quoted string contents."""

    lines: list[str] = []
    for line in text.splitlines():
        quoted: str | None = None
        code: list[str] = []
        index = 0
        while index < len(line):
            character = line[index]
            if quoted is None:
                if character in {"'", '"'}:
                    quoted = character
                    code.append(character)
                elif character == "!":
                    break
                else:
                    code.append(character)
            else:
                code.append(character)
                if character == quoted:
                    # Fortran escapes a quote in a string by doubling it.
                    if index + 1 < len(line) and line[index + 1] == quoted:
                        code.append(line[index + 1])
                        index += 1
                    else:
                        quoted = None
            index += 1
        lines.append("".join(code))
    return "\n".join(lines)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


@dataclass(frozen=True)
class WriterForceAccountingAudit:
    """Result of a source-token scan, never a runtime readiness decision."""

    source_path: Path
    source_sha256: str | None
    status: str
    models: dict[str, dict[str, Any]]
    reasons: tuple[str, ...]

    @property
    def tokens_present(self) -> bool:
        return self.status == "source_tokens_present_requires_runtime_test"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": self.status,
            "interpretation": (
                "static source-token scan only; this does not attest a compiled "
                "executable, runtime-derived values, or emitted sidecars"
            ),
            "source": {
                "path": str(self.source_path),
                "sha256": self.source_sha256,
            },
            "models": self.models,
            "reasons": list(self.reasons),
        }


def _model_scan(code: str, token: str, evidence: tuple[str, ...]) -> dict[str, Any]:
    compact = _compact(code)
    token_compact = _compact(token)
    found_evidence = {name: name.lower() in compact for name in evidence}
    return {
        "token": token,
        "token_present": token_compact in compact,
        "evidence": found_evidence,
        "evidence_complete": all(found_evidence.values()),
    }


def audit_lagramses_writer_force_accounting(
    source: str | Path,
    *,
    required_models: tuple[str, ...] = ("cdm", "sidm"),
) -> WriterForceAccountingAudit:
    """Scan one writer source file without compiling or executing it."""

    requested = tuple(dict.fromkeys(required_models))
    if not requested or any(model not in _MODEL_REQUIREMENTS for model in requested):
        raise ValueError(
            "required_models must contain one or more of cdm, sidm, or fdm"
        )

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        return WriterForceAccountingAudit(
            source_path=source_path,
            source_sha256=None,
            status="source_unavailable",
            models={},
            reasons=("writer source file is missing or is not a regular file",),
        )
    try:
        raw = source_path.read_text(encoding="utf-8")
        source_sha256 = _file_sha256(source_path)
    except (OSError, UnicodeError) as error:
        return WriterForceAccountingAudit(
            source_path=source_path,
            source_sha256=None,
            status="source_unavailable",
            models={},
            reasons=(f"writer source cannot be read: {error}",),
        )

    code = _fortran_code(raw)
    models = {
        model: _model_scan(code, *_MODEL_REQUIREMENTS[model])
        for model in requested
    }
    reasons: list[str] = []
    for model, result in models.items():
        if not result["token_present"]:
            reasons.append(f"{model} force_accounting token is absent from the writer source")
        if not result["evidence_complete"]:
            missing = [name for name, present in result["evidence"].items() if not present]
            reasons.append(f"{model} writer evidence fields are absent: {', '.join(missing)}")
    status = (
        "source_tokens_present_requires_runtime_test"
        if not reasons
        else "source_token_missing"
    )
    return WriterForceAccountingAudit(
        source_path=source_path,
        source_sha256=source_sha256,
        status=status,
        models=models,
        reasons=tuple(reasons),
    )


def write_writer_force_accounting_audit(path: str | Path, audit: WriterForceAccountingAudit) -> None:
    """Write an audit record atomically."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(audit.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
