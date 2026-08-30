"""Read and bind model-agnostic lagRamses DM run-provenance sidecars."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

from .capture_ledger import CaptureEvent


DM_RUN_PROVENANCE_MAGIC = "# dm_run_provenance_v1"
_MODELS = {"cdm", "sidm", "fdm", "none"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read DM run provenance: {error}") from error
    if not lines or lines[0].strip() != DM_RUN_PROVENANCE_MAGIC:
        raise ValueError("unsupported DM run-provenance schema")
    records: dict[str, str] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"DM run-provenance line {line_number} is not key = value")
        key, value = (item.strip() for item in line.split("=", 1))
        if not key or not value or key in records:
            raise ValueError(f"DM run-provenance line {line_number} is invalid")
        records[key] = value
    return records


def _logical(records: dict[str, str], key: str) -> bool:
    value = records.get(key, "").lower()
    if value in {"t", ".true.", "true"}:
        return True
    if value in {"f", ".false.", "false"}:
        return False
    raise ValueError(f"{key} must be a Fortran logical")


def _number(
    records: dict[str, str], key: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    try:
        value = float(records[key].replace("D", "E").replace("d", "e"))
    except (KeyError, ValueError) as error:
        raise ValueError(f"{key} must be numeric") from error
    if not math.isfinite(value) or (positive and value <= 0.0) or (nonnegative and value < 0.0):
        qualifier = "finite and positive" if positive else "finite and non-negative"
        raise ValueError(f"{key} must be {qualifier}")
    return value


def _integer(records: dict[str, str], key: str, *, nonnegative: bool = False) -> int:
    try:
        value = int(records[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{key} must be an integer") from error
    if nonnegative and value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _required(records: dict[str, str], key: str) -> str:
    value = records.get(key, "")
    if not value:
        raise ValueError(f"{key} is required")
    return value


@dataclass(frozen=True)
class DarkMatterRunProvenance:
    """One output's active CDM, SIDM, FDM, or no-DM realization."""

    source_path: Path
    source_sha256: str
    dark_matter_model: str
    nstep_coarse: int
    time_code: float
    scale_factor: float
    pic_enabled: bool
    sidm_enabled: bool
    fdm_enabled: bool
    build_git_hash: str
    namelist_copy: str
    compilation_copy: str
    smbh_capture_ledger_enabled: bool
    smbh_capture_ledger_file: str
    model_parameters: tuple[tuple[str, Any], ...]

    def parameter(self, key: str) -> Any:
        return dict(self.model_parameters).get(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": {"path": str(self.source_path), "sha256": self.source_sha256},
            "dark_matter_model": self.dark_matter_model,
            "nstep_coarse": self.nstep_coarse,
            "time_code": self.time_code,
            "aexp": self.scale_factor,
            "pic_enabled": self.pic_enabled,
            "sidm_enabled": self.sidm_enabled,
            "fdm_enabled": self.fdm_enabled,
            "build_git_hash": self.build_git_hash,
            "namelist_copy": self.namelist_copy,
            "compilation_copy": self.compilation_copy,
            "smbh_capture_ledger_enabled": self.smbh_capture_ledger_enabled,
            "smbh_capture_ledger_file": self.smbh_capture_ledger_file,
            "model_parameters": dict(self.model_parameters),
        }


def read_dark_matter_run_provenance(path: str | Path) -> DarkMatterRunProvenance:
    """Read a sidecar without interpreting the capture as a coalescence."""

    source = Path(path).expanduser().resolve()
    records = _records(source)
    model = _required(records, "dark_matter_model")
    if model not in _MODELS:
        raise ValueError("dark_matter_model must be cdm, sidm, fdm, or none")
    pic = _logical(records, "pic_enabled")
    sidm = _logical(records, "sidm_enabled")
    fdm = _logical(records, "fdm_enabled")
    parameters: dict[str, Any] = {}
    has_merge_radius = "smbh_merge_radius_cells" in records
    has_compaction_mode = "smbh_compaction_mode" in records
    if has_merge_radius != has_compaction_mode:
        raise ValueError(
            "smbh_merge_radius_cells and smbh_compaction_mode must appear together"
        )
    if has_merge_radius:
        merge_radius = _number(records, "smbh_merge_radius_cells", nonnegative=True)
        compaction_mode = _required(records, "smbh_compaction_mode")
        if compaction_mode not in {"enabled", "no_finite_radius_rmerge_zero"}:
            raise ValueError("smbh_compaction_mode is unsupported")
        if (compaction_mode == "no_finite_radius_rmerge_zero") != (merge_radius == 0.0):
            raise ValueError("smbh_compaction_mode disagrees with smbh_merge_radius_cells")
        parameters["smbh_merge_radius_cells"] = merge_radius
        parameters["smbh_compaction_mode"] = compaction_mode
    if model == "cdm":
        if not pic or sidm or fdm or records.get("dm_transport") != "collisionless_nbody":
            raise ValueError("CDM run-provenance flags are inconsistent")
        parameters["dm_transport"] = "collisionless_nbody"
    elif model == "sidm":
        if not pic or not sidm or fdm:
            raise ValueError("SIDM run-provenance flags are inconsistent")
        for key in ("sidm_cross_section_cm2_g", "sidm_v0_km_s", "sidm_power"):
            parameters[key] = _number(records, key, positive=key == "sidm_cross_section_cm2_g")
        for key in ("sidm_type", "sidm_angular"):
            parameters[key] = _required(records, key)
        parameters["sidm_inelastic"] = _logical(records, "sidm_inelastic")
        parameters["sidm_max_scatter_probability"] = _number(
            records, "sidm_max_scatter_probability", nonnegative=True
        )
    elif model == "fdm":
        if sidm or not fdm or records.get("fdm_force_accounting") != "resolved_wave_only":
            raise ValueError("FDM run-provenance flags or force accounting are inconsistent")
        parameters["m_axion_ev"] = _number(records, "m_axion_ev", positive=True)
        parameters["fdm_use_hjm"] = _logical(records, "fdm_use_hjm")
        parameters["fdm_first_wave_level"] = _integer(
            records, "fdm_first_wave_level", nonnegative=True
        )
        parameters["fdm_outer_ledger_enabled"] = _logical(
            records, "fdm_outer_ledger_enabled"
        )
        parameters["fdm_force_accounting"] = "resolved_wave_only"
    elif pic or sidm or fdm:
        raise ValueError("no-DM run-provenance flags are inconsistent")
    return DarkMatterRunProvenance(
        source_path=source,
        source_sha256=_file_sha256(source),
        dark_matter_model=model,
        nstep_coarse=_integer(records, "nstep_coarse", nonnegative=True),
        time_code=_number(records, "time_code"),
        scale_factor=_number(records, "aexp", positive=True),
        pic_enabled=pic,
        sidm_enabled=sidm,
        fdm_enabled=fdm,
        build_git_hash=records.get("build_git_hash", ""),
        namelist_copy=_required(records, "namelist_copy"),
        compilation_copy=_required(records, "compilation_copy"),
        smbh_capture_ledger_enabled=_logical(records, "smbh_capture_ledger_enabled"),
        smbh_capture_ledger_file=_required(records, "smbh_capture_ledger_file"),
        model_parameters=tuple(sorted(parameters.items())),
    )


@dataclass(frozen=True)
class CaptureDMRunBinding:
    """Provenance decision, not a dynamical comparison between DM models."""

    capture_event_uid: str
    capture_event_sha256: str
    dark_matter_model: str
    capture_ledger_path: Path
    run_provenance: DarkMatterRunProvenance
    status: str
    reasons: tuple[str, ...]

    @property
    def bound(self) -> bool:
        return self.status == "capture_dm_run_provenance_bound"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "capture-to-run provenance only; this does not declare a physical "
                "binary, compare DM-model dynamics, or estimate a delay"
            ),
            "capture_event": {
                "event_uid": self.capture_event_uid,
                "event_sha256": self.capture_event_sha256,
                "ledger_path": str(self.capture_ledger_path),
            },
            "run_provenance": self.run_provenance.as_dict(),
            "reasons": list(self.reasons),
        }


def bind_capture_event_to_dark_matter_run(
    event: CaptureEvent,
    provenance: DarkMatterRunProvenance,
) -> CaptureDMRunBinding:
    """Bind one ledger event only to a later output from the same ledger file."""

    reasons: list[str] = []
    if not provenance.smbh_capture_ledger_enabled:
        reasons.append("run provenance records SMBH capture ledger as disabled")
    ledger_reference = Path(provenance.smbh_capture_ledger_file).expanduser()
    if ledger_reference.is_absolute():
        candidate_paths = (ledger_reference.resolve(),)
    else:
        output_directory = provenance.source_path.parent
        candidate_directories = [output_directory]
        if output_directory.name.startswith("group_"):
            candidate_directories.append(output_directory.parent)
        if candidate_directories[-1].name.startswith("output_"):
            candidate_directories.append(candidate_directories[-1].parent)
        candidate_paths = tuple(
            (directory / ledger_reference).resolve() for directory in candidate_directories
        )
    if event.source_path.resolve() not in candidate_paths:
        reasons.append("run provenance capture-ledger path does not resolve to the event source")
    if provenance.nstep_coarse < event.nstep_coarse:
        reasons.append("run provenance output predates the capture event")
    return CaptureDMRunBinding(
        capture_event_uid=event.event_uid,
        capture_event_sha256=event.event_sha256,
        dark_matter_model=provenance.dark_matter_model,
        capture_ledger_path=event.source_path,
        run_provenance=provenance,
        status=("capture_dm_run_provenance_bound" if not reasons else "capture_dm_run_provenance_not_bound"),
        reasons=tuple(reasons),
    )
