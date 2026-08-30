"""Fail-closed consumer for lagRamses normal-output physics inventories.

The inventory records which raw files and ledgers exist beside one completed
normal output.  It is deliberately not a force decomposition, a convergence
test, or a binary-delay calculation.  In particular, an unavailable ledger
is preserved as censored evidence instead of being converted into a zero
contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from .lagramses_fdm_provenance import read_lagramses_fdm_outer_wave_provenance


LAGRAMSES_RESOLVED_PHYSICS_INVENTORY_MAGIC = "# lagramses_resolved_physics_inventory_v1"
_INVENTORY_NAME = re.compile(r"resolved_physics_inventory_(\d{5})\.txt$")
_OUTPUT_DIRECTORY = re.compile(r"output_(\d{5})$")
_GROUP_DIRECTORY = re.compile(r"group_\d{5}$")
_MODELS = {"cdm", "sidm", "fdm", "none"}
_BASE_KEYS = {
    "output_number",
    "nstep_coarse",
    "time_code",
    "aexp",
    "dark_matter_model",
    "raw_snapshot_directory",
    "completion_marker",
    "star_formation_enabled",
    "stars_channel_status",
    "stars_particle_snapshot_prefix",
    "gas_channel_status",
    "gas_snapshot_prefix",
    "dark_matter_channel_status",
    "particle_snapshot_prefix",
    "potential_snapshot_prefix",
    "potential_checkpoint_status",
    "sink_info_file",
    "force_source_ledger_status",
    "force_source_ledger_reason",
    "conservation_ledger_status",
    "conservation_ledger_reason",
}
_SIDM_KEYS = {
    "sidm_scattering_ledger_status",
    "sidm_scattering_ledger_reason",
}
_FDM_KEYS = {
    "fdm_field_snapshot_status",
    "fdm_field_snapshot_prefix",
    "fdm_wave_provenance_status",
    "fdm_wave_provenance_path",
    "fdm_force_accounting",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read lagRamses resolved physics inventory: {error}") from error
    if not lines or lines[0].strip() != LAGRAMSES_RESOLVED_PHYSICS_INVENTORY_MAGIC:
        raise ValueError("unsupported lagRamses resolved physics inventory schema")
    values: dict[str, str] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"resolved physics inventory line {line_number} is not key = value")
        key, value = (item.strip() for item in stripped.split("=", 1))
        if not key or not value or key in values:
            raise ValueError(f"resolved physics inventory line {line_number} is invalid")
        values[key] = value
    return values


def _finite(value: str, label: str, *, positive: bool = False) -> float:
    try:
        parsed = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return parsed


def _nonnegative_integer(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a non-negative integer") from error
    if parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


def _logical(value: str, label: str) -> bool:
    normalized = value.lower()
    if normalized in {"t", ".true.", "true"}:
        return True
    if normalized in {"f", ".false.", "false"}:
        return False
    raise ValueError(f"{label} must be a Fortran logical")


def _status(value: str, label: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ValueError(f"{label} is unsupported")
    return value


def _basename_or_none(value: str, label: str) -> str | None:
    if value == "none":
        return None
    candidate = Path(value)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name in {"", ".", ".."}:
        raise ValueError(f"{label} must be a basename or none")
    return candidate.name


def _require_snapshot_file(directory: Path, prefix: str | None, label: str) -> None:
    if prefix is None:
        return
    if not any(directory.glob(prefix + "*")):
        raise ValueError(f"{label} does not name a file in {directory}")


def _output_and_raw_directories(source: Path, output_number: str) -> tuple[Path, Path]:
    raw_directory = source.parent
    if _GROUP_DIRECTORY.fullmatch(raw_directory.name) is not None:
        output_directory = raw_directory.parent
    else:
        output_directory = raw_directory
    match = _OUTPUT_DIRECTORY.fullmatch(output_directory.name)
    if match is None or match.group(1) != output_number:
        raise ValueError("resolved physics inventory must be inside its output_00000 directory")
    return output_directory, raw_directory


def _validate_potential_marker(
    output_directory: Path,
    *,
    nstep_coarse: int,
    time_code: float,
    aexp: float,
) -> None:
    marker = output_directory / "POISSON_PHI_VALID"
    try:
        lines = marker.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read validated potential marker: {error}") from error
    if len(lines) != 2 or lines[0].strip() != "LAGRAMSES_POISSON_PHI_VALID_V1":
        raise ValueError("validated potential marker has an unsupported schema")
    values = lines[1].split()
    if len(values) != 4:
        raise ValueError("validated potential marker has an invalid payload")
    if _nonnegative_integer(values[0], "potential marker nstep_coarse") != nstep_coarse:
        raise ValueError("validated potential marker nstep_coarse differs from inventory")
    _nonnegative_integer(values[1], "potential marker nlevelmax")
    if not math.isclose(
        _finite(values[2], "potential marker time_code"),
        time_code,
        rel_tol=1.0e-12,
        abs_tol=1.0e-14,
    ):
        raise ValueError("validated potential marker time_code differs from inventory")
    if not math.isclose(
        _finite(values[3], "potential marker aexp", positive=True),
        aexp,
        rel_tol=1.0e-12,
        abs_tol=1.0e-14,
    ):
        raise ValueError("validated potential marker aexp differs from inventory")


def _require_snapshot_status(
    status: str,
    prefix: str | None,
    *,
    label: str,
    available_statuses: set[str] = {"available"},
) -> None:
    if status in available_statuses and prefix is None:
        raise ValueError(f"available {label} requires a snapshot prefix")
    if status == "absent" and prefix is not None:
        raise ValueError(f"absent {label} cannot name a snapshot prefix")


@dataclass(frozen=True)
class LagRamsesResolvedPhysicsInventory:
    """Raw normal-output availability record for one DM realization."""

    source_path: Path
    source_sha256: str
    output_number: str
    nstep_coarse: int
    time_code: float
    aexp: float
    dark_matter_model: str
    raw_snapshot_directory: str
    star_formation_enabled: bool
    stars_channel_status: str
    stars_particle_snapshot_prefix: str | None
    gas_channel_status: str
    gas_snapshot_prefix: str | None
    dark_matter_channel_status: str
    particle_snapshot_prefix: str | None
    potential_checkpoint_status: str
    potential_snapshot_prefix: str | None
    sink_info_file: str | None
    force_source_ledger_status: str
    force_source_ledger_reason: str
    conservation_ledger_status: str
    conservation_ledger_reason: str
    sidm_scattering_ledger_status: str | None
    sidm_scattering_ledger_reason: str | None
    fdm_field_snapshot_status: str | None
    fdm_field_snapshot_prefix: str | None
    fdm_wave_provenance_status: str | None
    fdm_wave_provenance_path: str | None
    fdm_force_accounting: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "lagramses_resolved_physics_inventory",
            "interpretation": (
                "raw normal-output availability only; this does not measure a force, "
                "establish resolution or conservation, or estimate an SMBH delay"
            ),
            "source": {"path": str(self.source_path), "sha256": self.source_sha256},
            "output_number": self.output_number,
            "nstep_coarse": self.nstep_coarse,
            "time_code": self.time_code,
            "aexp": self.aexp,
            "dark_matter_model": self.dark_matter_model,
            "raw_snapshot_directory": self.raw_snapshot_directory,
            "channels": {
                "stars": {
                    "status": self.stars_channel_status,
                    "snapshot_prefix": self.stars_particle_snapshot_prefix,
                },
                "gas": {
                    "status": self.gas_channel_status,
                    "snapshot_prefix": self.gas_snapshot_prefix,
                },
                "dark_matter": {
                    "status": self.dark_matter_channel_status,
                    "snapshot_prefix": self.particle_snapshot_prefix,
                },
            },
            "potential_checkpoint_status": self.potential_checkpoint_status,
            "potential_snapshot_prefix": self.potential_snapshot_prefix,
            "sink_info_file": self.sink_info_file,
            "force_source_ledger": {
                "status": self.force_source_ledger_status,
                "reason": self.force_source_ledger_reason,
            },
            "conservation_ledger": {
                "status": self.conservation_ledger_status,
                "reason": self.conservation_ledger_reason,
            },
            "sidm_scattering_ledger": (
                None
                if self.sidm_scattering_ledger_status is None
                else {
                    "status": self.sidm_scattering_ledger_status,
                    "reason": self.sidm_scattering_ledger_reason,
                }
            ),
            "fdm": (
                None
                if self.fdm_field_snapshot_status is None
                else {
                    "field_snapshot_status": self.fdm_field_snapshot_status,
                    "field_snapshot_prefix": self.fdm_field_snapshot_prefix,
                    "wave_provenance_status": self.fdm_wave_provenance_status,
                    "wave_provenance_path": self.fdm_wave_provenance_path,
                    "force_accounting": self.fdm_force_accounting,
                }
            ),
        }


def read_lagramses_resolved_physics_inventory(
    path: str | Path,
) -> LagRamsesResolvedPhysicsInventory:
    """Read one completed-output inventory and reject malformed metadata."""

    source = Path(path).expanduser().resolve()
    filename = _INVENTORY_NAME.fullmatch(source.name)
    if filename is None:
        raise ValueError("resolved physics inventory filename is invalid")
    values = _records(source)
    model = values.get("dark_matter_model")
    if model not in _MODELS:
        raise ValueError("dark_matter_model must be cdm, sidm, fdm, or none")
    expected = _BASE_KEYS | (_SIDM_KEYS if model == "sidm" else set()) | (
        _FDM_KEYS if model == "fdm" else set()
    )
    if set(values) != expected:
        missing = sorted(expected - set(values))
        unknown = sorted(set(values) - expected)
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unknown:
            details.append(f"unknown keys: {unknown}")
        raise ValueError("resolved physics inventory fields are invalid: " + "; ".join(details))
    output_number = values["output_number"]
    if output_number != filename.group(1):
        raise ValueError("resolved physics inventory output_number differs from its filename")
    output_directory, raw_directory = _output_and_raw_directories(source, output_number)
    expected_raw_directory = (
        output_directory.name
        if raw_directory == output_directory
        else output_directory.name + "/" + raw_directory.name
    )
    if values["raw_snapshot_directory"].rstrip("/") != expected_raw_directory:
        raise ValueError("raw_snapshot_directory differs from the inventory directory")
    if values["completion_marker"] != "COMPLETE":
        raise ValueError("resolved physics inventory completion_marker is unsupported")
    try:
        completion = (output_directory / "COMPLETE").read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"cannot read output COMPLETE marker: {error}") from error
    if completion != output_number:
        raise ValueError("output COMPLETE marker differs from the inventory output number")
    nstep_coarse = _nonnegative_integer(values["nstep_coarse"], "nstep_coarse")
    time_code = _finite(values["time_code"], "time_code")
    aexp = _finite(values["aexp"], "aexp", positive=True)
    stars_status = _status(
        values["stars_channel_status"],
        "stars_channel_status",
        {"available", "absent", "requires_particle_classification"},
    )
    gas_status = _status(values["gas_channel_status"], "gas_channel_status", {"available", "absent"})
    dark_status = _status(
        values["dark_matter_channel_status"], "dark_matter_channel_status", {"available", "absent"}
    )
    potential_status = _status(
        values["potential_checkpoint_status"],
        "potential_checkpoint_status",
        {"validated", "unvalidated", "absent"},
    )
    stars_prefix = _basename_or_none(values["stars_particle_snapshot_prefix"], "stars_particle_snapshot_prefix")
    gas_prefix = _basename_or_none(values["gas_snapshot_prefix"], "gas_snapshot_prefix")
    particle_prefix = _basename_or_none(values["particle_snapshot_prefix"], "particle_snapshot_prefix")
    potential_prefix = _basename_or_none(values["potential_snapshot_prefix"], "potential_snapshot_prefix")
    _require_snapshot_status(stars_status, stars_prefix, label="stars channel")
    _require_snapshot_status(gas_status, gas_prefix, label="gas channel")
    _require_snapshot_file(raw_directory, stars_prefix, "stars snapshot prefix")
    _require_snapshot_file(raw_directory, gas_prefix, "gas snapshot prefix")
    if potential_status in {"validated", "unvalidated"} and potential_prefix is None:
        raise ValueError("non-absent potential checkpoint requires a snapshot prefix")
    if potential_status == "absent" and potential_prefix is not None:
        raise ValueError("absent potential checkpoint cannot name a snapshot prefix")
    _require_snapshot_file(raw_directory, potential_prefix, "potential snapshot prefix")
    if potential_status == "validated":
        _validate_potential_marker(
            output_directory,
            nstep_coarse=nstep_coarse,
            time_code=time_code,
            aexp=aexp,
        )
    if model == "none" and dark_status != "absent":
        raise ValueError("no-dark-matter inventory must mark its dark-matter channel absent")
    if model != "none" and dark_status != "available":
        raise ValueError("active dark-matter inventory must mark its dark-matter channel available")
    if model in {"cdm", "sidm"} and particle_prefix is None:
        raise ValueError("particle dark-matter inventory requires a particle snapshot prefix")
    _require_snapshot_file(raw_directory, particle_prefix, "particle snapshot prefix")
    sink_info_file = _basename_or_none(values["sink_info_file"], "sink_info_file")
    _require_snapshot_file(raw_directory, sink_info_file, "sink info file")
    force_status = _status(values["force_source_ledger_status"], "force_source_ledger_status", {"unavailable"})
    conservation_status = _status(
        values["conservation_ledger_status"], "conservation_ledger_status", {"unavailable"}
    )
    force_reason = values["force_source_ledger_reason"]
    conservation_reason = values["conservation_ledger_reason"]
    if not force_reason or not conservation_reason:
        raise ValueError("unavailable ledger reasons must be non-empty")

    sidm_status: str | None = None
    sidm_reason: str | None = None
    if model == "sidm":
        sidm_status = _status(
            values["sidm_scattering_ledger_status"], "sidm_scattering_ledger_status", {"unavailable"}
        )
        sidm_reason = values["sidm_scattering_ledger_reason"]
        if not sidm_reason:
            raise ValueError("SIDM scattering-ledger reason must be non-empty")

    fdm_field_status: str | None = None
    fdm_field_prefix: str | None = None
    fdm_wave_status: str | None = None
    fdm_wave_path: str | None = None
    fdm_force_accounting: str | None = None
    if model == "fdm":
        fdm_field_status = _status(
            values["fdm_field_snapshot_status"], "fdm_field_snapshot_status", {"available", "unavailable"}
        )
        fdm_field_prefix = _basename_or_none(
            values["fdm_field_snapshot_prefix"], "fdm_field_snapshot_prefix"
        )
        if fdm_field_status == "available" and fdm_field_prefix is None:
            raise ValueError("available FDM field snapshots require a snapshot prefix")
        if fdm_field_status == "unavailable" and fdm_field_prefix is not None:
            raise ValueError("unavailable FDM field snapshots cannot name a snapshot prefix")
        _require_snapshot_file(raw_directory, fdm_field_prefix, "FDM field snapshot prefix")
        fdm_wave_status = _status(
            values["fdm_wave_provenance_status"], "fdm_wave_provenance_status", {"available", "unavailable"}
        )
        wave_reference = values["fdm_wave_provenance_path"]
        if fdm_wave_status == "available":
            expected_wave = f"output_{output_number}/fdm_outer_wave_provenance_{output_number}.txt"
            if wave_reference != expected_wave:
                raise ValueError("FDM wave provenance path differs from its output number")
            wave_path = output_directory / Path(wave_reference).name
            try:
                wave = read_lagramses_fdm_outer_wave_provenance(wave_path)
            except ValueError as error:
                raise ValueError(f"cannot validate FDM wave provenance: {error}") from error
            if (
                wave.nstep_coarse != nstep_coarse
                or not math.isclose(
                    wave.time_code, time_code, rel_tol=1.0e-12, abs_tol=1.0e-14
                )
                or not math.isclose(wave.aexp, aexp, rel_tol=1.0e-12, abs_tol=1.0e-14)
                or wave.psi_snapshot_prefix != fdm_field_prefix
                or wave.force_accounting != values["fdm_force_accounting"]
            ):
                raise ValueError("FDM wave provenance differs from the resolved physics inventory")
            fdm_wave_path = wave_reference
        elif wave_reference != "none":
            raise ValueError("unavailable FDM wave provenance must use path none")
        fdm_force_accounting = values["fdm_force_accounting"]
        if fdm_force_accounting != "resolved_wave_only":
            raise ValueError("FDM inventory force accounting must be resolved_wave_only")

    return LagRamsesResolvedPhysicsInventory(
        source_path=source,
        source_sha256=_sha256(source),
        output_number=output_number,
        nstep_coarse=nstep_coarse,
        time_code=time_code,
        aexp=aexp,
        dark_matter_model=model,
        raw_snapshot_directory=values["raw_snapshot_directory"],
        star_formation_enabled=_logical(values["star_formation_enabled"], "star_formation_enabled"),
        stars_channel_status=stars_status,
        stars_particle_snapshot_prefix=stars_prefix,
        gas_channel_status=gas_status,
        gas_snapshot_prefix=gas_prefix,
        dark_matter_channel_status=dark_status,
        particle_snapshot_prefix=particle_prefix,
        potential_checkpoint_status=potential_status,
        potential_snapshot_prefix=potential_prefix,
        sink_info_file=sink_info_file,
        force_source_ledger_status=force_status,
        force_source_ledger_reason=force_reason,
        conservation_ledger_status=conservation_status,
        conservation_ledger_reason=conservation_reason,
        sidm_scattering_ledger_status=sidm_status,
        sidm_scattering_ledger_reason=sidm_reason,
        fdm_field_snapshot_status=fdm_field_status,
        fdm_field_snapshot_prefix=fdm_field_prefix,
        fdm_wave_provenance_status=fdm_wave_status,
        fdm_wave_provenance_path=fdm_wave_path,
        fdm_force_accounting=fdm_force_accounting,
    )


@dataclass(frozen=True)
class ResolvedPhysicsInventoryAssessment:
    """Fail-closed statement of whether raw evidence may enter analysis."""

    inventory: LagRamsesResolvedPhysicsInventory
    stars_required: bool
    gas_required: bool
    status: str
    reasons: tuple[str, ...]

    @property
    def ready_for_registered_analysis(self) -> bool:
        return self.status == "resolved_physics_inventory_ready"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "raw-evidence availability gate only; a ready inventory still requires "
                "measured profiles, force work, conservation, resolution, and phase evidence"
            ),
            "inventory": self.inventory.as_dict(),
            "required_baryon_channels": {
                "stars": self.stars_required,
                "gas": self.gas_required,
            },
            "reasons": list(self.reasons),
        }


def assess_lagramses_resolved_physics_inventory(
    inventory: LagRamsesResolvedPhysicsInventory,
    *,
    stars_required: bool,
    gas_required: bool,
) -> ResolvedPhysicsInventoryAssessment:
    """Require explicit channels and required raw ledgers without inventing data."""

    if not isinstance(stars_required, bool) or not isinstance(gas_required, bool):
        raise ValueError("stars_required and gas_required must be booleans")
    reasons: list[str] = []
    if inventory.sink_info_file is None:
        reasons.append("sink state file is absent")
    if inventory.potential_checkpoint_status != "validated":
        reasons.append("potential checkpoint is not validated")
    if inventory.stars_channel_status == "requires_particle_classification":
        reasons.append("stars channel requires particle classification")
    if stars_required and inventory.stars_channel_status != "available":
        reasons.append("required stars channel is not available")
    if gas_required and inventory.gas_channel_status != "available":
        reasons.append("required gas channel is not available")
    if inventory.force_source_ledger_status != "available":
        reasons.append(
            "force-source ledger is unavailable: " + inventory.force_source_ledger_reason
        )
    if inventory.conservation_ledger_status != "available":
        reasons.append(
            "conservation ledger is unavailable: " + inventory.conservation_ledger_reason
        )
    if inventory.dark_matter_model == "sidm" and inventory.sidm_scattering_ledger_status != "available":
        reasons.append(
            "SIDM scattering ledger is unavailable: "
            + str(inventory.sidm_scattering_ledger_reason)
        )
    if inventory.dark_matter_model == "fdm":
        if inventory.fdm_field_snapshot_status != "available":
            reasons.append("FDM field snapshots are unavailable")
        if inventory.fdm_wave_provenance_status != "available":
            reasons.append("FDM wave provenance is unavailable")
        if inventory.fdm_force_accounting != "resolved_wave_only":
            reasons.append("FDM force accounting is not resolved_wave_only")
    return ResolvedPhysicsInventoryAssessment(
        inventory=inventory,
        stars_required=stars_required,
        gas_required=gas_required,
        status="resolved_physics_inventory_ready" if not reasons else "censored",
        reasons=tuple(reasons),
    )
