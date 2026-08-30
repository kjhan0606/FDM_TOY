"""Preflight a materialized pure-FDM dual-soliton lagRamses run.

This module is deliberately limited to configuration identity and numerical
input consistency.  A ready result is *not* a relaxation, convergence, or
physical-calibration verdict, and it never submits a run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .dual_soliton_seed import (
    PureFDMDualSolitonSeed,
    pure_fdm_dual_soliton_seed_from_mapping,
)
from .lagramses_fdm_provenance import (
    LagRamsesFDMOuterWaveProvenance,
    read_lagramses_fdm_outer_wave_provenance,
)


_ASSIGNMENT = re.compile(
    r"(?:^[ \t]*|,[ \t]*)([A-Za-z][A-Za-z0-9_]*(?:\([^!\r\n=]*\))?)[ \t]*="
    r"[ \t]*(.*?)(?=,[ \t]*[A-Za-z][A-Za-z0-9_]*(?:\([^!\r\n=]*\))?"
    r"[ \t]*=|![^\r\n]*$|/[ \t]*(?:!.*)?$|\r?$)",
    re.MULTILINE,
)
_NAMELIST_GROUP = re.compile(
    r"^[ \t]*&[ \t]*(?P<name>[A-Za-z][A-Za-z0-9_]*)\b(?P<body>.*?)"
    r"(?:^[ \t]*/[ \t]*(?:!.*)?$|^[ \t]*&end\b.*$)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _logical(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"t", ".true.", "true"}:
        return True
    if normalized in {"f", ".false.", "false"}:
        return False
    raise ValueError(f"not a Fortran logical: {value!r}")


def _number(value: str) -> float:
    try:
        result = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(f"not a finite Fortran number: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"not a finite Fortran number: {value!r}")
    return result


def _assignment_values(path: Path) -> dict[str, list[dict[str, list[str]]]]:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except OSError as error:
        raise ValueError(f"cannot read run namelist: {error}") from error
    result: dict[str, list[dict[str, list[str]]]] = {}
    for group in _NAMELIST_GROUP.finditer(text):
        assignments: dict[str, list[str]] = {}
        for name, value in _ASSIGNMENT.findall(group.group("body")):
            key = re.sub(r"\s+", "", name).lower()
            assignments.setdefault(key, []).append(value.strip())
        result.setdefault(group.group("name").lower(), []).append(assignments)
    return result


def _unique_value(
    values: Mapping[str, list[Mapping[str, list[str]]]], group: str, name: str
) -> str:
    group_records = values.get(group.lower(), [])
    if not group_records:
        raise ValueError(f"missing &{group.upper()} group")
    if len(group_records) != 1:
        raise ValueError(f"&{group.upper()} group occurs more than once")
    matches = group_records[0].get(name.lower(), [])
    if not matches:
        raise ValueError(f"missing {name} in &{group.upper()}")
    if len(matches) != 1:
        raise ValueError(f"{name} is assigned more than once in &{group.upper()}")
    return matches[0]


def read_lagramses_namelist_assignment(
    path: str | Path, *, group: str, name: str
) -> str:
    """Read one exact assignment from the solver-consumed namelist group."""

    return _unique_value(
        _assignment_values(Path(path).expanduser().resolve()), group, name
    )


def _array_override_reasons(
    values: Mapping[str, list[Mapping[str, list[str]]]],
    *,
    group: str,
    expected_names: set[str],
) -> list[str]:
    """Reject whole-array/section aliases of elementwise checked FDM inputs."""

    group_records = values.get(group.lower(), [])
    if len(group_records) != 1:
        return []
    roots = {name.split("(", 1)[0] for name in expected_names}
    reasons: list[str] = []
    for name in group_records[0]:
        if name.split("(", 1)[0] in roots and name not in expected_names:
            reasons.append(
                f"{name} in &{group.upper()} aliases an elementwise checked FDM array"
            )
    return reasons


def _sink_rows(path: Path) -> np.ndarray:
    try:
        rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as error:
        raise ValueError(f"cannot read ic_sink: {error}") from error
    if len(rows) != 2 or any(len(row) != 12 for row in rows):
        raise ValueError("ic_sink must contain exactly two data-only rows of twelve fields")
    try:
        return np.asarray([[_number(value) for value in row] for row in rows], dtype=float)
    except ValueError as error:
        raise ValueError(f"ic_sink is invalid: {error}") from error


def _expected_sink_rows(seed: PureFDMDualSolitonSeed) -> np.ndarray:
    rows: list[list[float]] = []
    for sink in seed.sinks:
        offset = [seed.box_length_code * (value - 0.5) for value in sink.center_box]
        rows.append(
            [
                sink.sink_mass_code,
                *offset,
                *sink.velocity_code,
                *sink.angular_momentum_code,
                sink.smbh_mass_code,
                sink.dark_matter_fraction,
            ]
        )
    return np.asarray(rows, dtype=float)


def _manifest_seed(manifest_path: Path) -> tuple[PureFDMDualSolitonSeed, dict[str, Any]]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read seed manifest: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported dual-soliton seed manifest schema")
    if payload.get("status") != "pure_fdm_dual_soliton_seed_materialized":
        raise ValueError("dual-soliton seed manifest does not record materialization")
    seed_record = payload.get("seed")
    if not isinstance(seed_record, dict):
        raise ValueError("dual-soliton seed manifest lacks the seed record")
    canonical_seed = dict(seed_record)
    policy = canonical_seed.pop("solver_policy", None)
    if policy != {
        "fdm_use_hjm": False,
        "fdm_outer_ledger": True,
        "analytic_fdm_drag": False,
        "initial_state": "coherent_dual_soliton_seed_requires_relaxation",
    }:
        raise ValueError("dual-soliton seed manifest has an invalid solver policy")
    seed = pure_fdm_dual_soliton_seed_from_mapping(canonical_seed)
    if payload.get("seed_sha256") != _canonical_sha256(seed_record):
        raise ValueError("dual-soliton seed manifest seed hash does not match")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("dual-soliton seed manifest lacks file provenance")
    for key in ("fdm_namelist_fragment", "ic_sink"):
        source = files.get(key)
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ValueError(f"dual-soliton seed manifest lacks {key} provenance")
        source_path = Path(source["path"]).expanduser().resolve()
        expected_hash = source.get("sha256")
        if not isinstance(expected_hash, str) or _sha256(source_path) != expected_hash:
            raise ValueError(f"materialized {key} hash does not match the manifest")
    return seed, payload


@dataclass(frozen=True)
class DualSolitonRunPreflight:
    """Configuration-only decision for a manual or Slurm operator submission."""

    seed_manifest_path: Path
    run_namelist_path: Path
    run_ic_sink_path: Path
    seed_case_id: str
    status: str
    reasons: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.status == "ready_for_operator_submission"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "configuration identity only; this does not certify relaxation, "
                "conservation, resolution, phase replication, or physical calibration"
            ),
            "seed_case_id": self.seed_case_id,
            "sources": {
                "seed_manifest": {
                    "path": str(self.seed_manifest_path),
                    "sha256": _sha256(self.seed_manifest_path),
                },
                "run_namelist": {
                    "path": str(self.run_namelist_path),
                    "sha256": _sha256(self.run_namelist_path),
                },
                "run_ic_sink": {
                    "path": str(self.run_ic_sink_path),
                    "sha256": _sha256(self.run_ic_sink_path),
                },
            },
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DualSolitonRuntimeIdentity:
    """Binding between a materialized seed and one raw V2/V3 solver output."""

    seed_manifest_path: Path
    provenance_path: Path
    seed_case_id: str
    status: str
    reasons: tuple[str, ...]
    provenance_decision: dict[str, Any]

    @property
    def verified(self) -> bool:
        return self.status == "runtime_seed_identity_verified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "runtime configuration identity only; this does not certify "
                "relaxation, conservation, resolution, phase replication, or "
                "physical calibration"
            ),
            "seed_case_id": self.seed_case_id,
            "sources": {
                "seed_manifest": {
                    "path": str(self.seed_manifest_path),
                    "sha256": _sha256(self.seed_manifest_path),
                },
                "raw_fdm_provenance": {
                    "path": str(self.provenance_path),
                    "sha256": _sha256(self.provenance_path),
                },
            },
            "raw_provenance_decision": self.provenance_decision,
            "reasons": list(self.reasons),
        }


def read_verified_pure_fdm_dual_soliton_run_preflight(
    path: str | Path,
) -> DualSolitonRunPreflight:
    """Re-run a saved ready preflight from the exact current input files."""

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dual-soliton run preflight: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "interpretation",
        "seed_case_id",
        "sources",
        "reasons",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_fields
        or record.get("schema_version") != 1
        or record.get("status") != "ready_for_operator_submission"
        or record.get("reasons") != []
    ):
        raise ValueError("dual-soliton run preflight is not ready")
    sources = record.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "seed_manifest",
        "run_namelist",
        "run_ic_sink",
    }:
        raise ValueError("dual-soliton run preflight sources are invalid")
    paths: dict[str, Path] = {}
    for name in ("seed_manifest", "run_namelist", "run_ic_sink"):
        artifact = sources[name]
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "sha256"}
            or not isinstance(artifact.get("path"), str)
            or not isinstance(artifact.get("sha256"), str)
        ):
            raise ValueError(f"dual-soliton run preflight {name} source is invalid")
        candidate = Path(artifact["path"]).expanduser().resolve()
        try:
            actual = _sha256(candidate)
        except OSError as error:
            raise ValueError(f"cannot re-read dual-soliton preflight {name}: {error}") from error
        if actual != artifact["sha256"]:
            raise ValueError(f"dual-soliton preflight {name} SHA-256 no longer matches")
        paths[name] = candidate
    decision = preflight_pure_fdm_dual_soliton_run(
        seed_manifest_path=paths["seed_manifest"],
        run_namelist_path=paths["run_namelist"],
        run_ic_sink_path=paths["run_ic_sink"],
    )
    if not decision.ready or decision.as_dict() != record:
        raise ValueError("dual-soliton run preflight no longer matches its source inputs")
    return decision


def read_verified_pure_fdm_dual_soliton_runtime_identity(
    path: str | Path,
) -> DualSolitonRuntimeIdentity:
    """Rebuild one saved verified runtime identity from its current sources.

    The compact raw provenance is intentionally only a configuration and
    force/current-coverage record.  This reader consequently does not turn it
    into a relaxation, convergence, or physical-delay result; it only prevents
    a downstream consumer from trusting a stale or hand-written status field.
    """

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dual-soliton runtime identity: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "interpretation",
        "seed_case_id",
        "sources",
        "raw_provenance_decision",
        "reasons",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_fields
        or record.get("schema_version") != 1
        or record.get("status") != "runtime_seed_identity_verified"
        or record.get("reasons") != []
    ):
        raise ValueError("dual-soliton runtime identity is not verified")
    sources = record.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "seed_manifest",
        "raw_fdm_provenance",
    }:
        raise ValueError("dual-soliton runtime identity sources are invalid")
    paths: dict[str, Path] = {}
    for name in ("seed_manifest", "raw_fdm_provenance"):
        artifact = sources[name]
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "sha256"}
            or not isinstance(artifact.get("path"), str)
            or not isinstance(artifact.get("sha256"), str)
        ):
            raise ValueError(f"dual-soliton runtime identity {name} source is invalid")
        candidate = Path(artifact["path"]).expanduser().resolve()
        try:
            actual = _sha256(candidate)
        except OSError as error:
            raise ValueError(
                f"cannot re-read dual-soliton runtime identity {name}: {error}"
            ) from error
        if actual != artifact["sha256"]:
            raise ValueError(
                f"dual-soliton runtime identity {name} SHA-256 no longer matches"
            )
        paths[name] = candidate
    decision = validate_pure_fdm_dual_soliton_runtime_identity(
        seed_manifest_path=paths["seed_manifest"],
        provenance_path=paths["raw_fdm_provenance"],
    )
    if not decision.verified or decision.as_dict() != record:
        raise ValueError("dual-soliton runtime identity no longer matches its source inputs")
    return decision


def preflight_pure_fdm_dual_soliton_run(
    *,
    seed_manifest_path: str | Path,
    run_namelist_path: str | Path,
    run_ic_sink_path: str | Path,
) -> DualSolitonRunPreflight:
    """Verify an exact seed is faithfully represented in a complete run input.

    A configuration mismatch is reported as ``not_ready_for_operator_submission``
    rather than being treated as a scientific censoring decision.
    """

    manifest = Path(seed_manifest_path).expanduser().resolve()
    namelist = Path(run_namelist_path).expanduser().resolve()
    sink_path = Path(run_ic_sink_path).expanduser().resolve()
    seed, manifest_record = _manifest_seed(manifest)
    values = _assignment_values(namelist)
    reasons: list[str] = []

    expected_logicals = {
        ("physics_params", "smbh"): True,
        ("physics_params", "smbh_capture_ledger"): True,
        ("run_params", "use_fdm"): True,
        ("run_params", "poisson"): True,
        ("run_params", "sink"): True,
        ("run_params", "hydro"): seed.gas_status == "available",
        ("fdm_params", "fdm_dual_soliton_ic"): True,
        ("fdm_params", "fdm_use_hjm"): False,
        ("fdm_params", "fdm_outer_ledger"): True,
    }
    for (group, name), expected in expected_logicals.items():
        try:
            actual = _logical(_unique_value(values, group, name))
            if actual != expected:
                reasons.append(
                    f"{name} in &{group.upper()} must be {str(expected).lower()}"
                )
        except ValueError as error:
            reasons.append(str(error))

    expected_numbers: dict[tuple[str, str], float] = {
        ("physics_params", "rmerge"): 0.0,
        ("amr_params", "boxlen"): seed.box_length_code,
        ("fdm_params", "m_axion"): seed.m_axion_ev,
        ("fdm_params", "fdm_dual_soliton_profile_c"): seed.profile_c,
    }
    for index, soliton in enumerate(seed.solitons, start=1):
        expected_numbers[("fdm_params", f"fdm_dual_soliton_rho0({index})")] = soliton.rho0_code
        expected_numbers[("fdm_params", f"fdm_dual_soliton_rc_box({index})")] = soliton.core_radius_box
        expected_numbers[("fdm_params", f"fdm_dual_soliton_phase({index})")] = soliton.phase_radians
        for dimension in range(1, 4):
            expected_numbers[("fdm_params", f"fdm_dual_soliton_center_box({index},{dimension})")] = soliton.center_box[dimension - 1]
            expected_numbers[("fdm_params", f"fdm_dual_soliton_velocity({index},{dimension})")] = soliton.velocity_code[dimension - 1]
    for (group, name), expected in expected_numbers.items():
        try:
            actual = _number(_unique_value(values, group, name))
            if group == "physics_params" and name == "rmerge":
                if actual != 0.0:
                    reasons.append("rmerge in &PHYSICS_PARAMS must be exactly 0 for non-compacting FDM")
            elif not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-14):
                reasons.append(
                    f"{name} in &{group.upper()} does not match the materialized seed"
                )
        except ValueError as error:
            reasons.append(str(error))
    reasons.extend(
        _array_override_reasons(
            values,
            group="fdm_params",
            expected_names={
                name for group, name in expected_numbers if group == "fdm_params" and "(" in name
            },
        )
    )

    try:
        actual_rows = _sink_rows(sink_path)
        if not np.allclose(actual_rows, _expected_sink_rows(seed), rtol=1.0e-12, atol=1.0e-14):
            reasons.append("run ic_sink does not exactly match the materialized two-SMBH seed")
        manifest_sink = manifest_record["files"]["ic_sink"]
        if _sha256(sink_path) != manifest_sink["sha256"]:
            reasons.append("run ic_sink SHA-256 does not match the materialized seed")
    except ValueError as error:
        reasons.append(str(error))

    return DualSolitonRunPreflight(
        seed_manifest_path=manifest,
        run_namelist_path=namelist,
        run_ic_sink_path=sink_path,
        seed_case_id=seed.case_id,
        status=(
            "ready_for_operator_submission"
            if not reasons
            else "not_ready_for_operator_submission"
        ),
        reasons=tuple(reasons),
    )


def _same_number(actual: float | None, expected: float) -> bool:
    return actual is not None and math.isclose(
        actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-14
    )


def _same_vector(
    actual: tuple[float, ...] | None, expected: tuple[float, ...]
) -> bool:
    return actual is not None and len(actual) == len(expected) and all(
        _same_number(value, reference) for value, reference in zip(actual, expected)
    )


def _validate_runtime_components(
    provenance: LagRamsesFDMOuterWaveProvenance,
    seed: PureFDMDualSolitonSeed,
) -> list[str]:
    reasons: list[str] = []
    if provenance.source_schema_version not in {2, 3}:
        return ["raw provenance V2 or V3 is required for dual-soliton runtime identity"]
    if provenance.fdm_dual_soliton_ic is not True:
        reasons.append("raw provenance does not confirm fdm_dual_soliton_ic=true")
    if provenance.fdm_use_hjm:
        reasons.append("raw provenance reports an HJM seam for a coherent dual-soliton seed")
    if not _same_number(provenance.m_axion_ev, seed.m_axion_ev):
        reasons.append("raw provenance m_axion_ev does not match the materialized seed")
    if not _same_number(provenance.fdm_dual_soliton_profile_c, seed.profile_c):
        reasons.append("raw provenance dual-soliton profile_c does not match the seed")
    if not _same_vector(
        provenance.fdm_dual_soliton_rho0,
        tuple(component.rho0_code for component in seed.solitons),
    ):
        reasons.append("raw provenance dual-soliton rho0 does not match the seed")
    if not _same_vector(
        provenance.fdm_dual_soliton_rc_box,
        tuple(component.core_radius_box for component in seed.solitons),
    ):
        reasons.append("raw provenance dual-soliton core radii do not match the seed")
    if provenance.fdm_dual_soliton_centres_box is None or len(
        provenance.fdm_dual_soliton_centres_box
    ) != 2:
        reasons.append("raw provenance dual-soliton centres are absent")
    else:
        for index, (actual, component) in enumerate(
            zip(provenance.fdm_dual_soliton_centres_box, seed.solitons), start=1
        ):
            if not _same_vector(actual, component.center_box):
                reasons.append(f"raw provenance soliton {index} centre does not match the seed")
    if provenance.fdm_dual_soliton_velocities is None or len(
        provenance.fdm_dual_soliton_velocities
    ) != 2:
        reasons.append("raw provenance dual-soliton velocities are absent")
    else:
        for index, (actual, component) in enumerate(
            zip(provenance.fdm_dual_soliton_velocities, seed.solitons), start=1
        ):
            if not _same_vector(actual, component.velocity_code):
                reasons.append(f"raw provenance soliton {index} velocity does not match the seed")
    if not _same_vector(
        provenance.fdm_dual_soliton_phase,
        tuple(component.phase_radians for component in seed.solitons),
    ):
        reasons.append("raw provenance dual-soliton phases do not match the seed")
    return reasons


def validate_pure_fdm_dual_soliton_runtime_identity(
    *, seed_manifest_path: str | Path, provenance_path: str | Path
) -> DualSolitonRuntimeIdentity:
    """Verify the raw V2/V3 output retained the exact materialized two-core state."""

    manifest = Path(seed_manifest_path).expanduser().resolve()
    raw_provenance = Path(provenance_path).expanduser().resolve()
    seed, _ = _manifest_seed(manifest)
    provenance = read_lagramses_fdm_outer_wave_provenance(raw_provenance)
    raw_decision = provenance.decision()
    reasons = _validate_runtime_components(provenance, seed)
    if raw_decision["status"] != "available_raw_provenance":
        reasons.append("raw FDM provenance does not pass its force/current coverage gate")
    return DualSolitonRuntimeIdentity(
        seed_manifest_path=manifest,
        provenance_path=raw_provenance,
        seed_case_id=seed.case_id,
        status=(
            "runtime_seed_identity_verified"
            if not reasons
            else "runtime_seed_identity_not_verified"
        ),
        reasons=tuple(reasons),
        provenance_decision=raw_decision,
    )
