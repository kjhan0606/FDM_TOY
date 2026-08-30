"""Materialize a provenance-bound, non-submitting CDM zoom run contract.

The contract ties exactly one preflighted resolution/phase case to an original
CDM binary-capture event and to an already prepared lagRamses namelist.  It
does not edit that namelist, submit a job, or turn numerical capture into an
SMBH coalescence prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from .cdm_zoom_plan import CDMNonCompactingZoomPlan, load_cdm_noncompacting_zoom_plan
from .lagramses_cdm_orbit import read_bound_cdm_capture
from .zoom_calibration import GalaxyMergerZoomCase


CDM_NONCOMPACTING_ZOOM_RUN_CONTRACT_SCHEMA_VERSION = 1
_BUILD_GIT_HASH = re.compile(r"[0-9a-f]{40}")
_REQUIRED_CASE_INPUT_ARTIFACTS = (
    "host_orbit_initial_conditions",
    "initial_conditions",
    "sink_initial_conditions",
)
_ASSIGNMENT = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z0-9_]*)[ \t]*=[ \t]*([^!,/\r\n]+)",
    re.MULTILINE,
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


def _read_namelist_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read lagRamses run namelist: {error}") from error


def _assignments(source: str) -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = {}
    for name, value in _ASSIGNMENT.findall(source):
        assignments.setdefault(name.lower(), []).append(value.strip())
    return assignments


def _physics_params_assignments(source: str) -> dict[str, list[str]]:
    """Read controls consumed by lagRamses ``read_hydro_params``.

    The materialized fragment belongs in ``&PHYSICS_PARAMS``.  Counting a
    same-named setting from a different namelist group would only attest text
    that the solver does not consume, so reject that otherwise ambiguous form.
    """

    groups = list(_PHYSICS_PARAMS_GROUP.finditer(source))
    if len(groups) != 1:
        raise ValueError("namelist must contain exactly one &PHYSICS_PARAMS group")
    return _assignments(groups[0].group("body"))


def _unique(assignments: Mapping[str, list[str]], name: str) -> str:
    values = assignments.get(name, [])
    if not values:
        raise ValueError(f"missing {name}")
    if len(values) != 1:
        raise ValueError(f"{name} is assigned more than once")
    return values[0]


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


def _fortran_string(value: str) -> str:
    normalized = value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1]
    if not normalized:
        raise ValueError("smbh_capture_ledger_file must be non-empty")
    return normalized


def _case(plan: CDMNonCompactingZoomPlan, case_id: str) -> GalaxyMergerZoomCase:
    matches = [case for case in plan.grid.cases if case.case_id == case_id]
    if len(matches) != 1:
        raise ValueError("case_id does not identify exactly one CDM zoom manifest case")
    return matches[0]


def _required_smbh_controls(ledger_file: str, execution_identity: Mapping[str, str]) -> str:
    escaped = ledger_file.replace("'", "''")
    return "\n".join(
        (
            "! Generated CDM non-compacting zoom controls. Do not append this group",
            "! to a namelist that already defines the same controls; its exact values",
            "! are verified separately against the operator's complete namelist.",
            "&PHYSICS_PARAMS",
            "smbh=.true.",
            "rmerge=0.0d0",
            "smbh_capture_ledger=.true.",
            f"smbh_capture_ledger_file='{escaped}'",
            *(f"{name}='{execution_identity[name]}'" for name in sorted(execution_identity)),
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


def _source(path: Path) -> dict[str, str | None]:
    """Preserve a missing namelist as an auditable not-ready condition."""

    try:
        digest = _sha256(path)
    except OSError:
        digest = None
    return {"path": str(path), "sha256": digest}


def _artifact(path: str | Path, label: str) -> dict[str, str]:
    source = Path(path).expanduser().resolve()
    try:
        digest = _sha256(source)
    except OSError as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    return {"path": str(source), "sha256": digest}


def _case_input_artifacts(paths: Mapping[str, str | Path]) -> dict[str, dict[str, str]]:
    if set(paths) != set(_REQUIRED_CASE_INPUT_ARTIFACTS):
        raise ValueError(
            "case input artifacts must name host_orbit_initial_conditions, "
            "initial_conditions, and sink_initial_conditions"
        )
    return {name: _artifact(paths[name], f"case input artifact {name}") for name in sorted(paths)}


def _expected_build_hash(value: str) -> str:
    normalized = value.strip().lower()
    if _BUILD_GIT_HASH.fullmatch(normalized) is None:
        raise ValueError("expected_build_git_hash must be a 40-character Git SHA-1")
    return normalized


def _execution_identity(
    plan: CDMNonCompactingZoomPlan,
    capture_binding: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    capture_sha256 = capture_binding.get("capture_event_sha256")
    if not isinstance(capture_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", capture_sha256) is None:
        raise ValueError("capture binding does not provide a lowercase event SHA-256")
    return {
        "cdm_zoom_plan_manifest_sha256": plan.grid.manifest_sha256,
        "cdm_zoom_capture_event_sha256": capture_sha256,
        "cdm_zoom_host_orbit_initial_conditions_sha256": artifacts[
            "host_orbit_initial_conditions"
        ]["sha256"],
        "cdm_zoom_initial_conditions_sha256": artifacts["initial_conditions"]["sha256"],
        "cdm_zoom_sink_initial_conditions_sha256": artifacts[
            "sink_initial_conditions"
        ]["sha256"],
    }


@dataclass(frozen=True)
class CDMNonCompactingZoomRunContract:
    """Exact non-submitting input identity for one CDM zoom realization."""

    specification_path: Path
    run_namelist_path: Path
    plan: CDMNonCompactingZoomPlan
    case: GalaxyMergerZoomCase
    capture_binding: Mapping[str, Any]
    capture_ledger_file: str
    expected_build_git_hash: str
    expected_compilation: Mapping[str, str]
    case_input_artifacts: Mapping[str, Mapping[str, str]]
    execution_identity: Mapping[str, str]
    status: str
    reasons: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.status == "ready_for_operator_submission"

    def as_dict(self, *, controls_fragment_path: Path) -> dict[str, Any]:
        return {
            "schema_version": CDM_NONCOMPACTING_ZOOM_RUN_CONTRACT_SCHEMA_VERSION,
            "status": self.status,
            "interpretation": (
                "input identity and non-compaction requirements only; this does not "
                "submit a run, certify force/conservation/resolution adequacy, or "
                "estimate a physical SMBH coalescence time"
            ),
            "dark_matter_model": "cdm",
            "case": self.case.as_dict(),
            "plan": {
                "path": str(self.specification_path),
                "sha256": _sha256(self.specification_path),
                "manifest_sha256": self.plan.grid.manifest_sha256,
            },
            "capture_binding": dict(self.capture_binding),
            "case_input_identity": {
                "expected_build_git_hash": self.expected_build_git_hash,
                "expected_compilation": dict(self.expected_compilation),
                "input_artifacts": {
                    name: dict(artifact)
                    for name, artifact in self.case_input_artifacts.items()
                },
                "execution_identity": dict(self.execution_identity),
            },
            "run_inputs": {
                "namelist": _source(self.run_namelist_path),
                "required_smbh_controls_fragment": {
                    "path": str(controls_fragment_path),
                    "sha256": _sha256(controls_fragment_path),
                },
                "smbh_capture_ledger_file": self.capture_ledger_file,
            },
            "sampling_requirements": {
                "target_separation_pc": self.plan.target_separation_pc,
                "maximum_output_cadence_myr": self.plan.maximum_output_cadence_myr,
                "samples_per_secular_block": self.plan.samples_per_secular_block,
                "minimum_secular_blocks": self.plan.minimum_secular_blocks,
                "minimum_complete_outputs": self.plan.minimum_complete_outputs,
                "position_coordinate": self.plan.position_coordinate,
                "time_coordinate": self.plan.time_coordinate,
            },
            "execution": "operator Slurm submission only after status is ready",
            "reasons": list(self.reasons),
        }


def assess_cdm_noncompacting_zoom_run_inputs(
    *,
    specification_path: str | Path,
    case_id: str,
    capture_binding_path: str | Path,
    capture_event_uid: str,
    primary_sink_id: int,
    secondary_sink_id: int,
    run_namelist_path: str | Path,
    capture_ledger_file: str,
    expected_build_git_hash: str,
    expected_compilation_path: str | Path,
    case_input_artifact_paths: Mapping[str, str | Path],
) -> tuple[CDMNonCompactingZoomRunContract, CDMNonCompactingZoomPlan]:
    """Check one complete lagRamses input without modifying it or submitting."""

    specification = Path(specification_path).expanduser().resolve()
    namelist = Path(run_namelist_path).expanduser().resolve()
    plan = load_cdm_noncompacting_zoom_plan(specification)
    selected_case = _case(plan, case_id)
    binding = read_bound_cdm_capture(
        capture_binding_path,
        capture_event_uid=capture_event_uid,
        primary_sink_id=primary_sink_id,
        secondary_sink_id=secondary_sink_id,
    )
    expected_ledger_file = _fortran_string(capture_ledger_file)
    build_hash = _expected_build_hash(expected_build_git_hash)
    compilation = _artifact(expected_compilation_path, "expected compilation manifest")
    input_artifacts = _case_input_artifacts(case_input_artifact_paths)
    execution_identity = _execution_identity(plan, binding, input_artifacts)
    original_ledger = str(binding["capture_ledger_path"])
    if Path(expected_ledger_file).expanduser().is_absolute() and (
        str(Path(expected_ledger_file).expanduser().resolve()) == original_ledger
    ):
        raise ValueError("zoom capture ledger file must not overwrite the original capture ledger")

    reasons: list[str] = []
    try:
        source = _read_namelist_source(namelist)
        assignments = _assignments(source)
        physics_assignments = _physics_params_assignments(source)
    except ValueError as error:
        assignments = {}
        physics_assignments = {}
        reasons.append(str(error))
    expected_logicals = {
        "smbh": True,
        "smbh_capture_ledger": True,
    }
    for name, expected in expected_logicals.items():
        try:
            if _logical(_unique(physics_assignments, name)) != expected:
                reasons.append(f"{name} must be {str(expected).lower()}")
        except ValueError as error:
            reasons.append(str(error))
    try:
        if _number(_unique(physics_assignments, "rmerge")) != 0.0:
            reasons.append("rmerge must be exactly 0.0 for no finite-radius compaction")
    except ValueError as error:
        reasons.append(str(error))
    for name, expected in execution_identity.items():
        try:
            if _fortran_string(_unique(physics_assignments, name)).lower() != expected:
                reasons.append(f"{name} differs from the materialized CDM zoom identity")
        except ValueError as error:
            reasons.append(str(error))
    try:
        if _number(_unique(assignments, "levelmax")) != selected_case.numerics.levelmax:
            reasons.append("levelmax differs from the selected CDM zoom case")
    except ValueError as error:
        reasons.append(str(error))
    try:
        actual_ledger_file = _fortran_string(
            _unique(physics_assignments, "smbh_capture_ledger_file")
        )
        if actual_ledger_file != expected_ledger_file:
            reasons.append("smbh_capture_ledger_file differs from the materialized run contract")
    except ValueError as error:
        reasons.append(str(error))

    return (
        CDMNonCompactingZoomRunContract(
            specification_path=specification,
            run_namelist_path=namelist,
            plan=plan,
            case=selected_case,
            capture_binding=binding,
            capture_ledger_file=expected_ledger_file,
            expected_build_git_hash=build_hash,
            expected_compilation=compilation,
            case_input_artifacts=input_artifacts,
            execution_identity=execution_identity,
            status=(
                "ready_for_operator_submission"
                if not reasons
                else "not_ready_for_operator_submission"
            ),
            reasons=tuple(reasons),
        ),
        plan,
    )


def materialize_cdm_noncompacting_zoom_run_contract(
    *,
    specification_path: str | Path,
    case_id: str,
    capture_binding_path: str | Path,
    capture_event_uid: str,
    primary_sink_id: int,
    secondary_sink_id: int,
    run_namelist_path: str | Path,
    capture_ledger_file: str,
    expected_build_git_hash: str,
    expected_compilation_path: str | Path,
    case_input_artifact_paths: Mapping[str, str | Path],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write a new run-contract directory; never modify run inputs or submit."""

    contract, _ = assess_cdm_noncompacting_zoom_run_inputs(
        specification_path=specification_path,
        case_id=case_id,
        capture_binding_path=capture_binding_path,
        capture_event_uid=capture_event_uid,
        primary_sink_id=primary_sink_id,
        secondary_sink_id=secondary_sink_id,
        run_namelist_path=run_namelist_path,
        capture_ledger_file=capture_ledger_file,
        expected_build_git_hash=expected_build_git_hash,
        expected_compilation_path=expected_compilation_path,
        case_input_artifact_paths=case_input_artifact_paths,
    )
    destination = Path(output_directory).expanduser().resolve()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("run-contract output directory must not already exist") from error
    controls_path = destination / "required_smbh_controls.nml"
    _write_atomic(
        controls_path,
        _required_smbh_controls(contract.capture_ledger_file, contract.execution_identity),
    )
    record = contract.as_dict(controls_fragment_path=controls_path)
    _write_atomic(
        destination / "cdm_noncompacting_zoom_run_contract.json",
        json.dumps(record, indent=2, sort_keys=True) + "\n",
    )
    return record
