"""Provenance-first CDM/SIDM/FDM comparison-family workflow."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .capture_ledger import CaptureEvent, read_capture_ledger
from .dm_run_provenance import (
    CaptureDMRunBinding,
    DarkMatterRunProvenance,
    bind_capture_event_to_dark_matter_run,
    read_dark_matter_run_provenance,
)
from .fdm_outer_wave_ledger import FDMOuterWaveLedger
from .resolved_physics_inventory import (
    read_lagramses_resolved_physics_inventory_assessment,
)


_MODELS = ("cdm", "sidm", "fdm")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_OUTPUT_NUMBER = re.compile(r"dm_run_provenance_(\d{5})\.txt$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be a JSON object")
    return record


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value.lower()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value


def _resolve(reference: str, base: Path) -> Path:
    path = Path(reference).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _normal_output_directory(path: Path, label: str) -> Path:
    """Return the enclosing output directory for grouped or ungrouped evidence."""

    directory = path.parent
    if directory.name.startswith("group_"):
        directory = directory.parent
    if directory.name != label:
        raise ValueError("evidence does not sit inside its declared normal output directory")
    return directory


@dataclass(frozen=True)
class HashedArtifact:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _nonempty(self.path, "artifact path"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "artifact sha256"))

    def verify(self, base: Path) -> Path:
        source = _resolve(self.path, base)
        try:
            digest = _file_sha256(source)
        except OSError as error:
            raise ValueError(f"cannot read artifact {self.path}: {error}") from error
        if digest != self.sha256:
            raise ValueError(f"artifact SHA-256 differs: {self.path}")
        return source

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


def _artifact_mapping(
    record: Any,
    expected: set[str],
    label: str,
) -> tuple[tuple[str, HashedArtifact], ...]:
    if not isinstance(record, Mapping) or set(record) != expected:
        raise ValueError(f"{label} fields are invalid")
    parsed: list[tuple[str, HashedArtifact]] = []
    for name in sorted(expected):
        value = record[name]
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} {name} must be an artifact object")
        try:
            parsed.append((name, HashedArtifact(**value)))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} {name} is invalid: {error}") from error
    return tuple(parsed)


@dataclass(frozen=True)
class DMComparisonFamilyManifest:
    """Immutable shared inputs and expected output sidecars for one family."""

    source_path: Path
    source_sha256: str
    family_id: str
    shared_inputs: tuple[tuple[str, HashedArtifact], ...]
    run_provenance_paths: tuple[tuple[str, str], ...]

    @property
    def shared_input_map(self) -> dict[str, HashedArtifact]:
        return dict(self.shared_inputs)

    @property
    def run_provenance_map(self) -> dict[str, str]:
        return dict(self.run_provenance_paths)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "family_id": self.family_id,
            "shared_inputs": {
                name: artifact.as_dict() for name, artifact in self.shared_inputs
            },
            "run_provenance": dict(self.run_provenance_paths),
        }


def read_dm_comparison_family_manifest(path: str | Path) -> DMComparisonFamilyManifest:
    """Read a comparison manifest without assuming outputs already exist."""

    source = Path(path).expanduser().resolve()
    record = _json_object(source, "DM comparison-family manifest")
    expected = {"schema_version", "family_id", "shared_inputs", "run_provenance"}
    if record.get("schema_version") != 1 or set(record) != expected:
        raise ValueError("DM comparison-family manifest fields are invalid")
    shared_inputs = _artifact_mapping(
        record.get("shared_inputs"),
        {"initial_conditions", "baryon_configuration", "smbh_seed_catalog"},
        "shared_inputs",
    )
    run_provenance = record.get("run_provenance")
    if not isinstance(run_provenance, Mapping) or set(run_provenance) != set(_MODELS):
        raise ValueError("run_provenance must name exactly cdm, sidm, and fdm")
    paths = []
    for model in _MODELS:
        paths.append((model, _nonempty(run_provenance[model], f"{model} provenance path")))
    return DMComparisonFamilyManifest(
        source_path=source,
        source_sha256=_file_sha256(source),
        family_id=_nonempty(record.get("family_id"), "family_id"),
        shared_inputs=shared_inputs,
        run_provenance_paths=tuple(paths),
    )


@dataclass(frozen=True)
class DMComparisonFamilyPreflight:
    manifest: DMComparisonFamilyManifest
    provenance: tuple[tuple[str, DarkMatterRunProvenance | None], ...]
    status: str
    reasons: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.status == "comparison_family_preflight_ready"

    def provenance_for(self, model: str) -> DarkMatterRunProvenance | None:
        return dict(self.provenance).get(model)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "shared-input and run-identity preflight only; this does not "
                "execute a model, classify a physical binary, or estimate a delay"
            ),
            "family": {
                "family_id": self.manifest.family_id,
                "manifest_path": str(self.manifest.source_path),
                "manifest_sha256": self.manifest.source_sha256,
                "shared_inputs": {
                    name: artifact.as_dict() for name, artifact in self.manifest.shared_inputs
                },
            },
            "run_provenance": {
                model: None if item is None else item.as_dict() for model, item in self.provenance
            },
            "reasons": list(self.reasons),
        }


def preflight_dm_comparison_family(
    manifest: DMComparisonFamilyManifest,
) -> DMComparisonFamilyPreflight:
    """Require common input files and one pinned output sidecar per model."""

    reasons: list[str] = []
    base = manifest.source_path.parent
    for name, artifact in manifest.shared_inputs:
        try:
            artifact.verify(base)
        except ValueError as error:
            reasons.append(f"{name}: {error}")
    provenance: list[tuple[str, DarkMatterRunProvenance | None]] = []
    for model, reference in manifest.run_provenance_paths:
        try:
            item = read_dark_matter_run_provenance(_resolve(reference, base))
            if item.dark_matter_model != model:
                reasons.append(f"{model} sidecar declares {item.dark_matter_model}")
            provenance.append((model, item))
        except ValueError as error:
            reasons.append(f"{model} sidecar: {error}")
            provenance.append((model, None))
    available = [item for _, item in provenance if item is not None]
    build_hashes = {item.build_git_hash for item in available}
    if any(not item.build_git_hash or item.build_git_hash == "unknown" for item in available):
        reasons.append("all run sidecars require a concrete build_git_hash")
    if len(available) == len(_MODELS) and len(build_hashes) != 1:
        reasons.append("CDM/SIDM/FDM sidecars do not share one build_git_hash")
    fdm = dict(provenance).get("fdm")
    if fdm is not None and fdm.parameter("fdm_outer_ledger_enabled") is not True:
        reasons.append("FDM sidecar must enable fdm_outer_ledger for the comparison family")
    return DMComparisonFamilyPreflight(
        manifest=manifest,
        provenance=tuple(provenance),
        status="comparison_family_preflight_ready" if not reasons else "comparison_family_preflight_not_ready",
        reasons=tuple(reasons),
    )


def _output_complete_marker(provenance: DarkMatterRunProvenance) -> tuple[Path, str]:
    match = _OUTPUT_NUMBER.fullmatch(provenance.source_path.name)
    if match is None:
        raise ValueError("DM sidecar filename does not identify an output number")
    directory = provenance.source_path.parent
    if directory.name.startswith("group_"):
        directory = directory.parent
    if directory.name != f"output_{match.group(1)}":
        raise ValueError("DM sidecar is not located in its declared output directory")
    return directory / "COMPLETE", match.group(1)


@dataclass(frozen=True)
class DMComparisonSmokeAssessment:
    preflight: DMComparisonFamilyPreflight
    status: str
    reasons: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return self.status == "comparison_family_smoke_verified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "completed-output and model-sidecar smoke check only; no capture "
                "ensemble or physical model comparison is accepted here"
            ),
            "preflight": self.preflight.as_dict(),
            "reasons": list(self.reasons),
        }


def assess_dm_comparison_smoke_outputs(
    preflight: DMComparisonFamilyPreflight,
) -> DMComparisonSmokeAssessment:
    """Require each sidecar to sit in a fully written normal output directory."""

    reasons = list(preflight.reasons)
    if preflight.ready:
        for model, provenance in preflight.provenance:
            assert provenance is not None
            try:
                marker, output_number = _output_complete_marker(provenance)
                if marker.read_text(encoding="utf-8").strip() != output_number:
                    reasons.append(f"{model} output COMPLETE marker is invalid")
            except (OSError, ValueError) as error:
                reasons.append(f"{model} output completeness: {error}")
    return DMComparisonSmokeAssessment(
        preflight=preflight,
        status="comparison_family_smoke_verified" if not reasons else "comparison_family_smoke_not_verified",
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class DMComparisonCaptureRegistration:
    source_path: Path
    family_manifest_path: str
    family_manifest_sha256: str
    captures: tuple[tuple[str, tuple[str, str]], ...]


def read_dm_comparison_capture_registration(path: str | Path) -> DMComparisonCaptureRegistration:
    source = Path(path).expanduser().resolve()
    record = _json_object(source, "DM comparison capture registration")
    if record.get("schema_version") != 1 or set(record) != {
        "schema_version", "family_manifest_path", "family_manifest_sha256", "captures"
    }:
        raise ValueError("DM comparison capture-registration fields are invalid")
    captures = record.get("captures")
    if not isinstance(captures, Mapping) or set(captures) != set(_MODELS):
        raise ValueError("capture registration must name exactly cdm, sidm, and fdm")
    parsed: list[tuple[str, tuple[str, str]]] = []
    for model in _MODELS:
        value = captures[model]
        if not isinstance(value, Mapping) or set(value) != {"ledger_path", "event_uid"}:
            raise ValueError(f"{model} capture-registration fields are invalid")
        parsed.append(
            (
                model,
                (
                    _nonempty(value.get("ledger_path"), f"{model} ledger_path"),
                    _nonempty(value.get("event_uid"), f"{model} event_uid"),
                ),
            )
        )
    return DMComparisonCaptureRegistration(
        source_path=source,
        family_manifest_path=_nonempty(record.get("family_manifest_path"), "family_manifest_path"),
        family_manifest_sha256=_sha256(record.get("family_manifest_sha256"), "family_manifest_sha256"),
        captures=tuple(parsed),
    )


@dataclass(frozen=True)
class DMComparisonCaptureEnsemble:
    smoke: DMComparisonSmokeAssessment
    bindings: tuple[tuple[str, CaptureDMRunBinding | None], ...]
    status: str
    reasons: tuple[str, ...]

    @property
    def registered(self) -> bool:
        return self.status == "dm_comparison_capture_ensemble_registered"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "numerical-capture ensemble registration only; model events need not "
                "share a step or sink ID and are not physical coalescences"
            ),
            "smoke": self.smoke.as_dict(),
            "capture_bindings": {
                model: None if item is None else item.as_dict() for model, item in self.bindings
            },
            "reasons": list(self.reasons),
        }


def register_dm_comparison_capture_ensemble(
    registration: DMComparisonCaptureRegistration,
) -> DMComparisonCaptureEnsemble:
    """Register one complete numerical-capture event per verified model run."""

    reasons: list[str] = []
    manifest_path = _resolve(registration.family_manifest_path, registration.source_path.parent)
    try:
        manifest = read_dm_comparison_family_manifest(manifest_path)
        if manifest.source_sha256 != registration.family_manifest_sha256:
            reasons.append("capture registration family manifest SHA-256 differs")
        smoke = assess_dm_comparison_smoke_outputs(preflight_dm_comparison_family(manifest))
        reasons.extend(smoke.reasons)
    except ValueError as error:
        placeholder = DMComparisonFamilyPreflight(
            manifest=DMComparisonFamilyManifest(
                source_path=manifest_path,
                source_sha256="0" * 64,
                family_id="unreadable",
                shared_inputs=(),
                run_provenance_paths=(),
            ),
            provenance=(),
            status="comparison_family_preflight_not_ready",
            reasons=(str(error),),
        )
        smoke = DMComparisonSmokeAssessment(
            preflight=placeholder,
            status="comparison_family_smoke_not_verified",
            reasons=(str(error),),
        )
        reasons.append(str(error))
    bindings: list[tuple[str, CaptureDMRunBinding | None]] = []
    if smoke.verified:
        for model, (ledger_reference, event_uid) in registration.captures:
            provenance = smoke.preflight.provenance_for(model)
            assert provenance is not None
            try:
                events = [
                    item
                    for item in read_capture_ledger(
                        _resolve(ledger_reference, registration.source_path.parent)
                    ).events
                    if item.event_uid == event_uid
                ]
                if len(events) != 1:
                    raise ValueError("event_uid must identify exactly one complete capture event")
                binding = bind_capture_event_to_dark_matter_run(events[0], provenance)
                if not binding.bound:
                    reasons.extend(f"{model}: {reason}" for reason in binding.reasons)
                bindings.append((model, binding))
            except ValueError as error:
                reasons.append(f"{model} capture: {error}")
                bindings.append((model, None))
    else:
        bindings = [(model, None) for model in _MODELS]
    return DMComparisonCaptureEnsemble(
        smoke=smoke,
        bindings=tuple(bindings),
        status="dm_comparison_capture_ensemble_registered" if not reasons else "dm_comparison_capture_ensemble_not_registered",
        reasons=tuple(reasons),
    )


_PHYSICS_ARTIFACTS = {
    "cdm": {"environment_profile", "force_ledger", "conservation_ledger"},
    "sidm": {
        "environment_profile",
        "force_ledger",
        "conservation_ledger",
        "scattering_ledger",
    },
    "fdm": {
        "environment_profile",
        "force_ledger",
        "conservation_ledger",
        "wave_ledger",
        "wave_provenance",
        "field_snapshot_index",
    },
}
_V1_PHYSICS_ARTIFACTS = {
    model: set(names) - ({"wave_provenance"} if model == "fdm" else set())
    for model, names in _PHYSICS_ARTIFACTS.items()
}


@dataclass(frozen=True)
class DMComparisonPhysicsInput:
    source_path: Path
    source_schema_version: int
    capture_ensemble_path: str
    capture_ensemble_sha256: str
    artifacts: tuple[tuple[str, tuple[tuple[str, HashedArtifact], ...]], ...]
    normal_output_inventory_assessments: tuple[tuple[str, HashedArtifact], ...]

    def inventory_assessment_for(self, model: str) -> HashedArtifact | None:
        return dict(self.normal_output_inventory_assessments).get(model)


def read_dm_comparison_physics_input(path: str | Path) -> DMComparisonPhysicsInput:
    source = Path(path).expanduser().resolve()
    record = _json_object(source, "DM comparison physics-input record")
    source_schema_version = record.get("schema_version")
    expected = {"schema_version", "capture_ensemble_path", "capture_ensemble_sha256", "artifacts"}
    if source_schema_version == 2:
        expected.add("normal_output_inventory_assessments")
    elif source_schema_version != 1:
        raise ValueError("unsupported DM comparison physics-input schema")
    if set(record) != expected:
        raise ValueError("DM comparison physics-input fields are invalid")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(_MODELS):
        raise ValueError("DM comparison physics input must name exactly cdm, sidm, and fdm")
    inventory_assessments: tuple[tuple[str, HashedArtifact], ...] = ()
    if source_schema_version == 2:
        inventory_assessments = _artifact_mapping(
            record.get("normal_output_inventory_assessments"),
            set(_MODELS),
            "normal_output_inventory_assessments",
        )
    return DMComparisonPhysicsInput(
        source_path=source,
        source_schema_version=source_schema_version,
        capture_ensemble_path=_nonempty(record.get("capture_ensemble_path"), "capture_ensemble_path"),
        capture_ensemble_sha256=_sha256(record.get("capture_ensemble_sha256"), "capture_ensemble_sha256"),
        artifacts=tuple(
            (
                model,
                _artifact_mapping(
                    artifacts[model],
                    (_PHYSICS_ARTIFACTS if source_schema_version == 2 else _V1_PHYSICS_ARTIFACTS)[model],
                    f"{model} artifacts",
                ),
            )
            for model in _MODELS
        ),
        normal_output_inventory_assessments=inventory_assessments,
    )


@dataclass(frozen=True)
class DMComparisonPhysicsAssessment:
    physics_input: DMComparisonPhysicsInput
    status: str
    reasons: tuple[str, ...]

    @property
    def ready_for_model_specific_analysis(self) -> bool:
        return self.status == "dm_comparison_physics_inputs_verified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "resolved-input readiness only; CDM/SIDM are comparison baselines and "
                "the FDM branch still requires its separate outer-to-inner acceptance"
            ),
            "physics_input": {
                "path": str(self.physics_input.source_path),
                "sha256": _file_sha256(self.physics_input.source_path),
            },
            "reasons": list(self.reasons),
        }


def assess_dm_comparison_physics_inputs(
    physics_input: DMComparisonPhysicsInput,
) -> DMComparisonPhysicsAssessment:
    """Verify all model-specific resolved-environment artifacts by hash."""

    reasons: list[str] = []
    ensemble_path = _resolve(physics_input.capture_ensemble_path, physics_input.source_path.parent)
    ensemble: dict[str, Any] | None = None
    try:
        if _file_sha256(ensemble_path) != physics_input.capture_ensemble_sha256:
            reasons.append("capture ensemble SHA-256 differs")
        ensemble = _json_object(ensemble_path, "capture ensemble")
        if ensemble.get("schema_version") != 1:
            reasons.append("capture ensemble schema is unsupported")
        if ensemble.get("status") != "dm_comparison_capture_ensemble_registered":
            reasons.append("capture ensemble is not registered")
    except (OSError, ValueError) as error:
        reasons.append(f"capture ensemble: {error}")
    verified_artifacts: dict[str, dict[str, Path]] = {}
    for model, artifacts in physics_input.artifacts:
        verified_artifacts[model] = {}
        for name, artifact in artifacts:
            try:
                verified_artifacts[model][name] = artifact.verify(
                    physics_input.source_path.parent
                )
            except ValueError as error:
                reasons.append(f"{model} {name}: {error}")
    if physics_input.source_schema_version < 2:
        reasons.append("physics input schema lacks normal-output inventory assessments")
    else:
        for model, inventory_artifact in physics_input.normal_output_inventory_assessments:
            try:
                assessment_path = inventory_artifact.verify(physics_input.source_path.parent)
                assessment = read_lagramses_resolved_physics_inventory_assessment(
                    assessment_path
                )
                if assessment.inventory.dark_matter_model != model:
                    reasons.append(f"{model} inventory assessment declares {assessment.inventory.dark_matter_model}")
                if not assessment.ready_for_registered_analysis:
                    reasons.append(f"{model} normal-output inventory assessment is not ready")
                if ensemble is None:
                    reasons.append(f"{model} capture ensemble is unavailable for output binding")
                else:
                    bindings = ensemble.get("capture_bindings")
                    binding = bindings.get(model) if isinstance(bindings, Mapping) else None
                    run = binding.get("run_provenance") if isinstance(binding, Mapping) else None
                    run_source = run.get("source") if isinstance(run, Mapping) else None
                    reference = run_source.get("path") if isinstance(run_source, Mapping) else None
                    if not isinstance(reference, str) or not reference.strip():
                        reasons.append(f"{model} capture ensemble lacks a run-provenance source path")
                    else:
                        try:
                            inventory_output = _normal_output_directory(
                                assessment.inventory.source_path,
                                f"output_{assessment.inventory.output_number}",
                            )
                            capture_output = _normal_output_directory(
                                _resolve(reference, ensemble_path.parent),
                                f"output_{assessment.inventory.output_number}",
                            )
                            if capture_output != inventory_output:
                                reasons.append(
                                    f"{model} normal-output inventory differs from its capture ensemble output"
                                )
                        except ValueError as error:
                            reasons.append(f"{model} normal-output inventory output binding: {error}")
                artifact_map = dict(dict(physics_input.artifacts)[model])
                expected_ledgers = {
                    "force_ledger": assessment.inventory.force_source_ledger_sha256,
                    "conservation_ledger": assessment.inventory.conservation_ledger_sha256,
                }
                if model == "sidm":
                    expected_ledgers["scattering_ledger"] = (
                        assessment.inventory.sidm_scattering_ledger_sha256
                    )
                elif model == "fdm":
                    expected_ledgers["wave_provenance"] = (
                        assessment.inventory.fdm_wave_provenance_sha256
                    )
                for name, expected_sha256 in expected_ledgers.items():
                    if expected_sha256 is None:
                        reasons.append(f"{model} inventory does not attest {name}")
                    elif artifact_map[name].sha256 != expected_sha256:
                        reasons.append(f"{model} {name} SHA-256 differs from its normal-output inventory")
                if model == "fdm":
                    try:
                        ledger_path = verified_artifacts[model]["wave_ledger"]
                        ledger = FDMOuterWaveLedger.from_dict(
                            _json_object(ledger_path, "FDM outer-wave ledger")
                        )
                        source_path = _resolve(ledger.source_path, ledger_path.parent)
                        if _file_sha256(source_path) != ledger.source_sha256:
                            reasons.append("fdm wave ledger source SHA-256 differs")
                        if ledger.force_ledger_sha256 != artifact_map["force_ledger"].sha256:
                            reasons.append("fdm wave ledger force ledger differs from registered evidence")
                        if (
                            ledger.field_snapshot_index_sha256
                            != artifact_map["field_snapshot_index"].sha256
                        ):
                            reasons.append(
                                "fdm wave ledger field-snapshot index differs from registered evidence"
                            )
                        if (
                            ledger.profile_snapshot_index_sha256
                            != artifact_map["environment_profile"].sha256
                        ):
                            reasons.append(
                                "fdm wave ledger profile-snapshot index differs from registered evidence"
                            )
                        if ledger.force_accounting != "live_wave_only":
                            reasons.append(
                                "fdm wave ledger must use live_wave_only until a residual ledger is attested"
                            )
                    except (KeyError, ValueError) as error:
                        reasons.append(f"fdm wave ledger: {error}")
            except ValueError as error:
                reasons.append(f"{model} normal-output inventory assessment: {error}")
    return DMComparisonPhysicsAssessment(
        physics_input=physics_input,
        status="dm_comparison_physics_inputs_verified" if not reasons else "dm_comparison_physics_inputs_not_verified",
        reasons=tuple(reasons),
    )
