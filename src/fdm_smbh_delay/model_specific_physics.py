"""Censor-first resolved physics records for separate CDM/SIDM/FDM analyses.

The classes in this module validate measured inputs and resolution agreement.
They neither prescribe a drag force nor turn a numerical capture or rate
comparison into a physical SMBH coalescence time.
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

from .dm_comparison import (
    DMComparisonPhysicsInput,
    assess_dm_comparison_physics_inputs,
    read_verified_dm_comparison_capture_ensemble,
    read_dm_comparison_physics_input,
)
from .dm_run_provenance import DarkMatterRunProvenance, read_dark_matter_run_provenance
from .outer_inner_handoff import HandoffRatePoint
from .resolved_physics_inventory import (
    ResolvedPhysicsInventoryAssessment,
    read_lagramses_resolved_physics_inventory_assessment,
)
from .zoom_calibration import GalaxyMergerZoomCase


MODEL_SPECIFIC_PHYSICS_RESULT_SCHEMA_VERSION = 3
MODEL_SPECIFIC_RATE_LEDGER_SCHEMA_VERSION = 2
_MODELS = ("cdm", "sidm", "fdm")
_CHANNELS = ("stars", "gas", "dark_matter")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_FDM_FORCE_ACCOUNTING = {"live_wave_only"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character SHA-256")
    return value.lower()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _resolve(reference: str, base: Path) -> Path:
    path = Path(reference).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True)
class ResolvedEnvironmentChannel:
    """Availability and immutable evidence for one actual force channel."""

    status: str
    profile_sha256: str | None
    force_ledger_sha256: str | None

    def __post_init__(self) -> None:
        if self.status not in {"available", "absent"}:
            raise ValueError("environment channel status must be available or absent")
        if self.status == "available":
            object.__setattr__(self, "profile_sha256", _sha256(self.profile_sha256, "profile_sha256"))
            object.__setattr__(
                self,
                "force_ledger_sha256",
                _sha256(self.force_ledger_sha256, "force_ledger_sha256"),
            )
        elif self.profile_sha256 is not None or self.force_ledger_sha256 is not None:
            raise ValueError("absent environment channels cannot carry evidence hashes")

    def as_dict(self) -> dict[str, str | None]:
        return {
            "status": self.status,
            "profile_sha256": self.profile_sha256,
            "force_ledger_sha256": self.force_ledger_sha256,
        }

    @classmethod
    def from_dict(cls, record: Any, label: str) -> "ResolvedEnvironmentChannel":
        if not isinstance(record, Mapping) or set(record) != {
            "status",
            "profile_sha256",
            "force_ledger_sha256",
        }:
            raise ValueError(f"{label} channel fields are invalid")
        try:
            return cls(
                status=record.get("status"),
                profile_sha256=record.get("profile_sha256"),
                force_ledger_sha256=record.get("force_ledger_sha256"),
            )
        except ValueError as error:
            raise ValueError(f"{label} channel is invalid: {error}") from error


@dataclass(frozen=True)
class ModelSpecificEvidence:
    """Model-specific artifacts, without replacing one model by another."""

    dark_matter_model: str
    artifact_sha256s: tuple[tuple[str, str], ...]
    maximum_scatter_probability: float | None = None
    force_accounting: str | None = None
    minimum_de_broglie_resolution_cells: float | None = None
    minimum_wake_resolution_cells: float | None = None

    def __post_init__(self) -> None:
        if self.dark_matter_model not in _MODELS:
            raise ValueError("model-specific evidence requires cdm, sidm, or fdm")
        keys = [name for name, _ in self.artifact_sha256s]
        if len(keys) != len(set(keys)):
            raise ValueError("model-specific artifact hashes are duplicated")
        for name, digest in self.artifact_sha256s:
            _nonempty(name, "artifact name")
            _sha256(digest, f"{name} SHA-256")
        if self.dark_matter_model == "cdm":
            if any(
                value is not None
                for value in (
                    self.maximum_scatter_probability,
                    self.force_accounting,
                    self.minimum_de_broglie_resolution_cells,
                    self.minimum_wake_resolution_cells,
                )
            ):
                raise ValueError("CDM evidence cannot carry SIDM or FDM controls")
        elif self.dark_matter_model == "sidm":
            if self.maximum_scatter_probability is None:
                raise ValueError("SIDM evidence requires maximum_scatter_probability")
            probability = _finite(
                self.maximum_scatter_probability,
                "maximum_scatter_probability",
            )
            if not 0.0 <= probability <= 1.0:
                raise ValueError("maximum_scatter_probability must lie in [0, 1]")
            object.__setattr__(self, "maximum_scatter_probability", probability)
            if any(
                value is not None
                for value in (
                    self.force_accounting,
                    self.minimum_de_broglie_resolution_cells,
                    self.minimum_wake_resolution_cells,
                )
            ):
                raise ValueError("SIDM evidence cannot carry FDM wave controls")
        else:
            if self.force_accounting not in _FDM_FORCE_ACCOUNTING:
                raise ValueError("FDM evidence must preserve resolved-wave force accounting")
            for name in (
                "minimum_de_broglie_resolution_cells",
                "minimum_wake_resolution_cells",
            ):
                value = _finite(getattr(self, name), name, positive=True)
                object.__setattr__(self, name, value)
            if self.maximum_scatter_probability is not None:
                raise ValueError("FDM evidence cannot carry SIDM scattering controls")

    @property
    def artifact_map(self) -> dict[str, str]:
        return dict(self.artifact_sha256s)

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"artifact_sha256s": dict(self.artifact_sha256s)}
        if self.dark_matter_model == "sidm":
            record["maximum_scatter_probability"] = self.maximum_scatter_probability
        elif self.dark_matter_model == "fdm":
            record.update(
                {
                    "force_accounting": self.force_accounting,
                    "minimum_de_broglie_resolution_cells": self.minimum_de_broglie_resolution_cells,
                    "minimum_wake_resolution_cells": self.minimum_wake_resolution_cells,
                }
            )
        return record

    @classmethod
    def from_dict(
        cls,
        record: Any,
        *,
        model: str,
        expected_artifacts: Mapping[str, str],
    ) -> "ModelSpecificEvidence":
        expected_fields = {"artifact_sha256s"}
        if model == "sidm":
            expected_fields.add("maximum_scatter_probability")
        elif model == "fdm":
            expected_fields.update(
                {
                    "force_accounting",
                    "minimum_de_broglie_resolution_cells",
                    "minimum_wake_resolution_cells",
                }
            )
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise ValueError(f"{model} model_evidence fields are invalid")
        artifact_record = record.get("artifact_sha256s")
        if not isinstance(artifact_record, Mapping) or set(artifact_record) != set(expected_artifacts):
            raise ValueError(f"{model} model_evidence artifacts are invalid")
        artifacts: list[tuple[str, str]] = []
        for name in sorted(expected_artifacts):
            digest = _sha256(artifact_record[name], f"{model} {name} SHA-256")
            if digest != expected_artifacts[name]:
                raise ValueError(f"{model} {name} SHA-256 differs from accepted physics input")
            artifacts.append((name, digest))
        try:
            return cls(
                dark_matter_model=model,
                artifact_sha256s=tuple(artifacts),
                maximum_scatter_probability=record.get("maximum_scatter_probability"),
                force_accounting=record.get("force_accounting"),
                minimum_de_broglie_resolution_cells=record.get(
                    "minimum_de_broglie_resolution_cells"
                ),
                minimum_wake_resolution_cells=record.get("minimum_wake_resolution_cells"),
            )
        except ValueError as error:
            raise ValueError(f"{model} model_evidence is invalid: {error}") from error


@dataclass(frozen=True)
class ResolvedModelPhysicsResult:
    """One completed resolved result, still not a merger-time calibration."""

    case: GalaxyMergerZoomCase
    zoom_manifest_sha256: str
    source_path: Path
    source_sha256: str
    capture_event_uid: str
    capture_event_sha256: str
    physics_input_path: Path
    physics_input_sha256: str
    rate_ledger_path: Path
    rate_ledger_sha256: str
    comparison_family_id: str
    comparison_shared_input_sha256s: tuple[tuple[str, str], ...]
    channels: tuple[tuple[str, ResolvedEnvironmentChannel], ...]
    rate_points: tuple[HandoffRatePoint, ...]
    maximum_relative_energy_error: float
    maximum_relative_angular_momentum_error: float
    minimum_orbital_resolution_cells: float
    model_evidence: ModelSpecificEvidence

    def __post_init__(self) -> None:
        _sha256(self.zoom_manifest_sha256, "zoom_manifest_sha256")
        _sha256(self.source_sha256, "source_sha256")
        _sha256(self.physics_input_sha256, "physics_input_sha256")
        _sha256(self.rate_ledger_sha256, "rate_ledger_sha256")
        _nonempty(self.comparison_family_id, "comparison_family_id")
        if {name for name, _ in self.comparison_shared_input_sha256s} != {
            "initial_conditions",
            "baryon_configuration",
            "smbh_seed_catalog",
        }:
            raise ValueError("comparison shared-input identities are invalid")
        for name, digest in self.comparison_shared_input_sha256s:
            _sha256(digest, f"{name} SHA-256")
        _nonempty(self.capture_event_uid, "capture_event_uid")
        _sha256(self.capture_event_sha256, "capture_event_sha256")
        if self.model_evidence.dark_matter_model != self.case.physics.dark_matter_model:
            raise ValueError("result model evidence does not match its zoom case")
        names = tuple(name for name, _ in self.channels)
        if set(names) != set(_CHANNELS) or len(names) != len(_CHANNELS):
            raise ValueError("resolved result must name stars, gas, and dark_matter channels")
        channels = dict(self.channels)
        if channels["dark_matter"].status != "available":
            raise ValueError("the active dark_matter channel must be available")
        if self.case.physics.host_stellar_mass_msun > 0.0 and channels["stars"].status != "available":
            raise ValueError("nonzero stellar mass requires an available stars channel")
        if self.case.physics.gas_fraction > 0.0 and channels["gas"].status != "available":
            raise ValueError("nonzero gas fraction requires an available gas channel")
        if len(self.rate_points) < 3:
            raise ValueError("resolved result requires at least three rate points")
        separations = np.asarray([item.separation_pc for item in self.rate_points], dtype=float)
        if np.any(np.diff(separations) <= 0.0):
            raise ValueError("resolved result rate-point separations must be strictly increasing")
        for name in (
            "maximum_relative_energy_error",
            "maximum_relative_angular_momentum_error",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "minimum_orbital_resolution_cells",
            _finite(self.minimum_orbital_resolution_cells, "minimum_orbital_resolution_cells", positive=True),
        )

    @property
    def dark_matter_model(self) -> str:
        return self.case.physics.dark_matter_model

    @property
    def channel_map(self) -> dict[str, ResolvedEnvironmentChannel]:
        return dict(self.channels)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "resolved_model_physics_ready",
            "interpretation": (
                "resolved-input and rate provenance only; this is not a physical "
                "coalescence time or a cross-model delay calibration"
            ),
            "case_id": self.case.case_id,
            "dark_matter_model": self.dark_matter_model,
            "zoom_manifest_sha256": self.zoom_manifest_sha256,
            "source": {"path": str(self.source_path), "sha256": self.source_sha256},
            "capture_event_uid": self.capture_event_uid,
            "capture_event_sha256": self.capture_event_sha256,
            "physics_input": {
                "path": str(self.physics_input_path),
                "sha256": self.physics_input_sha256,
            },
            "rate_ledger": {
                "path": str(self.rate_ledger_path),
                "sha256": self.rate_ledger_sha256,
            },
            "comparison_family": {
                "family_id": self.comparison_family_id,
                "shared_input_sha256s": dict(self.comparison_shared_input_sha256s),
            },
            "environment_channels": {
                name: channel.as_dict() for name, channel in self.channels
            },
            "rate_points": [
                {
                    "separation_pc": point.separation_pc,
                    "orbital_power_pc2_myr3": point.orbital_power_pc2_myr3,
                    "orbital_torque_msun_pc2_myr": point.orbital_torque_msun_pc2_myr,
                    "eccentricity": point.eccentricity,
                }
                for point in self.rate_points
            ],
            "diagnostics": {
                "maximum_relative_energy_error": self.maximum_relative_energy_error,
                "maximum_relative_angular_momentum_error": self.maximum_relative_angular_momentum_error,
                "minimum_orbital_resolution_cells": self.minimum_orbital_resolution_cells,
            },
            "model_evidence": self.model_evidence.as_dict(),
        }


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be a JSON object")
    return record


def _accepted_artifacts(physics_input: DMComparisonPhysicsInput, model: str) -> dict[str, str]:
    for item_model, artifacts in physics_input.artifacts:
        if item_model == model:
            return {name: artifact.sha256 for name, artifact in artifacts}
    raise ValueError(f"accepted physics input lacks {model} artifacts")


def _accepted_inventory_assessment(
    physics_input: DMComparisonPhysicsInput,
    model: str,
) -> ResolvedPhysicsInventoryAssessment:
    artifact = physics_input.inventory_assessment_for(model)
    if artifact is None:
        raise ValueError(f"accepted physics input lacks {model} normal-output inventory assessment")
    try:
        path = artifact.verify(physics_input.source_path.parent)
    except ValueError as error:
        raise ValueError(f"{model} normal-output inventory assessment: {error}") from error
    try:
        assessment = read_lagramses_resolved_physics_inventory_assessment(path)
    except ValueError as error:
        raise ValueError(f"{model} normal-output inventory assessment: {error}") from error
    if assessment.inventory.dark_matter_model != model:
        raise ValueError(f"{model} normal-output inventory assessment declares another model")
    if not assessment.ready_for_registered_analysis:
        raise ValueError(f"{model} normal-output inventory assessment is not ready")
    return assessment


def _registered_capture_binding(
    physics_input: DMComparisonPhysicsInput,
    model: str,
) -> Mapping[str, Any]:
    """Re-read the accepted ensemble and return one model's bound capture record."""

    ensemble = _registered_capture_ensemble(physics_input).as_dict()
    bindings = ensemble.get("capture_bindings")
    binding = bindings.get(model) if isinstance(bindings, Mapping) else None
    if not isinstance(binding, Mapping):
        raise ValueError(f"accepted capture ensemble lacks a {model} binding")
    capture = binding.get("capture_event") if isinstance(binding, Mapping) else None
    event_uid = capture.get("event_uid") if isinstance(capture, Mapping) else None
    _nonempty(event_uid, f"accepted {model} capture_event_uid")
    run = binding.get("run_provenance")
    source = run.get("source") if isinstance(run, Mapping) else None
    if not isinstance(source, Mapping):
        raise ValueError(f"accepted {model} capture binding lacks run provenance")
    _nonempty(source.get("path"), f"accepted {model} run-provenance path")
    _sha256(source.get("sha256"), f"accepted {model} run-provenance SHA-256")
    return binding


def _registered_capture_ensemble(physics_input: DMComparisonPhysicsInput) -> Any:
    ensemble_path = _resolve(physics_input.capture_ensemble_path, physics_input.source_path.parent)
    try:
        if _file_sha256(ensemble_path) != physics_input.capture_ensemble_sha256:
            raise ValueError("capture ensemble SHA-256 differs")
        return read_verified_dm_comparison_capture_ensemble(ensemble_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"accepted capture ensemble is invalid: {error}") from error


def _registered_capture_event_uid(
    physics_input: DMComparisonPhysicsInput,
    model: str,
) -> str:
    binding = _registered_capture_binding(physics_input, model)
    capture = binding["capture_event"]
    assert isinstance(capture, Mapping)
    return _nonempty(capture.get("event_uid"), f"accepted {model} capture_event_uid")


def _artifact_path_and_sha256(
    record: Any,
    *,
    base: Path,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} artifact is invalid")
    path = _resolve(_nonempty(record.get("path"), f"{label} path"), base)
    digest = _sha256(record.get("sha256"), f"{label} SHA-256")
    try:
        actual = _file_sha256(path)
    except OSError as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if actual != digest:
        raise ValueError(f"{label} SHA-256 differs")
    return path, digest


def _normal_output_directory(path: Path) -> Path:
    directory = path.parent
    if re.fullmatch(r"group_\d{5}", directory.name):
        directory = directory.parent
    if not re.fullmatch(r"output_\d{5}", directory.name):
        raise ValueError("normal-output evidence is not inside output_00000")
    return directory


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-14)


def _require_model_zoom_execution_identity(
    run: DarkMatterRunProvenance,
    *,
    case: GalaxyMergerZoomCase,
    zoom_manifest_sha256: str,
    capture_binding: Mapping[str, Any],
    capture_ensemble: Any,
) -> None:
    """Bind the resolved result's requested case to the sidecar read by the solver."""

    if run.parameter("model_zoom_execution_identity_status") != "available":
        raise ValueError("normal-output sidecar lacks an available model zoom execution identity")
    expected_digests = {
        "model_zoom_manifest_sha256": _sha256(
            zoom_manifest_sha256, "zoom_manifest_sha256"
        ),
        "model_zoom_capture_event_sha256": _sha256(
            capture_binding["capture_event"]["event_sha256"], "capture event SHA-256"
        ),
        "model_zoom_initial_conditions_sha256": capture_ensemble.smoke.preflight.manifest.shared_input_map[
            "initial_conditions"
        ].sha256,
        "model_zoom_baryon_configuration_sha256": capture_ensemble.smoke.preflight.manifest.shared_input_map[
            "baryon_configuration"
        ].sha256,
        "model_zoom_sink_initial_conditions_sha256": capture_ensemble.smoke.preflight.manifest.shared_input_map[
            "smbh_seed_catalog"
        ].sha256,
    }
    if run.parameter("model_zoom_case_id") != case.case_id:
        raise ValueError("normal-output sidecar model zoom case differs from its result")
    for name, expected in expected_digests.items():
        if run.parameter(name) != expected:
            raise ValueError(f"normal-output sidecar {name} differs from its result")
    if run.dark_matter_model == "sidm":
        expected_sidm = {
            "sidm_cross_section_cm2_g": case.physics.sidm_cross_section_cm2_g,
            "sidm_v0_km_s": case.physics.sidm_v0_km_s,
            "sidm_power": case.physics.sidm_power,
            "sidm_type": case.physics.sidm_type,
            "sidm_angular": case.physics.sidm_angular,
            "sidm_inelastic": case.physics.sidm_inelastic,
        }
        if any(run.parameter(name) != value for name, value in expected_sidm.items()):
            raise ValueError("normal-output SIDM controls differ from the zoom case")
    elif run.dark_matter_model == "fdm":
        if not _same_float(
            float(run.parameter("m_axion_ev")), float(case.physics.fdm_particle_mass_ev)
        ):
            raise ValueError("normal-output FDM particle mass differs from the zoom case")


def _read_rate_ledger(
    record: Mapping[str, Any],
    *,
    result_path: Path,
    case: GalaxyMergerZoomCase,
    zoom_manifest_sha256: str,
    physics_input: DMComparisonPhysicsInput,
    inventory_assessment: ResolvedPhysicsInventoryAssessment,
    capture_binding: Mapping[str, Any],
    capture_ensemble: Any,
) -> tuple[Path, str]:
    """Require an independent, output-bound source for every accepted rate field."""

    ledger_path = _resolve(
        _nonempty(record.get("rate_ledger_path"), "rate_ledger_path"), result_path.parent
    )
    ledger_sha256 = _sha256(record.get("rate_ledger_sha256"), "rate_ledger_sha256")
    try:
        if _file_sha256(ledger_path) != ledger_sha256:
            raise ValueError("rate ledger SHA-256 differs")
        ledger = _read_object(ledger_path, "model-specific rate ledger")
    except (OSError, ValueError) as error:
        raise ValueError(f"model-specific rate ledger is invalid: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "case_id",
        "case",
        "dark_matter_model",
        "zoom_manifest_sha256",
        "capture_event_uid",
        "capture_event_sha256",
        "normal_output_inventory",
        "run_provenance",
        "environment_channels",
        "rate_points",
        "diagnostics",
        "model_evidence",
    }
    if (
        ledger.get("schema_version") != MODEL_SPECIFIC_RATE_LEDGER_SCHEMA_VERSION
        or ledger.get("status") != "diagnosed"
        or set(ledger) != expected_fields
    ):
        raise ValueError("model-specific rate ledger fields are invalid")
    if (
        ledger.get("case_id") != case.case_id
        or ledger.get("case") != case.as_dict()
        or ledger.get("dark_matter_model") != case.physics.dark_matter_model
        or _sha256(ledger.get("zoom_manifest_sha256"), "rate ledger zoom_manifest_sha256")
        != _sha256(zoom_manifest_sha256, "zoom_manifest_sha256")
        or ledger.get("capture_event_uid") != record.get("capture_event_uid")
        or _sha256(ledger.get("capture_event_sha256"), "rate ledger capture_event_sha256")
        != _sha256(record.get("capture_event_sha256"), "capture_event_sha256")
    ):
        raise ValueError("model-specific rate ledger provenance differs from its result")
    inventory_path, inventory_sha256 = _artifact_path_and_sha256(
        ledger.get("normal_output_inventory"),
        base=ledger_path.parent,
        label="rate ledger normal-output inventory",
    )
    inventory = inventory_assessment.inventory
    if inventory_path != inventory.source_path or inventory_sha256 != inventory.source_sha256:
        raise ValueError("rate ledger normal-output inventory differs from accepted evidence")
    run_path, run_sha256 = _artifact_path_and_sha256(
        ledger.get("run_provenance"),
        base=ledger_path.parent,
        label="rate ledger run provenance",
    )
    run = read_dark_matter_run_provenance(run_path)
    if run.source_sha256 != run_sha256:
        raise ValueError("rate ledger run-provenance parser SHA-256 differs")
    if run_path.name != f"dm_run_provenance_{inventory.output_number}.txt":
        raise ValueError("rate ledger run-provenance filename differs from its normal output")
    run_record = capture_binding["run_provenance"]
    run_source = run_record.get("source") if isinstance(run_record, Mapping) else None
    bound_path, bound_sha256 = _artifact_path_and_sha256(
        run_source,
        base=_resolve(physics_input.capture_ensemble_path, physics_input.source_path.parent).parent,
        label="registered capture run provenance",
    )
    if run_path != bound_path or run_sha256 != bound_sha256:
        raise ValueError("rate ledger run provenance differs from its registered capture output")
    if run.dark_matter_model != case.physics.dark_matter_model:
        raise ValueError("rate ledger run provenance declares another dark-matter model")
    if (
        run.nstep_coarse != inventory.nstep_coarse
        or not _same_float(run.time_code, inventory.time_code)
        or not _same_float(run.scale_factor, inventory.aexp)
        or _normal_output_directory(run_path) != _normal_output_directory(inventory_path)
    ):
        raise ValueError("rate ledger run provenance differs from its normal-output inventory")
    _require_model_zoom_execution_identity(
        run,
        case=case,
        zoom_manifest_sha256=zoom_manifest_sha256,
        capture_binding=capture_binding,
        capture_ensemble=capture_ensemble,
    )
    for key in ("environment_channels", "rate_points", "diagnostics", "model_evidence"):
        if ledger.get(key) != record.get(key):
            raise ValueError(f"model-specific rate ledger {key} differs from its result")
    return ledger_path, ledger_sha256


def read_resolved_model_physics_result(
    path: str | Path,
    *,
    case: GalaxyMergerZoomCase,
    zoom_manifest_sha256: str,
) -> ResolvedModelPhysicsResult:
    """Read one result only after its three-model physics-input record passes."""

    source = Path(path).expanduser().resolve()
    record = _read_object(source, "resolved model-physics result")
    expected = {
        "schema_version",
        "status",
        "case_id",
        "case",
        "dark_matter_model",
        "zoom_manifest_sha256",
        "capture_event_uid",
        "capture_event_sha256",
        "physics_input_path",
        "physics_input_sha256",
        "rate_ledger_path",
        "rate_ledger_sha256",
        "environment_channels",
        "rate_points",
        "diagnostics",
        "model_evidence",
    }
    if record.get("schema_version") != MODEL_SPECIFIC_PHYSICS_RESULT_SCHEMA_VERSION or set(record) != expected:
        raise ValueError("resolved model-physics result fields are invalid")
    if record.get("status") != "complete":
        raise ValueError("resolved model-physics result is not complete")
    if (
        record.get("case_id") != case.case_id
        or record.get("case") != case.as_dict()
        or record.get("dark_matter_model") != case.physics.dark_matter_model
        or _sha256(record.get("zoom_manifest_sha256"), "zoom_manifest_sha256")
        != _sha256(zoom_manifest_sha256, "zoom_manifest_sha256")
    ):
        raise ValueError("resolved model-physics result provenance does not match its zoom case")
    physics_input_path = _resolve(
        _nonempty(record.get("physics_input_path"), "physics_input_path"), source.parent
    )
    expected_input_sha = _sha256(record.get("physics_input_sha256"), "physics_input_sha256")
    try:
        actual_input_sha = _file_sha256(physics_input_path)
    except OSError as error:
        raise ValueError(f"cannot read physics_input_path: {error}") from error
    if actual_input_sha != expected_input_sha:
        raise ValueError("physics_input SHA-256 differs")
    physics_input = read_dm_comparison_physics_input(physics_input_path)
    if not assess_dm_comparison_physics_inputs(physics_input).ready_for_model_specific_analysis:
        raise ValueError("referenced physics input is not accepted")
    inventory_assessment = _accepted_inventory_assessment(
        physics_input, case.physics.dark_matter_model
    )
    if inventory_assessment.stars_required != (case.physics.host_stellar_mass_msun > 0.0):
        raise ValueError("normal-output inventory stars requirement differs from the zoom case")
    if inventory_assessment.gas_required != (case.physics.gas_fraction > 0.0):
        raise ValueError("normal-output inventory gas requirement differs from the zoom case")
    capture_event_uid = _nonempty(record.get("capture_event_uid"), "capture_event_uid")
    capture_event_sha256 = _sha256(record.get("capture_event_sha256"), "capture_event_sha256")
    capture_ensemble = _registered_capture_ensemble(physics_input)
    capture_binding = _registered_capture_binding(physics_input, case.physics.dark_matter_model)
    if capture_event_uid != _registered_capture_event_uid(physics_input, case.physics.dark_matter_model):
        raise ValueError("result capture_event_uid differs from its registered capture ensemble")
    capture = capture_binding.get("capture_event")
    if not isinstance(capture, Mapping) or capture_event_sha256 != _sha256(
        capture.get("event_sha256"), "registered capture_event_sha256"
    ):
        raise ValueError("result capture_event_sha256 differs from its registered capture ensemble")
    channels = record.get("environment_channels")
    if not isinstance(channels, Mapping) or set(channels) != set(_CHANNELS):
        raise ValueError("resolved environment channels are invalid")
    parsed_channels = tuple(
        (name, ResolvedEnvironmentChannel.from_dict(channels[name], name)) for name in _CHANNELS
    )
    rate_records = record.get("rate_points")
    if not isinstance(rate_records, list):
        raise ValueError("rate_points must be a list")
    points: list[HandoffRatePoint] = []
    for index, item in enumerate(rate_records):
        if not isinstance(item, Mapping) or set(item) != {
            "separation_pc",
            "orbital_power_pc2_myr3",
            "orbital_torque_msun_pc2_myr",
            "eccentricity",
        }:
            raise ValueError(f"rate point {index} fields are invalid")
        try:
            points.append(HandoffRatePoint(**item))
        except (TypeError, ValueError) as error:
            raise ValueError(f"rate point {index} is invalid: {error}") from error
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {
        "maximum_relative_energy_error",
        "maximum_relative_angular_momentum_error",
        "minimum_orbital_resolution_cells",
    }:
        raise ValueError("resolved model-physics diagnostics are invalid")
    model = case.physics.dark_matter_model
    evidence = ModelSpecificEvidence.from_dict(
        record.get("model_evidence"),
        model=model,
        expected_artifacts=_accepted_artifacts(physics_input, model),
    )
    rate_ledger_path, rate_ledger_sha256 = _read_rate_ledger(
        record,
        result_path=source,
        case=case,
        zoom_manifest_sha256=zoom_manifest_sha256,
        physics_input=physics_input,
        inventory_assessment=inventory_assessment,
        capture_binding=capture_binding,
        capture_ensemble=capture_ensemble,
    )
    try:
        return ResolvedModelPhysicsResult(
            case=case,
            zoom_manifest_sha256=_sha256(zoom_manifest_sha256, "zoom_manifest_sha256"),
            source_path=source,
            source_sha256=_file_sha256(source),
            capture_event_uid=capture_event_uid,
            capture_event_sha256=capture_event_sha256,
            physics_input_path=physics_input_path,
            physics_input_sha256=expected_input_sha,
            rate_ledger_path=rate_ledger_path,
            rate_ledger_sha256=rate_ledger_sha256,
            comparison_family_id=capture_ensemble.smoke.preflight.manifest.family_id,
            comparison_shared_input_sha256s=tuple(
                sorted(
                    (
                        name,
                        artifact.sha256,
                    )
                    for name, artifact in capture_ensemble.smoke.preflight.manifest.shared_inputs
                )
            ),
            channels=parsed_channels,
            rate_points=tuple(points),
            maximum_relative_energy_error=diagnostics.get("maximum_relative_energy_error"),
            maximum_relative_angular_momentum_error=diagnostics.get(
                "maximum_relative_angular_momentum_error"
            ),
            minimum_orbital_resolution_cells=diagnostics.get(
                "minimum_orbital_resolution_cells"
            ),
            model_evidence=evidence,
        )
    except ValueError as error:
        raise ValueError(f"resolved model-physics result is invalid: {error}") from error


@dataclass(frozen=True)
class ModelSpecificResolutionAssessment:
    """Resolution-only comparison of one model and one phase realization."""

    reference: ResolvedModelPhysicsResult
    comparison: ResolvedModelPhysicsResult
    status: str
    matched_rate_points: int
    maximum_power_fractional_difference: float | None
    maximum_torque_fractional_difference: float | None
    maximum_eccentricity_difference: float | None
    reasons: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.status == "accepted_model_specific_rates"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "same-model resolution agreement only; this does not compare DM "
                "models, integrate a delay, or declare a coalescence"
            ),
            "dark_matter_model": self.reference.dark_matter_model,
            "reference": {
                "case_id": self.reference.case.case_id,
                "source": str(self.reference.source_path),
                "sha256": self.reference.source_sha256,
            },
            "comparison": {
                "case_id": self.comparison.case.case_id,
                "source": str(self.comparison.source_path),
                "sha256": self.comparison.source_sha256,
            },
            "matched_rate_points": self.matched_rate_points,
            "maximum_power_fractional_difference": self.maximum_power_fractional_difference,
            "maximum_torque_fractional_difference": self.maximum_torque_fractional_difference,
            "maximum_eccentricity_difference": self.maximum_eccentricity_difference,
            "reasons": list(self.reasons),
        }


def _fractional_difference(left: float, right: float) -> float | None:
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return None
    return abs(left - right) / scale


def _match_rate_points(
    reference: tuple[HandoffRatePoint, ...],
    comparison: tuple[HandoffRatePoint, ...],
    *,
    maximum_log_separation_match: float,
) -> tuple[list[tuple[HandoffRatePoint, HandoffRatePoint]], list[str]]:
    reasons: list[str] = []
    low = max(reference[0].separation_pc, comparison[0].separation_pc)
    high = min(reference[-1].separation_pc, comparison[-1].separation_pc)
    if high <= low:
        return [], ["rate supports do not overlap"]
    if high / low < 2.0:
        reasons.append("rate support has insufficient separation width")
    candidates = [point for point in comparison if low <= point.separation_pc <= high]
    pairs: list[tuple[HandoffRatePoint, HandoffRatePoint]] = []
    used: set[int] = set()
    for point in reference:
        if point.separation_pc < low or point.separation_pc > high:
            continue
        distances = [
            (abs(math.log(other.separation_pc / point.separation_pc)), index)
            for index, other in enumerate(candidates)
            if index not in used
        ]
        if not distances:
            continue
        distance, index = min(distances)
        if distance <= maximum_log_separation_match:
            pairs.append((point, candidates[index]))
            used.add(index)
    return pairs, reasons


def compare_model_specific_resolution_pair(
    reference: ResolvedModelPhysicsResult,
    comparison: ResolvedModelPhysicsResult,
    *,
    maximum_conservation_error: float = 1.0e-3,
    minimum_orbital_resolution_cells: float = 4.0,
    minimum_matched_rate_points: int = 3,
    maximum_log_separation_match: float = 0.15,
    maximum_rate_fractional_difference: float = 0.20,
    maximum_eccentricity_difference: float = 0.02,
) -> ModelSpecificResolutionAssessment:
    """Accept measured same-model rates only after conservative paired gates."""

    controls = np.asarray(
        (
            maximum_conservation_error,
            minimum_orbital_resolution_cells,
            maximum_log_separation_match,
            maximum_rate_fractional_difference,
            maximum_eccentricity_difference,
        ),
        dtype=float,
    )
    if (
        np.any(~np.isfinite(controls))
        or np.any(controls <= 0.0)
        or minimum_matched_rate_points < 2
    ):
        raise ValueError("model-specific resolution gates are invalid")
    if reference.case.physics != comparison.case.physics or (
        reference.case.replicate != comparison.case.replicate
    ):
        raise ValueError("resolution pair does not share one model physics point and phase")
    if reference.case.numerics.finest_cell_size_pc >= comparison.case.numerics.finest_cell_size_pc:
        raise ValueError("reference must use a finer cell size than comparison")
    if reference.zoom_manifest_sha256 != comparison.zoom_manifest_sha256:
        raise ValueError("resolution pair does not share one zoom manifest")
    if (
        reference.capture_event_uid != comparison.capture_event_uid
        or reference.capture_event_sha256 != comparison.capture_event_sha256
        or reference.comparison_family_id != comparison.comparison_family_id
        or reference.comparison_shared_input_sha256s
        != comparison.comparison_shared_input_sha256s
    ):
        raise ValueError("resolution pair does not share one capture and comparison family")
    reasons: list[str] = []
    for run, label in ((reference, "reference"), (comparison, "comparison")):
        if run.maximum_relative_energy_error > maximum_conservation_error:
            reasons.append(f"{label} energy error exceeds the gate")
        if run.maximum_relative_angular_momentum_error > maximum_conservation_error:
            reasons.append(f"{label} angular-momentum error exceeds the gate")
        if run.minimum_orbital_resolution_cells < minimum_orbital_resolution_cells:
            reasons.append(f"{label} orbital scale is underresolved")
        if run.dark_matter_model == "fdm":
            assert run.model_evidence.minimum_de_broglie_resolution_cells is not None
            assert run.model_evidence.minimum_wake_resolution_cells is not None
            if run.model_evidence.minimum_de_broglie_resolution_cells < minimum_orbital_resolution_cells:
                reasons.append(f"{label} de Broglie scale is underresolved")
            if run.model_evidence.minimum_wake_resolution_cells < minimum_orbital_resolution_cells:
                reasons.append(f"{label} FDM wake is underresolved")
    pairs, match_reasons = _match_rate_points(
        reference.rate_points,
        comparison.rate_points,
        maximum_log_separation_match=maximum_log_separation_match,
    )
    reasons.extend(match_reasons)
    if len(pairs) < minimum_matched_rate_points:
        reasons.append("too few matched resolved rate points")
    powers: list[float] = []
    torques: list[float] = []
    eccentricities: list[float] = []
    for fine, coarse in pairs:
        power = _fractional_difference(
            fine.orbital_power_pc2_myr3, coarse.orbital_power_pc2_myr3
        )
        torque = _fractional_difference(
            fine.orbital_torque_msun_pc2_myr, coarse.orbital_torque_msun_pc2_myr
        )
        if power is None:
            reasons.append("zero orbital power cannot establish resolution agreement")
        else:
            powers.append(power)
        if torque is None:
            reasons.append("zero orbital torque cannot establish resolution agreement")
        else:
            torques.append(torque)
        if np.sign(fine.orbital_power_pc2_myr3) != np.sign(coarse.orbital_power_pc2_myr3):
            reasons.append("orbital power changes sign across the resolution pair")
        if np.sign(fine.orbital_torque_msun_pc2_myr) != np.sign(coarse.orbital_torque_msun_pc2_myr):
            reasons.append("orbital torque changes sign across the resolution pair")
        eccentricities.append(abs(fine.eccentricity - coarse.eccentricity))
    maximum_power = max(powers, default=None)
    maximum_torque = max(torques, default=None)
    maximum_eccentricity = max(eccentricities, default=None)
    if maximum_power is not None and maximum_power > maximum_rate_fractional_difference:
        reasons.append("orbital power rates disagree beyond the gate")
    if maximum_torque is not None and maximum_torque > maximum_rate_fractional_difference:
        reasons.append("orbital torque rates disagree beyond the gate")
    if maximum_eccentricity is not None and maximum_eccentricity > maximum_eccentricity_difference:
        reasons.append("eccentricity evolution disagrees beyond the gate")
    return ModelSpecificResolutionAssessment(
        reference=reference,
        comparison=comparison,
        status="accepted_model_specific_rates" if not reasons else "censored",
        matched_rate_points=len(pairs),
        maximum_power_fractional_difference=maximum_power,
        maximum_torque_fractional_difference=maximum_torque,
        maximum_eccentricity_difference=maximum_eccentricity,
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True)
class ModelSpecificPhaseEnsemble:
    """Independent phase replicas for one model and exact physics point."""

    members: tuple[ModelSpecificResolutionAssessment, ...]
    status: str
    reasons: tuple[str, ...]

    @property
    def ready_for_separate_model_interpretation(self) -> bool:
        return self.status == "accepted_model_specific_phase_ensemble"

    def as_dict(self) -> dict[str, Any]:
        reference = self.members[0].reference if self.members else None
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "phase-replica acceptance within one DM model only; no cross-model "
                "force substitution or physical coalescence time is inferred"
            ),
            "dark_matter_model": None if reference is None else reference.dark_matter_model,
            "physics_id": None if reference is None else reference.case.physics.physics_id,
            "replicates": [] if reference is None else [
                item.reference.case.replicate for item in self.members
            ],
            "members": [item.as_dict() for item in self.members],
            "reasons": list(self.reasons),
        }


def assess_model_specific_phase_ensemble(
    members: tuple[ModelSpecificResolutionAssessment, ...],
) -> ModelSpecificPhaseEnsemble:
    """Require at least two independent phase replicas before interpretation."""

    if len(members) < 2:
        raise ValueError("model-specific phase ensemble requires at least two members")
    first = members[0].reference
    replicates: set[int] = set()
    reasons: list[str] = []
    for member in members:
        reference = member.reference
        if (
            reference.dark_matter_model != first.dark_matter_model
            or reference.case.physics != first.case.physics
            or reference.zoom_manifest_sha256 != first.zoom_manifest_sha256
            or reference.capture_event_uid != first.capture_event_uid
            or reference.capture_event_sha256 != first.capture_event_sha256
            or reference.comparison_family_id != first.comparison_family_id
            or reference.comparison_shared_input_sha256s
            != first.comparison_shared_input_sha256s
        ):
            raise ValueError("phase ensemble members do not share one capture and model point")
        if reference.case.replicate in replicates:
            raise ValueError("model-specific phase ensemble has a duplicate replicate")
        replicates.add(reference.case.replicate)
        if not member.accepted:
            reasons.extend(
                f"replicate {reference.case.replicate}: {reason}" for reason in member.reasons
            )
    return ModelSpecificPhaseEnsemble(
        members=members,
        status="accepted_model_specific_phase_ensemble" if not reasons else "censored",
        reasons=tuple(reasons),
    )
