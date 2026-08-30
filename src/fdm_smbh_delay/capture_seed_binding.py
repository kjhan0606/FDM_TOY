"""Bind a capture-derived sink pair to one materialized pure-FDM seed."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .capture_fdm_seed import (
    CAPTURE_DERIVED_SINK_PAIR_SCHEMA_VERSION,
    CaptureFDMSeedFrame,
    CaptureFDMSeedFrameSpecification,
    CaptureSMBHMassProjection,
    derive_dual_smbh_sink_pair_from_capture,
    verify_mass_projection_source,
)
from .capture_ledger import read_capture_ledger
from .dual_soliton_seed import (
    DualSMBHSinkSeed,
    PureFDMDualSolitonSeed,
    read_materialized_pure_fdm_dual_soliton_seed,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description}: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return record


def _pair_sinks(record: Mapping[str, Any]) -> tuple[DualSMBHSinkSeed, DualSMBHSinkSeed]:
    sinks = record.get("sinks")
    if not isinstance(sinks, list) or len(sinks) != 2:
        raise ValueError("capture sink-pair record requires exactly two sinks")
    try:
        result = tuple(DualSMBHSinkSeed(**sink) for sink in sinks)
    except (TypeError, ValueError) as error:
        raise ValueError(f"capture sink-pair rows are invalid: {error}") from error
    return result  # type: ignore[return-value]


def _capture_pair_record(
    path: Path,
) -> tuple[str, str, CaptureFDMSeedFrame, CaptureSMBHMassProjection, tuple[DualSMBHSinkSeed, DualSMBHSinkSeed]]:
    record = _json_object(path, "capture sink-pair record")
    if (
        record.get("schema_version") != CAPTURE_DERIVED_SINK_PAIR_SCHEMA_VERSION
        or record.get("status") != "capture_binary_dual_smbh_sink_pair_derived"
    ):
        raise ValueError("capture sink-pair record is not a derived binary-pair result")
    capture_event = record.get("capture_event")
    frame = record.get("frame")
    projection = record.get("mass_projection")
    validation = record.get("mass_projection_validation")
    frame_validation = record.get("frame_specification_validation")
    if (
        not isinstance(capture_event, Mapping)
        or not isinstance(frame, Mapping)
        or not isinstance(projection, Mapping)
        or not isinstance(validation, Mapping)
        or not isinstance(frame_validation, Mapping)
    ):
        raise ValueError("capture sink-pair record lacks required provenance")
    event_uid = capture_event.get("event_uid")
    event_sha256 = capture_event.get("event_sha256")
    if not isinstance(event_uid, str) or not event_uid.strip():
        raise ValueError("capture sink-pair event UID is invalid")
    if (
        not isinstance(event_sha256, str)
        or len(event_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in event_sha256)
    ):
        raise ValueError("capture sink-pair event SHA-256 is invalid")
    ledger_path = capture_event.get("ledger_path")
    ledger_sha256 = capture_event.get("ledger_sha256")
    member_ids = capture_event.get("member_ids")
    if (
        not isinstance(ledger_path, str)
        or not ledger_path.strip()
        or not isinstance(ledger_sha256, str)
        or len(ledger_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in ledger_sha256)
        or not isinstance(member_ids, list)
        or len(member_ids) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in member_ids)
        or member_ids[0] == member_ids[1]
    ):
        raise ValueError("capture sink-pair ledger provenance is invalid")
    try:
        parsed_frame = CaptureFDMSeedFrame(**frame)
        parsed_projection = CaptureSMBHMassProjection(**projection)
    except (TypeError, ValueError) as error:
        raise ValueError(f"capture sink-pair provenance is invalid: {error}") from error
    if (
        parsed_projection.event_uid != event_uid
        or tuple(member_ids) != parsed_projection.member_ids
    ):
        raise ValueError("capture sink-pair event and mass-projection identities differ")
    if validation.get("status") != "source_sha256_verified":
        raise ValueError("capture sink-pair mass-projection source was not verified")
    source_path = validation.get("resolved_source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("capture sink-pair lacks a resolved mass-projection source path")
    try:
        frame_source = Path(frame_validation["resolved_source_path"]).expanduser().resolve()
        frame_digest = _sha256(frame_source)
        specification = CaptureFDMSeedFrameSpecification.from_dict(
            _json_object(frame_source, "capture frame specification")
        )
        projection_source = verify_mass_projection_source(
            specification.mass_projection,
            reference_directory=frame_source.parent,
        )
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"cannot re-read capture frame specification: {error}") from error
    if (
        frame_validation.get("status") != "source_sha256_verified"
        or frame_validation.get("sha256") != frame_digest
        or specification.event_uid != event_uid
        or specification.frame != parsed_frame
        or specification.mass_projection != parsed_projection
        or specification.assignment.member_ids != tuple(member_ids)
    ):
        raise ValueError("capture sink-pair frame specification no longer matches")
    if (
        validation.get("status") != "source_sha256_verified"
        or Path(source_path).expanduser().resolve() != projection_source
    ):
        raise ValueError("capture sink-pair mass-projection source no longer matches")
    ledger_source = Path(ledger_path).expanduser().resolve()
    try:
        if _sha256(ledger_source) != ledger_sha256:
            raise ValueError("capture sink-pair ledger SHA-256 no longer matches")
        matches = [
            event for event in read_capture_ledger(ledger_source).events if event.event_uid == event_uid
        ]
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot re-read capture ledger: {error}") from error
    if len(matches) != 1 or matches[0].event_sha256 != event_sha256:
        raise ValueError("capture sink-pair ledger event no longer matches")
    sinks = _pair_sinks(record)
    try:
        expected = derive_dual_smbh_sink_pair_from_capture(
            matches[0],
            frame=specification.frame,
            assignment=specification.assignment,
            mass_projection=specification.mass_projection,
        ).sinks
    except ValueError as error:
        raise ValueError(f"cannot re-derive capture sink-pair rows: {error}") from error
    for index, (sink, derived) in enumerate(zip(sinks, expected), start=1):
        if sink != derived:
            raise ValueError(
                f"capture sink-pair sink {index} does not match the re-derived ledger row"
            )
    return event_uid, event_sha256, parsed_frame, parsed_projection, sinks


def _materialized_seed(path: Path) -> PureFDMDualSolitonSeed:
    try:
        return read_materialized_pure_fdm_dual_soliton_seed(path)
    except ValueError as error:
        raise ValueError(str(error)) from error


@dataclass(frozen=True)
class CaptureSeedMaterializationBinding:
    """Non-submitting identity decision joining capture and local seed inputs."""

    capture_sink_pair_path: Path
    seed_manifest_path: Path
    capture_event_uid: str | None
    capture_event_sha256: str | None
    seed_case_id: str | None
    status: str
    reasons: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return self.status == "capture_seed_materialization_identity_verified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "input identity only; this does not fit solitons, relax the "
                "wave field, certify conservation or resolution, or produce a "
                "physical kpc-to-pc delay"
            ),
            "sources": {
                "capture_sink_pair": {
                    "path": str(self.capture_sink_pair_path),
                    "sha256": _sha256(self.capture_sink_pair_path),
                },
                "seed_manifest": {
                    "path": str(self.seed_manifest_path),
                    "sha256": _sha256(self.seed_manifest_path),
                },
            },
            "capture_event_uid": self.capture_event_uid,
            "capture_event_sha256": self.capture_event_sha256,
            "seed_case_id": self.seed_case_id,
            "reasons": list(self.reasons),
        }


def read_verified_capture_seed_materialization_binding(
    path: str | Path,
) -> CaptureSeedMaterializationBinding:
    """Rebuild a saved capture-to-seed decision from its source artifacts."""

    source = Path(path).expanduser().resolve()
    record = _json_object(source, "capture-to-seed binding")
    expected_fields = {
        "schema_version",
        "status",
        "interpretation",
        "sources",
        "capture_event_uid",
        "capture_event_sha256",
        "seed_case_id",
        "reasons",
    }
    if (
        set(record) != expected_fields
        or record.get("schema_version") != 1
        or record.get("status") != "capture_seed_materialization_identity_verified"
        or record.get("reasons") != []
    ):
        raise ValueError("capture-to-seed binding is not a verified decision")
    sources = record.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "capture_sink_pair",
        "seed_manifest",
    }:
        raise ValueError("capture-to-seed binding sources are invalid")
    paths: dict[str, Path] = {}
    for name in ("capture_sink_pair", "seed_manifest"):
        artifact = sources[name]
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "sha256"}
            or not isinstance(artifact.get("path"), str)
            or not isinstance(artifact.get("sha256"), str)
        ):
            raise ValueError(f"capture-to-seed binding {name} source is invalid")
        resolved = Path(artifact["path"]).expanduser().resolve()
        try:
            actual = _sha256(resolved)
        except OSError as error:
            raise ValueError(f"cannot re-read capture-to-seed {name}: {error}") from error
        if actual != artifact["sha256"]:
            raise ValueError(f"capture-to-seed {name} SHA-256 no longer matches")
        paths[name] = resolved
    decision = assess_capture_seed_materialization_binding(
        capture_sink_pair_path=paths["capture_sink_pair"],
        seed_manifest_path=paths["seed_manifest"],
    )
    if not decision.verified or decision.as_dict() != record:
        raise ValueError("capture-to-seed binding no longer matches its source artifacts")
    return decision


def assess_capture_seed_materialization_binding(
    *,
    capture_sink_pair_path: str | Path,
    seed_manifest_path: str | Path,
) -> CaptureSeedMaterializationBinding:
    """Require every materialized SMBH row to match its capture-derived row."""

    pair_path = Path(capture_sink_pair_path).expanduser().resolve()
    manifest_path = Path(seed_manifest_path).expanduser().resolve()
    reasons: list[str] = []
    event_uid: str | None = None
    event_sha256: str | None = None
    seed_case_id: str | None = None
    pair_sinks: tuple[DualSMBHSinkSeed, DualSMBHSinkSeed] | None = None
    seed: PureFDMDualSolitonSeed | None = None
    try:
        event_uid, event_sha256, _, _, pair_sinks = _capture_pair_record(pair_path)
    except ValueError as error:
        reasons.append(str(error))
    try:
        seed = _materialized_seed(manifest_path)
        seed_case_id = seed.case_id
    except ValueError as error:
        reasons.append(str(error))
    if pair_sinks is not None and seed is not None:
        for index, (expected, actual) in enumerate(zip(pair_sinks, seed.sinks), start=1):
            for name in ("sink_mass_code", "smbh_mass_code", "dark_matter_fraction"):
                if not math.isclose(
                    getattr(expected, name),
                    getattr(actual, name),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-14,
                ):
                    reasons.append(f"seed sink {index} {name} differs from the capture-derived row")
            for name in ("center_box", "velocity_code", "angular_momentum_code"):
                if not np.allclose(
                    getattr(expected, name), getattr(actual, name), rtol=1.0e-12, atol=1.0e-14
                ):
                    reasons.append(f"seed sink {index} {name} differs from the capture-derived row")
    status = (
        "capture_seed_materialization_identity_verified"
        if not reasons
        else "capture_seed_materialization_identity_not_verified"
    )
    return CaptureSeedMaterializationBinding(
        capture_sink_pair_path=pair_path,
        seed_manifest_path=manifest_path,
        capture_event_uid=event_uid,
        capture_event_sha256=event_sha256,
        seed_case_id=seed_case_id,
        status=status,
        reasons=tuple(reasons),
    )
