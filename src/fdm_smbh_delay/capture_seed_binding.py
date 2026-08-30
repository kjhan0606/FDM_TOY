"""Bind a capture-derived sink pair to one materialized pure-FDM seed."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .capture_fdm_seed import CaptureFDMSeedFrame, CaptureSMBHMassProjection
from .dual_soliton_seed import (
    DualSMBHSinkSeed,
    PureFDMDualSolitonSeed,
    pure_fdm_dual_soliton_seed_from_mapping,
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
        record.get("schema_version") != 1
        or record.get("status") != "capture_binary_dual_smbh_sink_pair_derived"
    ):
        raise ValueError("capture sink-pair record is not a derived binary-pair result")
    capture_event = record.get("capture_event")
    frame = record.get("frame")
    projection = record.get("mass_projection")
    validation = record.get("mass_projection_validation")
    if (
        not isinstance(capture_event, Mapping)
        or not isinstance(frame, Mapping)
        or not isinstance(projection, Mapping)
        or not isinstance(validation, Mapping)
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
    try:
        parsed_frame = CaptureFDMSeedFrame(**frame)
        parsed_projection = CaptureSMBHMassProjection(**projection)
    except (TypeError, ValueError) as error:
        raise ValueError(f"capture sink-pair provenance is invalid: {error}") from error
    member_ids = capture_event.get("member_ids")
    if (
        parsed_projection.event_uid != event_uid
        or not isinstance(member_ids, list)
        or tuple(member_ids) != parsed_projection.member_ids
    ):
        raise ValueError("capture sink-pair event and mass-projection identities differ")
    if validation.get("status") != "source_sha256_verified":
        raise ValueError("capture sink-pair mass-projection source was not verified")
    source_path = validation.get("resolved_source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("capture sink-pair lacks a resolved mass-projection source path")
    try:
        source_digest = _sha256(Path(source_path).expanduser().resolve())
    except OSError as error:
        raise ValueError(f"cannot re-read mass-projection source: {error}") from error
    if source_digest.lower() != parsed_projection.source_sha256.lower():
        raise ValueError("capture sink-pair mass-projection source hash no longer matches")
    sinks = _pair_sinks(record)
    for sink, mass_msun in zip(sinks, parsed_projection.smbh_masses_msun):
        if not math.isclose(
            sink.smbh_mass_code * parsed_frame.mass_unit_msun,
            mass_msun,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("capture sink-pair SMBH row does not match its mass projection")
    return event_uid, event_sha256, parsed_frame, parsed_projection, sinks


def _expected_ic_sink_rows(seed: PureFDMDualSolitonSeed) -> np.ndarray:
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


def _materialized_ic_sink_rows(path: Path) -> np.ndarray:
    try:
        fields = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as error:
        raise ValueError(f"cannot read materialized ic_sink: {error}") from error
    if len(fields) != 2 or any(len(row) != 12 for row in fields):
        raise ValueError("materialized ic_sink must have exactly two twelve-field rows")
    try:
        result = np.asarray(
            [[float(value.replace("D", "E").replace("d", "e")) for value in row] for row in fields],
            dtype=float,
        )
    except ValueError as error:
        raise ValueError(f"materialized ic_sink is not numeric: {error}") from error
    if np.any(~np.isfinite(result)):
        raise ValueError("materialized ic_sink contains non-finite values")
    return result


def _materialized_seed(path: Path) -> PureFDMDualSolitonSeed:
    record = _json_object(path, "dual-soliton seed manifest")
    if (
        record.get("schema_version") != 1
        or record.get("status") != "pure_fdm_dual_soliton_seed_materialized"
    ):
        raise ValueError("dual-soliton manifest is not a materialized seed")
    seed_record = record.get("seed")
    if not isinstance(seed_record, dict):
        raise ValueError("dual-soliton manifest lacks seed data")
    if record.get("seed_sha256") != _canonical_sha256(seed_record):
        raise ValueError("dual-soliton manifest seed SHA-256 does not match")
    canonical_seed = dict(seed_record)
    policy = canonical_seed.pop("solver_policy", None)
    if policy != {
        "fdm_use_hjm": False,
        "fdm_outer_ledger": True,
        "analytic_fdm_drag": False,
        "initial_state": "coherent_dual_soliton_seed_requires_relaxation",
    }:
        raise ValueError("dual-soliton manifest solver policy is invalid")
    try:
        seed = pure_fdm_dual_soliton_seed_from_mapping(canonical_seed)
    except ValueError as error:
        raise ValueError(f"dual-soliton manifest seed is invalid: {error}") from error
    files = record.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("dual-soliton manifest lacks materialized file provenance")
    sink_file = files.get("ic_sink")
    if (
        not isinstance(sink_file, Mapping)
        or not isinstance(sink_file.get("path"), str)
        or not isinstance(sink_file.get("sha256"), str)
    ):
        raise ValueError("dual-soliton manifest lacks ic_sink provenance")
    sink_path = Path(sink_file["path"]).expanduser().resolve()
    try:
        actual_hash = _sha256(sink_path)
    except OSError as error:
        raise ValueError(f"cannot read materialized ic_sink: {error}") from error
    if actual_hash != sink_file["sha256"]:
        raise ValueError("materialized ic_sink SHA-256 does not match the seed manifest")
    if not np.allclose(
        _materialized_ic_sink_rows(sink_path),
        _expected_ic_sink_rows(seed),
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("materialized ic_sink rows do not match the seed manifest")
    return seed


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
