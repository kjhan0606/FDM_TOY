"""Bind a capture-derived all-wave seed to one declared FDM zoom case.

The binding is intentionally an input-identity gate.  It cannot attest that a
solver consumed the seed, relaxed the two solitons, or resolved a wake; those
claims require an executed-output provenance record and later wave diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .capture_seed_binding import (
    CaptureSeedMaterializationBinding,
    read_verified_capture_seed_materialization_binding,
)
from .dual_soliton_seed import (
    PureFDMDualSolitonSeed,
    read_materialized_pure_fdm_dual_soliton_seed,
)
from .model_zoom_materialization import (
    VerifiedModelZoomExecutionContract,
    read_verified_model_zoom_execution_contract,
)


FDM_CAPTURE_SEED_ZOOM_BINDING_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str | None]:
    try:
        digest: str | None = _sha256(path)
    except OSError:
        digest = None
    return {"path": str(path), "sha256": digest}


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _expected_channel_status(present: bool) -> str:
    return "available" if present else "absent"


@dataclass(frozen=True)
class FDMCaptureSeedZoomBinding:
    """Recheckable declaration joining one capture seed and FDM zoom case."""

    model_zoom_contract_path: Path
    capture_seed_binding_path: Path
    seed_manifest_path: Path
    capture_event_uid: str | None
    capture_event_sha256: str | None
    zoom_case_id: str | None
    seed_case_id: str | None
    status: str
    reasons: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return self.status == "fdm_capture_seed_zoom_identity_verified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FDM_CAPTURE_SEED_ZOOM_BINDING_SCHEMA_VERSION,
            "status": self.status,
            "interpretation": (
                "declared capture-to-seed-to-FDM-zoom input identity only; this does "
                "not attest solver consumption, soliton relaxation, conservation, "
                "wave/wake resolution, an outer-to-inner handoff, or a physical delay"
            ),
            "sources": {
                "model_zoom_execution_contract": _artifact(self.model_zoom_contract_path),
                "capture_seed_binding": _artifact(self.capture_seed_binding_path),
                "seed_manifest": _artifact(self.seed_manifest_path),
            },
            "capture_event_uid": self.capture_event_uid,
            "capture_event_sha256": self.capture_event_sha256,
            "zoom_case_id": self.zoom_case_id,
            "seed_case_id": self.seed_case_id,
            "reasons": list(self.reasons),
        }


def _assess(
    contract: VerifiedModelZoomExecutionContract,
    capture_seed: CaptureSeedMaterializationBinding,
    seed: PureFDMDualSolitonSeed,
    *,
    reasons: list[str],
) -> None:
    case = contract.case
    physics = case.physics
    numerics = case.numerics
    if physics.dark_matter_model != "fdm":
        reasons.append("model-zoom contract does not select an FDM case")
        return
    if (
        capture_seed.capture_event_uid != contract.capture_event.event_uid
        or capture_seed.capture_event_sha256 != contract.capture_event.event_sha256
    ):
        reasons.append("capture-to-seed event identity differs from the FDM zoom contract")
    expected_manifest = contract.shared_inputs["smbh_seed_catalog"]
    if (
        str(capture_seed.seed_manifest_path) != expected_manifest["path"]
        or _sha256(capture_seed.seed_manifest_path) != expected_manifest["sha256"]
    ):
        reasons.append("FDM zoom sink-input identity does not name the verified seed manifest")
    if seed.dark_matter_model != "fdm":
        reasons.append("materialized seed is not pure FDM")
    if not math.isclose(
        seed.m_axion_ev,
        float(physics.fdm_particle_mass_ev),
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        reasons.append("materialized seed axion mass differs from the FDM zoom case")
    if numerics.fdm_use_hjm is not False:
        reasons.append("materialized all-wave seed requires fdm_use_hjm=false in the FDM zoom case")
    if numerics.fdm_first_wave_level is None:
        reasons.append("FDM zoom case must explicitly declare fdm_first_wave_level")
    expected_stars = _expected_channel_status(physics.host_stellar_mass_msun > 0.0)
    if seed.stellar_status != expected_stars:
        reasons.append(
            "materialized seed stellar status differs from the FDM zoom baryonic declaration"
        )
    expected_gas = _expected_channel_status(physics.gas_fraction > 0.0)
    if seed.gas_status != expected_gas:
        reasons.append(
            "materialized seed gas status differs from the FDM zoom baryonic declaration"
        )


def assess_fdm_capture_seed_zoom_binding(
    *,
    model_zoom_contract_path: str | Path,
    capture_seed_binding_path: str | Path,
) -> FDMCaptureSeedZoomBinding:
    """Return a fail-closed FDM seed/zoom declaration decision."""

    contract_path = Path(model_zoom_contract_path).expanduser().resolve()
    capture_path = Path(capture_seed_binding_path).expanduser().resolve()
    seed_path = Path(".").resolve()
    event_uid: str | None = None
    event_sha256: str | None = None
    zoom_case_id: str | None = None
    seed_case_id: str | None = None
    reasons: list[str] = []
    contract: VerifiedModelZoomExecutionContract | None = None
    capture_seed: CaptureSeedMaterializationBinding | None = None
    seed: PureFDMDualSolitonSeed | None = None
    try:
        contract = read_verified_model_zoom_execution_contract(contract_path)
        event_uid = contract.capture_event.event_uid
        event_sha256 = contract.capture_event.event_sha256
        zoom_case_id = contract.case.case_id
    except (OSError, ValueError) as error:
        reasons.append(str(error))
    try:
        capture_seed = read_verified_capture_seed_materialization_binding(capture_path)
        seed_path = capture_seed.seed_manifest_path
        seed_case_id = capture_seed.seed_case_id
    except (OSError, ValueError) as error:
        reasons.append(str(error))
    if capture_seed is not None:
        try:
            seed = read_materialized_pure_fdm_dual_soliton_seed(seed_path)
            seed_case_id = seed.case_id
        except (OSError, ValueError) as error:
            reasons.append(str(error))
    if contract is not None and capture_seed is not None and seed is not None:
        _assess(contract, capture_seed, seed, reasons=reasons)
    status = (
        "fdm_capture_seed_zoom_identity_verified"
        if not reasons
        else "fdm_capture_seed_zoom_identity_not_verified"
    )
    return FDMCaptureSeedZoomBinding(
        model_zoom_contract_path=contract_path,
        capture_seed_binding_path=capture_path,
        seed_manifest_path=seed_path,
        capture_event_uid=event_uid,
        capture_event_sha256=event_sha256,
        zoom_case_id=zoom_case_id,
        seed_case_id=seed_case_id,
        status=status,
        reasons=tuple(reasons),
    )


def read_verified_fdm_capture_seed_zoom_binding(
    path: str | Path,
) -> FDMCaptureSeedZoomBinding:
    """Rebuild a saved FDM capture-seed-zoom decision from its sources."""

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read FDM capture-seed-zoom binding: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "interpretation",
        "sources",
        "capture_event_uid",
        "capture_event_sha256",
        "zoom_case_id",
        "seed_case_id",
        "reasons",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_fields
        or record.get("schema_version") != FDM_CAPTURE_SEED_ZOOM_BINDING_SCHEMA_VERSION
        or record.get("status") != "fdm_capture_seed_zoom_identity_verified"
        or record.get("reasons") != []
    ):
        raise ValueError("FDM capture-seed-zoom binding is not a verified decision")
    sources = record.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "model_zoom_execution_contract",
        "capture_seed_binding",
        "seed_manifest",
    }:
        raise ValueError("FDM capture-seed-zoom binding sources are invalid")
    resolved: dict[str, Path] = {}
    for name in ("model_zoom_execution_contract", "capture_seed_binding", "seed_manifest"):
        artifact = sources[name]
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "sha256"}
            or not isinstance(artifact.get("path"), str)
            or not isinstance(artifact.get("sha256"), str)
        ):
            raise ValueError(f"FDM capture-seed-zoom {name} source is invalid")
        candidate = Path(artifact["path"]).expanduser().resolve()
        try:
            if _sha256(candidate) != artifact["sha256"]:
                raise ValueError(f"FDM capture-seed-zoom {name} SHA-256 no longer matches")
        except OSError as error:
            raise ValueError(f"cannot re-read FDM capture-seed-zoom {name}: {error}") from error
        resolved[name] = candidate
    decision = assess_fdm_capture_seed_zoom_binding(
        model_zoom_contract_path=resolved["model_zoom_execution_contract"],
        capture_seed_binding_path=resolved["capture_seed_binding"],
    )
    if (
        not decision.verified
        or decision.seed_manifest_path != resolved["seed_manifest"]
        or decision.as_dict() != record
    ):
        raise ValueError("FDM capture-seed-zoom binding no longer matches its source artifacts")
    return decision


def materialize_fdm_capture_seed_zoom_binding(
    *,
    model_zoom_contract_path: str | Path,
    capture_seed_binding_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write one non-submitting FDM capture-to-seed-to-zoom decision."""

    decision = assess_fdm_capture_seed_zoom_binding(
        model_zoom_contract_path=model_zoom_contract_path,
        capture_seed_binding_path=capture_seed_binding_path,
    )
    destination = Path(output_directory).expanduser().resolve()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("FDM capture-seed-zoom output directory must not already exist") from error
    record = decision.as_dict()
    _write_atomic(
        destination / "fdm_capture_seed_zoom_binding.json",
        json.dumps(record, indent=2, sort_keys=True) + "\n",
    )
    return record
