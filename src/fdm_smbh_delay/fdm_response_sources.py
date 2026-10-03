"""Byte-verified source files for candidate q/e outer-FDM responses.

This loader verifies only the exact small JSON artifacts named by a manifest.
It does not inspect raw wave fields or certify physical calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

from .fdm_outer_halo import FDMOuterHaloClosure
from .fdm_outer_response import FDMOuterResponseTable
from .fdm_outer_response_family import FDMOuterResponseFamily


_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_LIMIT_BYTES = 1_000_000
_NODE_LIMIT_BYTES = 8_000_000


def _small_file(path: Path, limit_bytes: int) -> bytes:
    try:
        if not path.is_file():
            raise ValueError(f"FDM response source is not a regular file: {path}")
        if path.stat().st_size > limit_bytes:
            raise ValueError(f"FDM response source exceeds {limit_bytes} bytes: {path}")
        with path.open("rb") as stream:
            payload = stream.read(limit_bytes + 1)
    except OSError as error:
        raise ValueError(f"cannot read FDM response source {path}: {error}") from error
    if len(payload) > limit_bytes:
        raise ValueError(f"FDM response source exceeds {limit_bytes} bytes: {path}")
    return payload


def _json_object(payload: bytes, name: str) -> dict:
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name} JSON: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{name} must be a JSON object")
    return record


def _coordinate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"FDM source {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"FDM source {name} must be finite")
    if name == "mass_ratio_q" and not 0.0 < number <= 1.0:
        raise ValueError("FDM source mass_ratio_q lies outside (0, 1]")
    if name == "eccentricity" and not 0.0 <= number < 1.0:
        raise ValueError("FDM source eccentricity lies outside [0, 1)")
    return number


@dataclass(frozen=True)
class FDMResponseSourceNode:
    mass_ratio_q: float
    eccentricity: float
    source_case_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class VerifiedFDMResponseSourceBundle:
    manifest_path: Path
    manifest_sha256: str
    nodes: tuple[FDMResponseSourceNode, ...]
    response_family: FDMOuterResponseFamily
    closure_by_source_sha256: Mapping[str, FDMOuterHaloClosure]
    status: str = "source_files_verified_physics_pending"

    def verify_current_files(self) -> None:
        """Re-hash named files before a new numerical run or restart."""

        if hashlib.sha256(_small_file(self.manifest_path, _MANIFEST_LIMIT_BYTES)).hexdigest() != self.manifest_sha256:
            raise ValueError("FDM response source manifest changed after loading")
        for node in self.nodes:
            if hashlib.sha256(_small_file(node.path, _NODE_LIMIT_BYTES)).hexdigest() != node.sha256:
                raise ValueError(f"FDM response source changed after loading: {node.path}")


def read_verified_fdm_response_sources(
    manifest_path: str | Path,
) -> VerifiedFDMResponseSourceBundle:
    """Verify one explicit rectangular q/e manifest and its small node files.

    Manifest schema 1 has ``status='candidate_outer_fdm_response_sources'``
    and a ``nodes`` array of q/e, path, SHA-256 records.  Each node JSON has
    the same q/e coordinates, a source case ID, one response table, and one
    radial halo closure.  No directory traversal or raw simulation scan is
    performed.  A matching digest proves byte identity, not source physics.
    """

    try:
        manifest = Path(manifest_path).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"cannot resolve FDM response manifest: {error}") from error
    manifest_bytes = _small_file(manifest, _MANIFEST_LIMIT_BYTES)
    record = _json_object(manifest_bytes, "FDM response manifest")
    if set(record) != {"schema_version", "status", "nodes"} or (
        record["schema_version"] != 1
        or record["status"] != "candidate_outer_fdm_response_sources"
        or not isinstance(record["nodes"], list)
        or not record["nodes"]
    ):
        raise ValueError("unsupported FDM response source manifest schema")
    if len(record["nodes"]) > 1000:
        raise ValueError("FDM response source manifest has too many nodes")

    by_coordinate: dict[tuple[float, float], tuple[FDMOuterResponseTable, FDMOuterHaloClosure, FDMResponseSourceNode]] = {}
    for entry in record["nodes"]:
        if not isinstance(entry, dict) or set(entry) != {
            "mass_ratio_q", "eccentricity", "path", "sha256"
        }:
            raise ValueError("FDM response source node manifest entry is invalid")
        q = _coordinate(entry["mass_ratio_q"], "mass_ratio_q")
        e = _coordinate(entry["eccentricity"], "eccentricity")
        if (q, e) in by_coordinate:
            raise ValueError("FDM response source manifest duplicates a q/e node")
        path_text, declared_sha = entry["path"], entry["sha256"]
        if not isinstance(path_text, str) or not path_text.strip():
            raise ValueError("FDM response source node path is required")
        if not isinstance(declared_sha, str) or _SHA256.fullmatch(declared_sha) is None:
            raise ValueError("FDM response source node requires lowercase SHA-256")
        try:
            node_path = (manifest.parent / path_text).resolve(strict=True)
        except OSError as error:
            raise ValueError(f"cannot resolve FDM response node {path_text}: {error}") from error
        node_bytes = _small_file(node_path, _NODE_LIMIT_BYTES)
        if hashlib.sha256(node_bytes).hexdigest() != declared_sha:
            raise ValueError(f"FDM response source SHA-256 mismatch: {node_path}")
        node_record = _json_object(node_bytes, "FDM response node")
        if set(node_record) != {
            "schema_version", "mass_ratio_q", "eccentricity", "source_case_id",
            "response", "closure",
        } or node_record["schema_version"] != 1:
            raise ValueError("unsupported FDM response node schema")
        if (
            _coordinate(node_record["mass_ratio_q"], "mass_ratio_q") != q
            or _coordinate(node_record["eccentricity"], "eccentricity") != e
        ):
            raise ValueError("FDM response node q/e coordinates disagree with manifest")
        case_id = node_record["source_case_id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("FDM response node source_case_id is required")
        table = FDMOuterResponseTable.from_dict(node_record["response"])
        closure = FDMOuterHaloClosure.from_dict(node_record["closure"])
        node = FDMResponseSourceNode(q, e, case_id, node_path, declared_sha)
        by_coordinate[(q, e)] = (table, closure, node)

    q_nodes = tuple(sorted({q for q, _ in by_coordinate}))
    e_nodes = tuple(sorted({e for _, e in by_coordinate}))
    if len(by_coordinate) != len(q_nodes) * len(e_nodes):
        raise ValueError("FDM response source q/e grid is incomplete")
    rows = tuple(tuple(by_coordinate[(q, e)] for e in e_nodes) for q in q_nodes)
    family = FDMOuterResponseFamily(
        mass_ratios_q=q_nodes,
        eccentricities=e_nodes,
        tables=tuple(tuple(cell[0] for cell in row) for row in rows),
        source_sha256=tuple(tuple(cell[2].sha256 for cell in row) for row in rows),
    )
    closures = MappingProxyType({
        cell[2].sha256: cell[1] for row in rows for cell in row
    })
    return VerifiedFDMResponseSourceBundle(
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        nodes=tuple(cell[2] for row in rows for cell in row),
        response_family=family,
        closure_by_source_sha256=closures,
    )
