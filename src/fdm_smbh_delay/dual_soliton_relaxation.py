"""Assess the initial relaxation/conservation window of a dual-soliton seed.

The evidence table is deliberately small and is expected to be produced from
time-resolved FDM outputs during bounded manual postprocessing.  It does not
read full wave fields, run an FFT, or infer an unrecorded conservation law.
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

import numpy as np

from .dual_soliton_preflight import (
    DualSolitonRuntimeIdentity,
    read_verified_pure_fdm_dual_soliton_runtime_identity,
    validate_pure_fdm_dual_soliton_runtime_identity,
)
from .fdm_zoom_runtime_identity import (
    VerifiedFDMZoomRuntimeOutputs,
    read_verified_fdm_declared_zoom_runtime_outputs,
)
from .lagramses_fdm_provenance import read_lagramses_fdm_outer_wave_provenance


DUAL_SOLITON_RELAXATION_SCHEMA_VERSION = 3
DUAL_SOLITON_RELAXATION_SAMPLE_LEDGER_SCHEMA_VERSION = 2
DUAL_SOLITON_RELAXATION_ASSESSMENT_SCHEMA_VERSION = 2
DUAL_SOLITON_RELAXATION_DIAGNOSTIC_PROVENANCE_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_artifact(record: Any, name: str) -> tuple[Path, str]:
    if (
        not isinstance(record, Mapping)
        or set(record) != {"path", "sha256"}
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in record["sha256"])
    ):
        raise ValueError(f"{name} artifact is invalid")
    path = Path(record["path"]).expanduser().resolve()
    try:
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"{name} SHA-256 no longer matches")
    except OSError as error:
        raise ValueError(f"cannot read {name} artifact: {error}") from error
    return path, record["sha256"]


def _write_atomic(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _positive_series(
    value: Any, name: str, samples: int | None = None, *, allow_zero: bool = False
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    invalid = array < 0.0 if allow_zero else array <= 0.0
    qualifier = "non-negative" if allow_zero else "positive"
    if array.ndim != 1 or array.size < 3 or np.any(~np.isfinite(array)) or np.any(invalid):
        raise ValueError(f"{name} must contain at least three finite {qualifier} values")
    if samples is not None and array.shape != (samples,):
        raise ValueError(f"{name} must have {samples} samples")
    return array


def _nonnegative_series(value: Any, name: str, samples: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (samples,) or np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must have {samples} finite non-negative values")
    return array


def _periodic_separation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = first - second
    delta -= np.rint(delta)
    return np.linalg.norm(delta, axis=1)


@dataclass(frozen=True)
class RelaxationConservationThresholds:
    maximum_relative_component_mass_drift: float = 0.02
    maximum_relative_component_radius_drift: float = 0.02
    maximum_relative_wave_mass_error: float = 1.0e-3
    maximum_relative_hamiltonian_error: float = 1.0e-3
    maximum_relative_angular_momentum_error: float = 1.0e-3
    minimum_core_separation_to_radius_ratio: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_relative_component_mass_drift",
            "maximum_relative_component_radius_drift",
            "maximum_relative_wave_mass_error",
            "maximum_relative_hamiltonian_error",
            "maximum_relative_angular_momentum_error",
            "minimum_core_separation_to_radius_ratio",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))

    def as_dict(self) -> dict[str, float]:
        return {
            "maximum_relative_component_mass_drift": self.maximum_relative_component_mass_drift,
            "maximum_relative_component_radius_drift": self.maximum_relative_component_radius_drift,
            "maximum_relative_wave_mass_error": self.maximum_relative_wave_mass_error,
            "maximum_relative_hamiltonian_error": self.maximum_relative_hamiltonian_error,
            "maximum_relative_angular_momentum_error": self.maximum_relative_angular_momentum_error,
            "minimum_core_separation_to_radius_ratio": self.minimum_core_separation_to_radius_ratio,
        }


@dataclass(frozen=True)
class _FileArtifact:
    """One immutable regular file used to identify a solver output set."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser().resolve()
        if not path.is_file():
            raise ValueError("snapshot artifact must be a regular file")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("snapshot artifact SHA-256 is invalid")
        object.__setattr__(self, "path", path)

    @classmethod
    def from_path(cls, path: Path) -> "_FileArtifact":
        resolved = path.expanduser().resolve()
        return cls(path=resolved, sha256=_sha256(resolved))

    @classmethod
    def from_dict(cls, record: Any, name: str) -> "_FileArtifact":
        path, digest = _read_artifact(record, name)
        return cls(path=path, sha256=digest)

    def as_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class DualSolitonRelaxationSample:
    """One discovered raw FDM+AMR source set for a relaxation-table sample."""

    time_code: float
    aexp: float
    nstep_coarse: int
    raw_provenance_path: Path
    raw_provenance_sha256: str
    wave_snapshot_files: tuple[_FileArtifact, ...]
    amr_topology_files: tuple[_FileArtifact, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_code", _finite(self.time_code, "time_code"))
        object.__setattr__(self, "aexp", _finite(self.aexp, "aexp", positive=True))
        if isinstance(self.nstep_coarse, bool) or not isinstance(self.nstep_coarse, int):
            raise ValueError("nstep_coarse must be an integer")
        if self.nstep_coarse < 0:
            raise ValueError("nstep_coarse must be non-negative")
        raw_path = Path(self.raw_provenance_path).expanduser().resolve()
        if (
            not isinstance(self.raw_provenance_sha256, str)
            or len(self.raw_provenance_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.raw_provenance_sha256)
        ):
            raise ValueError("raw FDM provenance SHA-256 is invalid")
        object.__setattr__(self, "raw_provenance_path", raw_path)
        for name in ("wave_snapshot_files", "amr_topology_files"):
            artifacts = tuple(getattr(self, name))
            if not artifacts or not all(isinstance(item, _FileArtifact) for item in artifacts):
                raise ValueError(f"{name} must contain at least one verified file")
            paths = [item.path for item in artifacts]
            if len(paths) != len(set(paths)):
                raise ValueError(f"{name} contains duplicate files")
            object.__setattr__(self, name, artifacts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_code": self.time_code,
            "aexp": self.aexp,
            "nstep_coarse": self.nstep_coarse,
            "raw_fdm_provenance": {
                "path": str(self.raw_provenance_path),
                "sha256": self.raw_provenance_sha256,
            },
            "wave_snapshot_files": [item.as_dict() for item in self.wave_snapshot_files],
            "amr_topology_files": [item.as_dict() for item in self.amr_topology_files],
        }

    @classmethod
    def from_dict(cls, record: Any) -> "DualSolitonRelaxationSample":
        if not isinstance(record, Mapping) or set(record) != {
            "time_code",
            "aexp",
            "nstep_coarse",
            "raw_fdm_provenance",
            "wave_snapshot_files",
            "amr_topology_files",
        }:
            raise ValueError("dual-soliton relaxation sample is invalid")
        raw_path, raw_hash = _read_artifact(
            record["raw_fdm_provenance"], "raw FDM provenance"
        )
        if not isinstance(record["wave_snapshot_files"], list) or not isinstance(
            record["amr_topology_files"], list
        ):
            raise ValueError("dual-soliton relaxation snapshot file sets are invalid")
        return cls(
            time_code=_finite(record["time_code"], "time_code"),
            aexp=_finite(record["aexp"], "aexp", positive=True),
            nstep_coarse=record["nstep_coarse"],
            raw_provenance_path=raw_path,
            raw_provenance_sha256=raw_hash,
            wave_snapshot_files=tuple(
                _FileArtifact.from_dict(item, "wave snapshot")
                for item in record["wave_snapshot_files"]
            ),
            amr_topology_files=tuple(
                _FileArtifact.from_dict(item, "AMR topology")
                for item in record["amr_topology_files"]
            ),
        )


@dataclass(frozen=True)
class DualSolitonRelaxationSampleLedger:
    """Recheckable source identity for every sample in a relaxation table.

    This is a provenance join, not an estimator.  It intentionally does not
    infer core properties or conservation errors from the listed artifacts.
    """

    source_path: Path
    source_sha256: str
    runtime_identity_path: Path
    runtime_identity_sha256: str
    runtime_output_identity_path: Path
    runtime_output_identity_sha256: str
    raw_provenance_path: Path
    raw_provenance_sha256: str
    samples: tuple[DualSolitonRelaxationSample, ...]

    def __post_init__(self) -> None:
        for name in (
            "source_path",
            "runtime_identity_path",
            "runtime_output_identity_path",
            "raw_provenance_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        for name in (
            "source_sha256",
            "runtime_identity_sha256",
            "runtime_output_identity_sha256",
            "raw_provenance_sha256",
        ):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} is invalid")
        if len(self.samples) < 3:
            raise ValueError("dual-soliton relaxation sample ledger requires at least three samples")
        times = np.asarray([sample.time_code for sample in self.samples], dtype=float)
        steps = np.asarray([sample.nstep_coarse for sample in self.samples], dtype=int)
        if np.any(np.diff(times) <= 0.0) or np.any(np.diff(steps) <= 0):
            raise ValueError("relaxation sample times and nstep_coarse must be strictly increasing")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DUAL_SOLITON_RELAXATION_SAMPLE_LEDGER_SCHEMA_VERSION,
            "status": "dual_soliton_relaxation_sample_ledger",
            "interpretation": (
                "per-sample wave/conservation source identity only; this does not "
                "derive relaxation, convergence, or a physical delay"
            ),
            "sources": {
                "runtime_identity": {
                    "path": str(self.runtime_identity_path),
                    "sha256": self.runtime_identity_sha256,
                },
                "runtime_output_identity": {
                    "path": str(self.runtime_output_identity_path),
                    "sha256": self.runtime_output_identity_sha256,
                },
                "raw_fdm_provenance": {
                    "path": str(self.raw_provenance_path),
                    "sha256": self.raw_provenance_sha256,
                },
            },
            "samples": [sample.as_dict() for sample in self.samples],
        }


def _snapshot_file_sets_for_provenance(
    provenance_path: Path,
) -> tuple[tuple[_FileArtifact, ...], tuple[_FileArtifact, ...]]:
    """Enumerate the complete per-CPU FDM and AMR file sets for one output."""

    provenance = read_lagramses_fdm_outer_wave_provenance(provenance_path)
    prefix = provenance.psi_snapshot_prefix
    if not prefix.startswith("fdm_"):
        raise ValueError("psi_snapshot_prefix must begin with fdm_ for AMR binding")
    amr_prefix = "amr_" + prefix.removeprefix("fdm_")
    directory = provenance.source_path.parent
    try:
        children = tuple(directory.iterdir())
    except OSError as error:
        raise ValueError(f"cannot enumerate raw FDM output directory: {error}") from error
    wave_paths = tuple(
        sorted(
            (path for path in children if path.name.startswith(prefix) and path.is_file()),
            key=lambda path: path.name,
        )
    )
    amr_paths = tuple(
        sorted(
            (path for path in children if path.name.startswith(amr_prefix) and path.is_file()),
            key=lambda path: path.name,
        )
    )
    if not wave_paths or not amr_paths:
        raise ValueError("raw FDM output lacks matching FDM or AMR snapshot files")
    wave_suffixes = tuple(path.name.removeprefix(prefix) for path in wave_paths)
    amr_suffixes = tuple(path.name.removeprefix(amr_prefix) for path in amr_paths)
    if wave_suffixes != amr_suffixes:
        raise ValueError("FDM and AMR snapshot shard suffixes do not match")
    try:
        return (
            tuple(_FileArtifact.from_path(path) for path in wave_paths),
            tuple(_FileArtifact.from_path(path) for path in amr_paths),
        )
    except OSError as error:
        raise ValueError(f"cannot hash raw FDM snapshot file set: {error}") from error


def _sample_from_raw_provenance(
    provenance_path: Path,
    *,
    seed_manifest_path: Path,
) -> DualSolitonRelaxationSample:
    identity = validate_pure_fdm_dual_soliton_runtime_identity(
        seed_manifest_path=seed_manifest_path,
        provenance_path=provenance_path,
    )
    if not identity.verified:
        raise ValueError("relaxation sample raw provenance does not match the verified seed")
    provenance = read_lagramses_fdm_outer_wave_provenance(provenance_path)
    wave_files, amr_files = _snapshot_file_sets_for_provenance(provenance_path)
    return DualSolitonRelaxationSample(
        time_code=provenance.time_code,
        aexp=provenance.aexp,
        nstep_coarse=provenance.nstep_coarse,
        raw_provenance_path=provenance.source_path,
        raw_provenance_sha256=provenance.source_sha256,
        wave_snapshot_files=wave_files,
        amr_topology_files=amr_files,
    )


def _read_relaxation_sample_manifest(path: Path) -> tuple[Path, ...]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dual-soliton relaxation sample manifest: {error}") from error
    if (
        not isinstance(record, Mapping)
        or set(record) != {"schema_version", "raw_fdm_provenance_paths"}
        or record.get("schema_version") != 1
        or not isinstance(record.get("raw_fdm_provenance_paths"), list)
    ):
        raise ValueError("dual-soliton relaxation sample manifest is invalid")
    paths: list[Path] = []
    for item in record["raw_fdm_provenance_paths"]:
        if not isinstance(item, str):
            raise ValueError("dual-soliton relaxation raw-provenance path is invalid")
        paths.append(Path(item).expanduser().resolve())
    if len(paths) != len(set(paths)):
        raise ValueError("dual-soliton relaxation sample manifest repeats a raw provenance path")
    return tuple(paths)


def _validate_relaxation_sample_ledger(
    ledger: DualSolitonRelaxationSampleLedger,
    runtime_identity: DualSolitonRuntimeIdentity,
    runtime_outputs: VerifiedFDMZoomRuntimeOutputs,
) -> None:
    expected_seed_path = runtime_outputs.declared_run.fdm_binding.seed_manifest_path
    if (
        runtime_identity.seed_manifest_path != expected_seed_path
        or _sha256(runtime_identity.seed_manifest_path) != _sha256(expected_seed_path)
    ):
        raise ValueError("FDM output-set identity seed manifest differs from runtime identity")
    if runtime_outputs.declared_run.declared.seed_case_id != runtime_identity.seed_case_id:
        raise ValueError("FDM output-set identity seed case differs from runtime identity")
    expected_raw_paths = set(runtime_outputs.raw_fdm_provenance_paths)
    sample_raw_paths = {sample.raw_provenance_path for sample in ledger.samples}
    if sample_raw_paths != expected_raw_paths:
        raise ValueError(
            "relaxation sample ledger raw provenance paths differ from the verified FDM output set"
        )
    if (
        ledger.raw_provenance_path != runtime_identity.provenance_path
        or ledger.raw_provenance_sha256 != _sha256(runtime_identity.provenance_path)
    ):
        raise ValueError(
            "relaxation sample ledger raw provenance differs from the runtime identity"
        )
    provenance = read_lagramses_fdm_outer_wave_provenance(ledger.raw_provenance_path)
    matching_samples = [
        sample for sample in ledger.samples if sample.raw_provenance_path == provenance.source_path
    ]
    if len(matching_samples) != 1:
        raise ValueError(
            "relaxation sample ledger must contain exactly one runtime raw-provenance sample"
        )
    runtime_sample = matching_samples[0]
    if (
        runtime_sample.raw_provenance_sha256 != provenance.source_sha256
        or runtime_sample.nstep_coarse != provenance.nstep_coarse
        or not math.isclose(
            runtime_sample.time_code, provenance.time_code, rel_tol=1.0e-12, abs_tol=1.0e-14
        )
        or not math.isclose(
            runtime_sample.aexp, provenance.aexp, rel_tol=1.0e-12, abs_tol=1.0e-14
        )
    ):
        raise ValueError("runtime raw-provenance sample metadata no longer matches")
    seen_files: set[Path] = set()
    for sample in ledger.samples:
        reconstructed = _sample_from_raw_provenance(
            sample.raw_provenance_path,
            seed_manifest_path=runtime_identity.seed_manifest_path,
        )
        if reconstructed != sample:
            raise ValueError("relaxation sample does not match its raw FDM/AMR output set")
        sample_provenance = read_lagramses_fdm_outer_wave_provenance(
            sample.raw_provenance_path
        )
        if (
            sample_provenance.fdm_first_wave_level != provenance.fdm_first_wave_level
            or sample_provenance.hbar_code != provenance.hbar_code
        ):
            raise ValueError("relaxation sample FDM runtime controls differ across outputs")
        files = [
            *(item.path for item in sample.wave_snapshot_files),
            *(item.path for item in sample.amr_topology_files),
        ]
        if seen_files.intersection(files):
            raise ValueError("relaxation sample ledger reuses a wave or AMR snapshot shard")
        seen_files.update(files)


def read_verified_dual_soliton_relaxation_sample_ledger(
    path: str | Path,
) -> DualSolitonRelaxationSampleLedger:
    """Re-read all artifact hashes and the runtime output that anchors a table."""

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dual-soliton relaxation sample ledger: {error}") from error
    if (
        not isinstance(record, Mapping)
        or set(record) != {"schema_version", "status", "interpretation", "sources", "samples"}
        or record.get("schema_version") != DUAL_SOLITON_RELAXATION_SAMPLE_LEDGER_SCHEMA_VERSION
        or record.get("status") != "dual_soliton_relaxation_sample_ledger"
        or not isinstance(record.get("samples"), list)
    ):
        raise ValueError("dual-soliton relaxation sample ledger is invalid")
    sources = record["sources"]
    if not isinstance(sources, Mapping) or set(sources) != {
        "runtime_identity",
        "runtime_output_identity",
        "raw_fdm_provenance",
    }:
        raise ValueError("dual-soliton relaxation sample ledger sources are invalid")
    identity_path, identity_hash = _read_artifact(sources["runtime_identity"], "runtime identity")
    output_identity_path, output_identity_hash = _read_artifact(
        sources["runtime_output_identity"], "FDM runtime output identity"
    )
    raw_path, raw_hash = _read_artifact(
        sources["raw_fdm_provenance"], "raw FDM provenance"
    )
    samples = tuple(DualSolitonRelaxationSample.from_dict(item) for item in record["samples"])
    ledger = DualSolitonRelaxationSampleLedger(
        source_path=source,
        source_sha256=_sha256(source),
        runtime_identity_path=identity_path,
        runtime_identity_sha256=identity_hash,
        runtime_output_identity_path=output_identity_path,
        runtime_output_identity_sha256=output_identity_hash,
        raw_provenance_path=raw_path,
        raw_provenance_sha256=raw_hash,
        samples=samples,
    )
    runtime_identity = read_verified_pure_fdm_dual_soliton_runtime_identity(identity_path)
    runtime_outputs = read_verified_fdm_declared_zoom_runtime_outputs(output_identity_path)
    _validate_relaxation_sample_ledger(ledger, runtime_identity, runtime_outputs)
    if ledger.as_dict() != record:
        raise ValueError("dual-soliton relaxation sample ledger no longer matches its sources")
    return ledger


def materialize_dual_soliton_relaxation_sample_ledger(
    *,
    runtime_identity_path: str | Path,
    runtime_output_identity_path: str | Path,
    sample_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind a manual relaxation sample manifest to one verified raw FDM output.

    The manifest supplies the actual wave snapshot and conservation-ledger
    files for every table sample.  This function hashes those files but does
    not calculate any physical diagnostic from them.
    """

    identity_path = Path(runtime_identity_path).expanduser().resolve()
    output_identity_path = Path(runtime_output_identity_path).expanduser().resolve()
    manifest_path = Path(sample_manifest_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    runtime_identity = read_verified_pure_fdm_dual_soliton_runtime_identity(identity_path)
    runtime_outputs = read_verified_fdm_declared_zoom_runtime_outputs(output_identity_path)
    raw_provenance_paths = _read_relaxation_sample_manifest(manifest_path)
    samples = tuple(
        _sample_from_raw_provenance(
            path, seed_manifest_path=runtime_identity.seed_manifest_path
        )
        for path in raw_provenance_paths
    )
    ledger = DualSolitonRelaxationSampleLedger(
        source_path=destination,
        source_sha256="0" * 64,
        runtime_identity_path=identity_path,
        runtime_identity_sha256=_sha256(identity_path),
        runtime_output_identity_path=output_identity_path,
        runtime_output_identity_sha256=_sha256(output_identity_path),
        raw_provenance_path=runtime_identity.provenance_path,
        raw_provenance_sha256=_sha256(runtime_identity.provenance_path),
        samples=samples,
    )
    _validate_relaxation_sample_ledger(ledger, runtime_identity, runtime_outputs)
    if destination.exists():
        raise ValueError("dual-soliton relaxation sample-ledger output must not already exist")
    record = ledger.as_dict()
    _write_atomic(destination, record)
    return record


@dataclass(frozen=True)
class DualSolitonRelaxationDiagnostics:
    """Measured relaxation series, separate from the acceptance evidence."""

    sample_times_code: np.ndarray
    component_core_mass_code: np.ndarray
    component_core_radius_box: np.ndarray
    component_centres_box: np.ndarray
    relative_wave_mass_error: np.ndarray
    relative_hamiltonian_error: np.ndarray
    relative_angular_momentum_error: np.ndarray

    def __post_init__(self) -> None:
        times = _positive_series(
            self.sample_times_code, "sample_times_code", allow_zero=True
        )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("sample_times_code must be strictly increasing")
        object.__setattr__(self, "sample_times_code", times)
        samples = int(times.size)
        for name in ("component_core_mass_code", "component_core_radius_box"):
            array = np.asarray(getattr(self, name), dtype=float)
            if (
                array.shape != (samples, 2)
                or np.any(~np.isfinite(array))
                or np.any(array <= 0.0)
            ):
                raise ValueError(f"{name} must have shape ({samples}, 2) and positive values")
            object.__setattr__(self, name, array)
        centres = np.asarray(self.component_centres_box, dtype=float)
        if centres.shape != (samples, 2, 3) or np.any(~np.isfinite(centres)):
            raise ValueError(f"component_centres_box must have shape ({samples}, 2, 3)")
        if np.any(centres < 0.0) or np.any(centres >= 1.0):
            raise ValueError("component_centres_box must lie in [0, 1)")
        object.__setattr__(self, "component_centres_box", centres)
        for name in (
            "relative_wave_mass_error",
            "relative_hamiltonian_error",
            "relative_angular_momentum_error",
        ):
            object.__setattr__(
                self, name, _nonnegative_series(getattr(self, name), name, samples)
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_times_code": self.sample_times_code.tolist(),
            "component_core_mass_code": self.component_core_mass_code.tolist(),
            "component_core_radius_box": self.component_core_radius_box.tolist(),
            "component_centres_box": self.component_centres_box.tolist(),
            "relative_wave_mass_error": self.relative_wave_mass_error.tolist(),
            "relative_hamiltonian_error": self.relative_hamiltonian_error.tolist(),
            "relative_angular_momentum_error": self.relative_angular_momentum_error.tolist(),
        }

    @classmethod
    def from_dict(cls, record: Any) -> "DualSolitonRelaxationDiagnostics":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError("dual-soliton relaxation diagnostics are invalid")
        try:
            return cls(**record)
        except (TypeError, ValueError) as error:
            raise ValueError(f"dual-soliton relaxation diagnostics are invalid: {error}") from error


@dataclass(frozen=True)
class DualSolitonRelaxationDiagnosticProvenance:
    """A recheckable extractor declaration and its measured relaxation series."""

    source_path: Path
    source_sha256: str
    sample_ledger_path: Path
    sample_ledger_sha256: str
    extractor_path: Path
    extractor_sha256: str
    extractor_version: str
    diagnostics: DualSolitonRelaxationDiagnostics

    def __post_init__(self) -> None:
        for name in ("source_path", "sample_ledger_path", "extractor_path"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        for name in ("source_sha256", "sample_ledger_sha256", "extractor_sha256"):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} is invalid")
        if not isinstance(self.extractor_version, str) or not self.extractor_version.strip():
            raise ValueError("extractor_version is required")
        if not isinstance(self.diagnostics, DualSolitonRelaxationDiagnostics):
            raise ValueError("diagnostics are invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DUAL_SOLITON_RELAXATION_DIAGNOSTIC_PROVENANCE_SCHEMA_VERSION,
            "status": "dual_soliton_relaxation_diagnostic_provenance",
            "interpretation": (
                "declared extractor and source-bound diagnostic series only; this does "
                "not by itself attest extractor execution, convergence, or a physical delay"
            ),
            "sources": {
                "sample_ledger": {
                    "path": str(self.sample_ledger_path),
                    "sha256": self.sample_ledger_sha256,
                },
                "extractor": {
                    "path": str(self.extractor_path),
                    "sha256": self.extractor_sha256,
                },
            },
            "extractor_version": self.extractor_version,
            "diagnostics": self.diagnostics.as_dict(),
        }


def _read_diagnostic_manifest(
    path: Path,
) -> tuple[Path, str, DualSolitonRelaxationDiagnostics]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dual-soliton diagnostic manifest: {error}") from error
    if not isinstance(record, Mapping) or set(record) != {
        "schema_version",
        "extractor_path",
        "extractor_version",
        "diagnostics",
    } or record.get("schema_version") != 1 or not isinstance(record.get("extractor_path"), str) or not isinstance(record.get("extractor_version"), str):
        raise ValueError("dual-soliton diagnostic manifest is invalid")
    extractor_path = Path(record["extractor_path"]).expanduser().resolve()
    try:
        extractor = _FileArtifact.from_path(extractor_path)
    except OSError as error:
        raise ValueError(f"cannot hash dual-soliton diagnostic extractor: {error}") from error
    return (
        extractor.path,
        record["extractor_version"],
        DualSolitonRelaxationDiagnostics.from_dict(record["diagnostics"]),
    )


def _validate_diagnostic_provenance(
    provenance: DualSolitonRelaxationDiagnosticProvenance,
    sample_ledger: DualSolitonRelaxationSampleLedger,
) -> None:
    if (
        provenance.sample_ledger_path != sample_ledger.source_path
        or provenance.sample_ledger_sha256 != sample_ledger.source_sha256
    ):
        raise ValueError("relaxation diagnostics differ from the verified sample ledger")
    ledger_times = np.asarray([sample.time_code for sample in sample_ledger.samples], dtype=float)
    if provenance.diagnostics.sample_times_code.shape != ledger_times.shape or not np.allclose(
        provenance.diagnostics.sample_times_code, ledger_times, rtol=1.0e-12, atol=1.0e-14
    ):
        raise ValueError("relaxation diagnostic times differ from the verified sample ledger")


def read_verified_dual_soliton_relaxation_diagnostic_provenance(
    path: str | Path,
) -> DualSolitonRelaxationDiagnosticProvenance:
    """Re-read source files and diagnostics before using a relaxation table."""

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dual-soliton diagnostic provenance: {error}") from error
    if (
        not isinstance(record, Mapping)
        or set(record) != {
            "schema_version",
            "status",
            "interpretation",
            "sources",
            "extractor_version",
            "diagnostics",
        }
        or record.get("schema_version") != DUAL_SOLITON_RELAXATION_DIAGNOSTIC_PROVENANCE_SCHEMA_VERSION
        or record.get("status") != "dual_soliton_relaxation_diagnostic_provenance"
        or not isinstance(record.get("extractor_version"), str)
    ):
        raise ValueError("dual-soliton diagnostic provenance is invalid")
    sources = record["sources"]
    if not isinstance(sources, Mapping) or set(sources) != {"sample_ledger", "extractor"}:
        raise ValueError("dual-soliton diagnostic provenance sources are invalid")
    sample_ledger_path, sample_ledger_hash = _read_artifact(
        sources["sample_ledger"], "relaxation sample ledger"
    )
    extractor_path, extractor_hash = _read_artifact(sources["extractor"], "diagnostic extractor")
    provenance = DualSolitonRelaxationDiagnosticProvenance(
        source_path=source,
        source_sha256=_sha256(source),
        sample_ledger_path=sample_ledger_path,
        sample_ledger_sha256=sample_ledger_hash,
        extractor_path=extractor_path,
        extractor_sha256=extractor_hash,
        extractor_version=record["extractor_version"],
        diagnostics=DualSolitonRelaxationDiagnostics.from_dict(record["diagnostics"]),
    )
    sample_ledger = read_verified_dual_soliton_relaxation_sample_ledger(sample_ledger_path)
    _validate_diagnostic_provenance(provenance, sample_ledger)
    if provenance.as_dict() != record:
        raise ValueError("dual-soliton diagnostic provenance no longer matches its sources")
    return provenance


def materialize_dual_soliton_relaxation_diagnostic_provenance(
    *,
    sample_ledger_path: str | Path,
    diagnostic_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind a diagnostic-series manifest to fixed sources and extractor bytes."""

    ledger_path = Path(sample_ledger_path).expanduser().resolve()
    manifest_path = Path(diagnostic_manifest_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    sample_ledger = read_verified_dual_soliton_relaxation_sample_ledger(ledger_path)
    extractor_path, extractor_version, diagnostics = _read_diagnostic_manifest(manifest_path)
    provenance = DualSolitonRelaxationDiagnosticProvenance(
        source_path=destination,
        source_sha256="0" * 64,
        sample_ledger_path=ledger_path,
        sample_ledger_sha256=_sha256(ledger_path),
        extractor_path=extractor_path,
        extractor_sha256=_sha256(extractor_path),
        extractor_version=extractor_version,
        diagnostics=diagnostics,
    )
    _validate_diagnostic_provenance(provenance, sample_ledger)
    if destination.exists():
        raise ValueError("dual-soliton diagnostic-provenance output must not already exist")
    record = provenance.as_dict()
    _write_atomic(destination, record)
    return record


@dataclass(frozen=True)
class DualSolitonRelaxationEvidence:
    """An initial window tied to verified sources and a diagnostic provenance."""

    seed_case_id: str
    runtime_identity_path: Path
    sample_ledger_path: Path
    diagnostic_provenance_path: Path
    relaxation_window_start_code: float
    thresholds: RelaxationConservationThresholds

    def __post_init__(self) -> None:
        if not isinstance(self.seed_case_id, str) or not self.seed_case_id.strip():
            raise ValueError("seed_case_id is required")
        object.__setattr__(
            self,
            "runtime_identity_path",
            Path(self.runtime_identity_path).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "sample_ledger_path",
            Path(self.sample_ledger_path).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "diagnostic_provenance_path",
            Path(self.diagnostic_provenance_path).expanduser().resolve(),
        )
        start = _finite(self.relaxation_window_start_code, "relaxation_window_start_code")
        object.__setattr__(self, "relaxation_window_start_code", start)
        if not isinstance(self.thresholds, RelaxationConservationThresholds):
            raise ValueError("thresholds must be a relaxation/conservation threshold record")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DUAL_SOLITON_RELAXATION_SCHEMA_VERSION,
            "seed_case_id": self.seed_case_id,
            "runtime_identity_path": str(self.runtime_identity_path),
            "sample_ledger_path": str(self.sample_ledger_path),
            "diagnostic_provenance_path": str(self.diagnostic_provenance_path),
            "relaxation_window_start_code": self.relaxation_window_start_code,
            "thresholds": self.thresholds.as_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "DualSolitonRelaxationEvidence":
        if not isinstance(record, Mapping) or record.get("schema_version") != DUAL_SOLITON_RELAXATION_SCHEMA_VERSION:
            raise ValueError("unsupported dual-soliton relaxation evidence schema")
        expected = set(cls.__dataclass_fields__) | {"schema_version"}
        unknown = set(record) - expected
        if unknown:
            raise ValueError(f"dual-soliton relaxation evidence has unknown fields: {sorted(unknown)}")
        thresholds = record.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("dual-soliton relaxation evidence lacks thresholds")
        try:
            return cls(
                seed_case_id=record.get("seed_case_id"),
                runtime_identity_path=record.get("runtime_identity_path"),
                sample_ledger_path=record.get("sample_ledger_path"),
                diagnostic_provenance_path=record.get("diagnostic_provenance_path"),
                relaxation_window_start_code=record.get("relaxation_window_start_code"),
                thresholds=RelaxationConservationThresholds(**thresholds),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"dual-soliton relaxation evidence is invalid: {error}") from error


@dataclass(frozen=True)
class DualSolitonRelaxationAssessment:
    source_path: Path
    source_sha256: str
    runtime_identity_path: Path
    runtime_identity_sha256: str
    sample_ledger_path: Path
    sample_ledger_sha256: str
    diagnostic_provenance_path: Path
    diagnostic_provenance_sha256: str
    seed_case_id: str
    status: str
    metrics: dict[str, float]
    reasons: tuple[str, ...]

    @property
    def conditionally_within_thresholds(self) -> bool:
        """Whether declared, source-bound diagnostics meet the stated thresholds.

        This remains conditional until solver output-set and extractor-execution
        attestations exist; it is deliberately not a scientific relaxation pass.
        """

        return self.status == "relaxation_conservation_declared_series_within_thresholds"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DUAL_SOLITON_RELAXATION_ASSESSMENT_SCHEMA_VERSION,
            "status": self.status,
            "interpretation": (
                "source-bound declared diagnostic series only; solver output-set and "
                "extractor-execution attestations remain required before a relaxation "
                "or conservation pass. This is neither an outer-merger calibration nor "
                "a physical coalescence delay"
            ),
            "source": {"path": str(self.source_path), "sha256": self.source_sha256},
            "runtime_identity": {
                "path": str(self.runtime_identity_path),
                "sha256": self.runtime_identity_sha256,
            },
            "sample_ledger": {
                "path": str(self.sample_ledger_path),
                "sha256": self.sample_ledger_sha256,
            },
            "diagnostic_provenance": {
                "path": str(self.diagnostic_provenance_path),
                "sha256": self.diagnostic_provenance_sha256,
            },
            "seed_case_id": self.seed_case_id,
            "metrics": self.metrics,
            "reasons": list(self.reasons),
        }


def _verified_runtime_identity(
    path: Path, seed_case_id: str
) -> DualSolitonRuntimeIdentity:
    try:
        identity = read_verified_pure_fdm_dual_soliton_runtime_identity(path)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"relaxation evidence requires a verified runtime seed identity: {error}"
        ) from error
    if identity.seed_case_id != seed_case_id:
        raise ValueError("runtime identity seed_case_id does not match relaxation evidence")
    return identity


def _verified_sample_ledger(
    path: Path,
    runtime_identity_path: Path,
) -> DualSolitonRelaxationSampleLedger:
    try:
        ledger = read_verified_dual_soliton_relaxation_sample_ledger(path)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"relaxation evidence requires a verified sample ledger: {error}"
        ) from error
    if ledger.runtime_identity_path != runtime_identity_path:
        raise ValueError("relaxation sample ledger runtime identity differs from relaxation evidence")
    return ledger


def _verified_diagnostic_provenance(
    path: Path,
    sample_ledger: DualSolitonRelaxationSampleLedger,
) -> DualSolitonRelaxationDiagnosticProvenance:
    try:
        provenance = read_verified_dual_soliton_relaxation_diagnostic_provenance(path)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"relaxation evidence requires verified diagnostic provenance: {error}"
        ) from error
    _validate_diagnostic_provenance(provenance, sample_ledger)
    return provenance


def assess_dual_soliton_relaxation(
    evidence_path: str | Path,
) -> DualSolitonRelaxationAssessment:
    """Gate an explicitly selected initial window without reading live outputs."""

    source = Path(evidence_path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dual-soliton relaxation evidence: {error}") from error
    evidence = DualSolitonRelaxationEvidence.from_dict(record)
    _verified_runtime_identity(evidence.runtime_identity_path, evidence.seed_case_id)
    sample_ledger = _verified_sample_ledger(
        evidence.sample_ledger_path,
        evidence.runtime_identity_path,
    )
    diagnostic_provenance = _verified_diagnostic_provenance(
        evidence.diagnostic_provenance_path, sample_ledger
    )
    diagnostics = diagnostic_provenance.diagnostics
    window = diagnostics.sample_times_code >= evidence.relaxation_window_start_code
    if int(np.count_nonzero(window)) < 3:
        raise ValueError("relaxation window must contain at least three diagnostic samples")
    masses = diagnostics.component_core_mass_code[window]
    radii = diagnostics.component_core_radius_box[window]
    mass_drift = np.max(np.abs(masses / masses[0] - 1.0))
    radius_drift = np.max(np.abs(radii / radii[0] - 1.0))
    separations = _periodic_separation(
        diagnostics.component_centres_box[window, 0],
        diagnostics.component_centres_box[window, 1],
    )
    core_scale = np.maximum(radii[:, 0], radii[:, 1])
    minimum_separation_ratio = float(np.min(separations / core_scale))
    metrics = {
        "maximum_relative_component_mass_drift": float(mass_drift),
        "maximum_relative_component_radius_drift": float(radius_drift),
        "maximum_relative_wave_mass_error": float(
            np.max(diagnostics.relative_wave_mass_error[window])
        ),
        "maximum_relative_hamiltonian_error": float(
            np.max(diagnostics.relative_hamiltonian_error[window])
        ),
        "maximum_relative_angular_momentum_error": float(
            np.max(diagnostics.relative_angular_momentum_error[window])
        ),
        "minimum_core_separation_to_radius_ratio": minimum_separation_ratio,
    }
    thresholds = evidence.thresholds
    reasons: list[str] = []
    for name in (
        "maximum_relative_component_mass_drift",
        "maximum_relative_component_radius_drift",
        "maximum_relative_wave_mass_error",
        "maximum_relative_hamiltonian_error",
        "maximum_relative_angular_momentum_error",
    ):
        if metrics[name] > getattr(thresholds, name):
            reasons.append(f"{name} exceeds the declared relaxation threshold")
    if minimum_separation_ratio < thresholds.minimum_core_separation_to_radius_ratio:
        reasons.append("two core centres enter the declared relaxation separation guard")
    return DualSolitonRelaxationAssessment(
        source_path=source,
        source_sha256=_sha256(source),
        runtime_identity_path=evidence.runtime_identity_path,
        runtime_identity_sha256=_sha256(evidence.runtime_identity_path),
        sample_ledger_path=evidence.sample_ledger_path,
        sample_ledger_sha256=sample_ledger.source_sha256,
        diagnostic_provenance_path=evidence.diagnostic_provenance_path,
        diagnostic_provenance_sha256=diagnostic_provenance.source_sha256,
        seed_case_id=evidence.seed_case_id,
        status=(
            "relaxation_conservation_declared_series_within_thresholds"
            if not reasons
            else "relaxation_conservation_declared_series_outside_thresholds"
        ),
        metrics=metrics,
        reasons=tuple(reasons),
    )
