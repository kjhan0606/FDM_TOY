"""Execution records for a bounded dual-soliton relaxation extractor.

The wrapper in :mod:`scripts.run_dual_soliton_relaxation_extractor` is the
only producer of the execution-attestation record.  It runs an operator
supplied extractor with ``shell=False`` and a one-thread numerical environment,
then verifies the extractor's JSON result against the already verified sample
ledger before publishing the attestation.  The record is a wrapper-declared
record of the command and returned bytes; it does not provide an OS-level proof
or prove the extractor's internal physical calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .dual_soliton_relaxation import (
    DualSolitonRelaxationDiagnostics,
    read_verified_dual_soliton_relaxation_sample_ledger,
)


EX_TEMPFAIL = 75
EXECUTOR_RESULT_SCHEMA_VERSION = 1
EXECUTOR_ATTESTATION_SCHEMA_VERSION = 1
EXECUTOR_RESULT_STATUS = "dual_soliton_relaxation_extractor_result"
EXECUTOR_ATTESTATION_STATUS = "dual_soliton_relaxation_extractor_executed"
EXECUTOR_INTERPRETATION = (
    "wrapper-declared record of one shell-free extractor command and its "
    "source-bound JSON output; this does not attest the extractor's internal "
    "physical method"
)

# These values are part of the execution contract.  They prevent a normal
# numerical-library thread fan-out when this wrapper is used on Lageunha.
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

# Keep the child environment deterministic without persisting secrets or
# inheriting arbitrary scheduler/library toggles.  The selected values are
# stored in the attestation alongside the mandatory one-thread overrides.
PASSTHROUGH_ENVIRONMENT = (
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "HOME",
    "LANG",
    "LC_ALL",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, name: str) -> tuple[Path, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    try:
        return resolved, _sha256(resolved)
    except OSError as error:
        raise ValueError(f"cannot hash {name}: {error}") from error


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
    path, digest = _artifact(Path(record["path"]), name)
    if digest != record["sha256"]:
        raise ValueError(f"{name} SHA-256 no longer matches")
    return path, digest


def _atomic_create_json(path: Path, record: Mapping[str, Any]) -> None:
    """Publish a JSON file without replacing a concurrent operator output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        # link(2) is atomic and fails if the destination appeared while the
        # extractor was running; os.replace would silently overwrite it.
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(f"output already exists: {path}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    return value


def _command(value: Any, name: str = "extractor command") -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must contain at least one token")
    return tuple(_string(item, f"{name} token") for item in value)


def _resolve_command(
    command: Sequence[str], working_directory: Path
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    """Resolve the executable and every existing file token in ``command``.

    Hashing file tokens catches the usual ``python extractor.py`` form while
    retaining support for compiled executables and command options.  Tokens
    that are not files (for example ``-m`` or a module name) are deliberately
    not guessed as source files; the operator should pass the script path as a
    command token when its bytes need to be bound.
    """

    if not command:
        raise ValueError("extractor command must not be empty")
    cwd = working_directory.expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError("extractor working directory must be a directory")
    first_token = command[0]
    first = Path(first_token).expanduser()
    if first.is_absolute() or "/" in first_token or "\\" in first_token:
        if not first.is_absolute():
            first = cwd / first
        executable = first.resolve()
    else:
        # Resolve bare commands exactly as the child will: relative PATH
        # entries are interpreted from the recorded child working directory.
        path_value = os.environ.get("PATH", "")
        executable = None
        for entry in path_value.split(os.pathsep):
            base = cwd if not entry or entry == "." else Path(entry).expanduser()
            if not base.is_absolute():
                base = cwd / base
            candidate = (base / first_token).resolve()
            if candidate.is_file() and os.access(candidate, os.X_OK):
                executable = candidate
                break
        if executable is None:
            found = shutil.which(first_token)
            if found is None:
                raise ValueError(f"extractor executable cannot be resolved: {first_token}")
            executable = Path(found).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("extractor command's first token is not executable")
    paths: list[Path] = [executable]
    for token in command[1:]:
        candidate = Path(token).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in paths:
                paths.append(resolved)
    return (str(executable), *command[1:]), tuple(paths)


def _resolve_command_files(
    command: Sequence[str], working_directory: Path
) -> tuple[Path, ...]:
    """Return command-file paths for an already resolved command."""

    resolved_command, paths = _resolve_command(command, working_directory)
    if tuple(command) != resolved_command:
        raise ValueError("extractor command must use its resolved absolute executable")
    return paths


def _command_file_records(paths: Sequence[Path]) -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []
    for path in paths:
        resolved, digest = _artifact(path, "extractor command file")
        records.append({"path": str(resolved), "sha256": digest})
    return tuple(records)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _child_environment() -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in PASSTHROUGH_ENVIRONMENT
        if key in os.environ
    }
    environment.update(THREAD_ENVIRONMENT)
    return environment


def _parse_utc(value: Any, name: str) -> datetime:
    text = _string(value, name)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must carry a UTC offset")
    return parsed.astimezone(timezone.utc)


def _validate_result(
    path: Path,
    *,
    expected_ledger_path: Path,
    expected_ledger_sha256: str,
) -> "DualSolitonRelaxationExtractorResult":
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read extractor result: {error}") from error
    if (
        not isinstance(record, Mapping)
        or set(record)
        != {
            "schema_version",
            "status",
            "sample_ledger",
            "extractor_version",
            "diagnostics",
        }
        or record.get("schema_version") != EXECUTOR_RESULT_SCHEMA_VERSION
        or record.get("status") != EXECUTOR_RESULT_STATUS
    ):
        raise ValueError("dual-soliton extractor result is invalid")
    ledger_path, ledger_hash = _read_artifact(
        record["sample_ledger"], "extractor-result sample ledger"
    )
    if ledger_path != expected_ledger_path or ledger_hash != expected_ledger_sha256:
        raise ValueError("extractor result names a different sample ledger")
    version = _string(record.get("extractor_version"), "extractor_version")
    diagnostics = DualSolitonRelaxationDiagnostics.from_dict(record["diagnostics"])
    ledger = read_verified_dual_soliton_relaxation_sample_ledger(expected_ledger_path)
    ledger_times = [sample.time_code for sample in ledger.samples]
    if diagnostics.sample_times_code.shape != (len(ledger_times),):
        raise ValueError("extractor diagnostic sample count differs from sample ledger")
    if not all(
        abs(float(actual) - float(expected)) <= max(1.0e-14, abs(float(expected)) * 1.0e-12)
        for actual, expected in zip(diagnostics.sample_times_code, ledger_times)
    ):
        raise ValueError("extractor diagnostic times differ from sample ledger")
    return DualSolitonRelaxationExtractorResult(
        source_path=path,
        source_sha256=_sha256(path),
        sample_ledger_path=ledger_path,
        sample_ledger_sha256=ledger_hash,
        extractor_version=version,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class DualSolitonRelaxationExtractorResult:
    """A parsed result emitted by the bounded extractor command."""

    source_path: Path
    source_sha256: str
    sample_ledger_path: Path
    sample_ledger_sha256: str
    extractor_version: str
    diagnostics: DualSolitonRelaxationDiagnostics


@dataclass(frozen=True)
class DualSolitonRelaxationExtractorAttestation:
    """Recheckable wrapper-declared command execution and result identity."""

    source_path: Path
    source_sha256: str
    sample_ledger_path: Path
    sample_ledger_sha256: str
    result_path: Path
    result_sha256: str
    execution_result_path: Path
    command: tuple[str, ...]
    argv: tuple[str, ...]
    command_files: tuple[tuple[Path, str], ...]
    working_directory: Path
    started_utc: str
    finished_utc: str
    hostname: str
    environment: Mapping[str, str]
    extractor_version: str
    diagnostics: DualSolitonRelaxationDiagnostics

    def __post_init__(self) -> None:
        for name in (
            "source_path",
            "sample_ledger_path",
            "result_path",
            "execution_result_path",
            "working_directory",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        for name in ("source_sha256", "sample_ledger_sha256", "result_sha256"):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} is invalid")
        object.__setattr__(self, "command", _command(self.command))
        object.__setattr__(self, "argv", _command(self.argv, "extractor argv"))
        if tuple(self.argv) != (
            *self.command,
            str(self.sample_ledger_path),
            str(self.execution_result_path),
        ):
            raise ValueError("extractor argv does not match command and bound paths")
        if self.execution_result_path == self.result_path:
            raise ValueError("private execution result path must differ from published result")
        if not self.working_directory.is_dir():
            raise ValueError("extractor working directory is not a directory")
        files: list[tuple[Path, str]] = []
        for item in self.command_files:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("extractor command file record is invalid")
            path, digest = _artifact(Path(item[0]), "extractor command file")
            if not isinstance(item[1], str) or digest != item[1]:
                raise ValueError("extractor command file SHA-256 no longer matches")
            files.append((path, digest))
        expected = _resolve_command_files(self.command, self.working_directory)
        if tuple(path for path, _ in files) != expected:
            raise ValueError("extractor command file set differs from command")
        object.__setattr__(self, "command_files", tuple(files))
        object.__setattr__(self, "environment", dict(self.environment))
        for key, value in THREAD_ENVIRONMENT.items():
            if self.environment.get(key) != value:
                raise ValueError("extractor thread environment is not the one-thread contract")
        if any(key not in (*PASSTHROUGH_ENVIRONMENT, *THREAD_ENVIRONMENT) for key in self.environment):
            raise ValueError("extractor environment contains an unbound variable")
        object.__setattr__(self, "started_utc", _parse_utc(self.started_utc, "started_utc").isoformat())
        object.__setattr__(self, "finished_utc", _parse_utc(self.finished_utc, "finished_utc").isoformat())
        if _parse_utc(self.finished_utc, "finished_utc") < _parse_utc(
            self.started_utc, "started_utc"
        ):
            raise ValueError("finished_utc precedes started_utc")
        object.__setattr__(self, "hostname", _string(self.hostname, "hostname"))
        object.__setattr__(self, "extractor_version", _string(self.extractor_version, "extractor_version"))
        if not isinstance(self.diagnostics, DualSolitonRelaxationDiagnostics):
            raise ValueError("diagnostics are invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTOR_ATTESTATION_SCHEMA_VERSION,
            "status": EXECUTOR_ATTESTATION_STATUS,
            "interpretation": EXECUTOR_INTERPRETATION,
            "sources": {
                "sample_ledger": {
                    "path": str(self.sample_ledger_path),
                    "sha256": self.sample_ledger_sha256,
                },
                "extractor_result": {
                    "path": str(self.result_path),
                    "sha256": self.result_sha256,
                },
            },
            "extractor": {
                "command": list(self.command),
                "argv": list(self.argv),
                "command_files": [
                    {"path": str(path), "sha256": digest}
                    for path, digest in self.command_files
                ],
                "execution_result_path": str(self.execution_result_path),
                "working_directory": str(self.working_directory),
                "started_utc": self.started_utc,
                "finished_utc": self.finished_utc,
                "hostname": self.hostname,
                "environment": dict(self.environment),
                "version": self.extractor_version,
            },
        }


class ExtractorExecutionError(RuntimeError):
    """The command did not produce a consumable, successful result."""

    def __init__(self, message: str, returncode: int = EX_TEMPFAIL) -> None:
        super().__init__(message)
        self.returncode = returncode


def run_dual_soliton_relaxation_extractor(
    *,
    sample_ledger_path: str | Path,
    result_path: str | Path,
    attestation_path: str | Path,
    extractor_command: Sequence[str],
    working_directory: str | Path | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run and attest one extractor without shell expansion or thread fan-out.

    The command receives two appended positional arguments: the verified
    sample-ledger path and the result JSON path.  It must write the result with
    :data:`EXECUTOR_RESULT_STATUS`; no attestation is written on any failure.
    """

    ledger_path = Path(sample_ledger_path).expanduser().resolve()
    result = Path(result_path).expanduser().resolve()
    attestation = Path(attestation_path).expanduser().resolve()
    if len({ledger_path, result, attestation}) != 3:
        raise ValueError("sample ledger, result, and attestation paths must differ")
    if result.exists():
        raise ValueError("extractor result output must not already exist")
    if attestation.exists():
        raise ValueError("extractor attestation output must not already exist")
    ledger = read_verified_dual_soliton_relaxation_sample_ledger(ledger_path)
    ledger_hash = _sha256(ledger_path)
    cwd = (
        Path(working_directory).expanduser().resolve()
        if working_directory is not None
        else Path.cwd().resolve()
    )
    requested_command = _command(extractor_command)
    command, command_files = _resolve_command(requested_command, cwd)
    command_file_records = _command_file_records(command_files)
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive when supplied")
    result.parent.mkdir(parents=True, exist_ok=True)
    private_run_directory = Path(
        tempfile.mkdtemp(prefix=f".{result.name}.run-", dir=result.parent)
    )
    execution_result = private_run_directory / "extractor-result.json"
    argv = (*command, str(ledger_path), str(execution_result))
    environment = _child_environment()
    started = _utc_now()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=environment,
            check=False,
            shell=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        shutil.rmtree(private_run_directory, ignore_errors=True)
        raise ExtractorExecutionError("extractor timed out", EX_TEMPFAIL) from error
    except OSError as error:
        shutil.rmtree(private_run_directory, ignore_errors=True)
        raise ExtractorExecutionError(f"extractor could not be started: {error}") from error
    try:
        if completed.returncode != 0:
            raise ExtractorExecutionError(
                f"extractor exited with return code {completed.returncode}",
                completed.returncode if isinstance(completed.returncode, int) else EX_TEMPFAIL,
            )
        if not execution_result.is_file():
            raise ExtractorExecutionError("extractor exited successfully without a result JSON")
        try:
            parsed = _validate_result(
                execution_result,
                expected_ledger_path=ledger.source_path,
                expected_ledger_sha256=ledger_hash,
            )
            # Reject a source swap that persisted through the child run.  The
            # command's pre-run snapshot is retained in the attestation.
            if _command_file_records(command_files) != command_file_records:
                raise ExtractorExecutionError("extractor command file changed during execution")
            execution_result_hash = _sha256(execution_result)
            os.link(execution_result, result)
            result_hash = _sha256(result)
        except FileExistsError as error:
            raise ExtractorExecutionError("extractor result output appeared during execution") from error
        except (OSError, ValueError) as error:
            if isinstance(error, ExtractorExecutionError):
                raise
            raise ExtractorExecutionError(f"extractor result is not consumable: {error}") from error
        if result_hash != execution_result_hash:
            raise ExtractorExecutionError("published extractor result hash differs from private result")
        finished = _utc_now()
        record = DualSolitonRelaxationExtractorAttestation(
            source_path=attestation,
            source_sha256="0" * 64,
            sample_ledger_path=ledger.source_path,
            sample_ledger_sha256=ledger_hash,
            result_path=result,
            result_sha256=result_hash,
            execution_result_path=execution_result,
            command=command,
            argv=argv,
            command_files=tuple(
                (Path(item["path"]), item["sha256"]) for item in command_file_records
            ),
            working_directory=cwd,
            started_utc=started,
            finished_utc=finished,
            hostname=socket.gethostname(),
            environment=environment,
            extractor_version=parsed.extractor_version,
            diagnostics=parsed.diagnostics,
        )
        _atomic_create_json(attestation, record.as_dict())
        return record.as_dict()
    finally:
        shutil.rmtree(private_run_directory, ignore_errors=True)


def read_verified_dual_soliton_relaxation_extractor_attestation(
    path: str | Path,
) -> DualSolitonRelaxationExtractorAttestation:
    """Re-hash all command/result/ledger sources and validate the full record."""

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read extractor attestation: {error}") from error
    expected = {"schema_version", "status", "interpretation", "sources", "extractor"}
    if (
        not isinstance(record, Mapping)
        or set(record) != expected
        or record.get("schema_version") != EXECUTOR_ATTESTATION_SCHEMA_VERSION
        or record.get("status") != EXECUTOR_ATTESTATION_STATUS
        or record.get("interpretation") != EXECUTOR_INTERPRETATION
    ):
        raise ValueError("dual-soliton extractor attestation is invalid")
    sources = record["sources"]
    if not isinstance(sources, Mapping) or set(sources) != {"sample_ledger", "extractor_result"}:
        raise ValueError("extractor attestation sources are invalid")
    ledger_path, ledger_hash = _read_artifact(sources["sample_ledger"], "sample ledger")
    result_path, result_hash = _read_artifact(sources["extractor_result"], "extractor result")
    extractor = record["extractor"]
    if not isinstance(extractor, Mapping) or set(extractor) != {
        "command",
        "argv",
        "command_files",
        "execution_result_path",
        "working_directory",
        "started_utc",
        "finished_utc",
        "hostname",
        "environment",
        "version",
    }:
        raise ValueError("extractor execution metadata is invalid")
    command = _command(extractor["command"])
    argv = _command(extractor["argv"], "extractor argv")
    execution_result_path = Path(
        _string(extractor["execution_result_path"], "execution_result_path")
    ).expanduser().resolve()
    cwd = Path(_string(extractor["working_directory"], "working_directory")).expanduser().resolve()
    command_files_record = extractor["command_files"]
    if not isinstance(command_files_record, list):
        raise ValueError("extractor command files are invalid")
    command_files: list[tuple[Path, str]] = []
    for item in command_files_record:
        item_path, item_hash = _read_artifact(item, "extractor command file")
        command_files.append((item_path, item_hash))
    environment = extractor["environment"]
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("extractor environment is invalid")
    ledger = read_verified_dual_soliton_relaxation_sample_ledger(ledger_path)
    parsed_result = _validate_result(
        result_path,
        expected_ledger_path=ledger_path,
        expected_ledger_sha256=ledger_hash,
    )
    if parsed_result.source_sha256 != result_hash:
        raise ValueError("extractor result hash changed during attestation validation")
    attestation = DualSolitonRelaxationExtractorAttestation(
        source_path=source,
        source_sha256=_sha256(source),
        sample_ledger_path=ledger_path,
        sample_ledger_sha256=ledger_hash,
        result_path=result_path,
        result_sha256=result_hash,
        execution_result_path=execution_result_path,
        command=command,
        argv=argv,
        command_files=tuple(command_files),
        working_directory=cwd,
        started_utc=_string(extractor["started_utc"], "started_utc"),
        finished_utc=_string(extractor["finished_utc"], "finished_utc"),
        hostname=_string(extractor["hostname"], "hostname"),
        environment=dict(environment),
        extractor_version=_string(extractor["version"], "extractor version"),
        diagnostics=parsed_result.diagnostics,
    )
    if attestation.extractor_version != parsed_result.extractor_version:
        raise ValueError("attestation extractor version differs from result")
    if attestation.as_dict() != record:
        raise ValueError("extractor attestation no longer matches its sources")
    return attestation


EXECUTED_DIAGNOSTIC_PROVENANCE_SCHEMA_VERSION = 1
EXECUTED_DIAGNOSTIC_PROVENANCE_STATUS = (
    "dual_soliton_relaxation_executed_diagnostic_provenance"
)
EXECUTED_DIAGNOSTIC_PROVENANCE_INTERPRETATION = (
    "execution-attested extractor output bound to a verified sample ledger; this "
    "does not validate the extractor's internal physical method, convergence, or "
    "a physical delay"
)


@dataclass(frozen=True)
class DualSolitonRelaxationExecutedDiagnosticProvenance:
    """A diagnostic result whose command execution is independently recorded."""

    source_path: Path
    source_sha256: str
    sample_ledger_path: Path
    sample_ledger_sha256: str
    extractor_attestation_path: Path
    extractor_attestation_sha256: str
    extractor_result_path: Path
    extractor_result_sha256: str
    extractor_version: str
    diagnostics: DualSolitonRelaxationDiagnostics

    def __post_init__(self) -> None:
        for name in (
            "source_path",
            "sample_ledger_path",
            "extractor_attestation_path",
            "extractor_result_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        for name in (
            "source_sha256",
            "sample_ledger_sha256",
            "extractor_attestation_sha256",
            "extractor_result_sha256",
        ):
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
            "schema_version": EXECUTED_DIAGNOSTIC_PROVENANCE_SCHEMA_VERSION,
            "status": EXECUTED_DIAGNOSTIC_PROVENANCE_STATUS,
            "interpretation": EXECUTED_DIAGNOSTIC_PROVENANCE_INTERPRETATION,
            "sources": {
                "sample_ledger": {
                    "path": str(self.sample_ledger_path),
                    "sha256": self.sample_ledger_sha256,
                },
                "extractor_attestation": {
                    "path": str(self.extractor_attestation_path),
                    "sha256": self.extractor_attestation_sha256,
                },
                "extractor_result": {
                    "path": str(self.extractor_result_path),
                    "sha256": self.extractor_result_sha256,
                },
            },
            "extractor_version": self.extractor_version,
            "diagnostics": self.diagnostics.as_dict(),
        }


def materialize_dual_soliton_relaxation_executed_diagnostic_provenance(
    *,
    sample_ledger_path: str | Path,
    extractor_attestation_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind a verified extractor execution to its sample ledger and result."""

    ledger_path = Path(sample_ledger_path).expanduser().resolve()
    attestation_path = Path(extractor_attestation_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    ledger = read_verified_dual_soliton_relaxation_sample_ledger(ledger_path)
    attestation = read_verified_dual_soliton_relaxation_extractor_attestation(
        attestation_path
    )
    if (
        attestation.sample_ledger_path != ledger.source_path
        or attestation.sample_ledger_sha256 != _sha256(ledger_path)
    ):
        raise ValueError("extractor attestation names a different sample ledger")
    provenance = DualSolitonRelaxationExecutedDiagnosticProvenance(
        source_path=destination,
        source_sha256="0" * 64,
        sample_ledger_path=ledger_path,
        sample_ledger_sha256=_sha256(ledger_path),
        extractor_attestation_path=attestation_path,
        extractor_attestation_sha256=_sha256(attestation_path),
        extractor_result_path=attestation.result_path,
        extractor_result_sha256=attestation.result_sha256,
        extractor_version=attestation.extractor_version,
        diagnostics=attestation.diagnostics,
    )
    if destination.exists():
        raise ValueError("executed diagnostic-provenance output must not already exist")
    record = provenance.as_dict()
    _atomic_create_json(destination, record)
    return record


def read_verified_dual_soliton_relaxation_executed_diagnostic_provenance(
    path: str | Path,
) -> DualSolitonRelaxationExecutedDiagnosticProvenance:
    """Re-read the ledger, attestation, result, and diagnostic arrays."""

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read executed diagnostic provenance: {error}") from error
    if (
        not isinstance(record, Mapping)
        or set(record)
        != {
            "schema_version",
            "status",
            "interpretation",
            "sources",
            "extractor_version",
            "diagnostics",
        }
        or record.get("schema_version") != EXECUTED_DIAGNOSTIC_PROVENANCE_SCHEMA_VERSION
        or record.get("status") != EXECUTED_DIAGNOSTIC_PROVENANCE_STATUS
        or record.get("interpretation") != EXECUTED_DIAGNOSTIC_PROVENANCE_INTERPRETATION
        or not isinstance(record.get("extractor_version"), str)
    ):
        raise ValueError("executed diagnostic provenance is invalid")
    sources = record["sources"]
    if not isinstance(sources, Mapping) or set(sources) != {
        "sample_ledger",
        "extractor_attestation",
        "extractor_result",
    }:
        raise ValueError("executed diagnostic provenance sources are invalid")
    ledger_path, ledger_hash = _read_artifact(sources["sample_ledger"], "sample ledger")
    attestation_path, attestation_hash = _read_artifact(
        sources["extractor_attestation"], "extractor attestation"
    )
    result_path, result_hash = _read_artifact(sources["extractor_result"], "extractor result")
    ledger = read_verified_dual_soliton_relaxation_sample_ledger(ledger_path)
    attestation = read_verified_dual_soliton_relaxation_extractor_attestation(
        attestation_path
    )
    if (
        attestation.sample_ledger_path != ledger_path
        or attestation.sample_ledger_sha256 != ledger_hash
        or attestation.result_path != result_path
        or attestation.result_sha256 != result_hash
    ):
        raise ValueError("executed diagnostic provenance sources differ from attestation")
    result = _validate_result(
        result_path,
        expected_ledger_path=ledger_path,
        expected_ledger_sha256=ledger_hash,
    )
    provenance = DualSolitonRelaxationExecutedDiagnosticProvenance(
        source_path=source,
        source_sha256=_sha256(source),
        sample_ledger_path=ledger_path,
        sample_ledger_sha256=ledger_hash,
        extractor_attestation_path=attestation_path,
        extractor_attestation_sha256=attestation_hash,
        extractor_result_path=result_path,
        extractor_result_sha256=result_hash,
        extractor_version=str(record["extractor_version"]),
        diagnostics=DualSolitonRelaxationDiagnostics.from_dict(record["diagnostics"]),
    )
    if (
        provenance.extractor_version != attestation.extractor_version
        or provenance.extractor_version != result.extractor_version
        or provenance.diagnostics.as_dict() != attestation.diagnostics.as_dict()
        or provenance.diagnostics.as_dict() != result.diagnostics.as_dict()
    ):
        raise ValueError("executed diagnostic provenance differs from extractor result")
    ledger_times = [sample.time_code for sample in ledger.samples]
    if provenance.diagnostics.sample_times_code.shape != (len(ledger_times),):
        raise ValueError("executed diagnostic sample count differs from sample ledger")
    if any(
        abs(float(actual) - float(expected))
        > max(1.0e-14, abs(float(expected)) * 1.0e-12)
        for actual, expected in zip(provenance.diagnostics.sample_times_code, ledger_times)
    ):
        raise ValueError("executed diagnostic times differ from sample ledger")
    if provenance.as_dict() != record:
        raise ValueError("executed diagnostic provenance no longer matches its sources")
    return provenance
