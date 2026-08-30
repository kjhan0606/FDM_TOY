"""Extract a provenance-bound raw relative-orbit track from lagRamses outputs.

The output is intentionally an instantaneous relative-state series, not a
monotonic decay-rate table.  Eccentric binaries need an orbit-averaging stage
before their separation evolution can be integrated into a CDM delay.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .capture_ledger import read_capture_ledger
from .cdm_zoom import assess_noncompacting_cdm_zoom_run
from .dm_run_provenance import read_dark_matter_run_provenance


LAGRAMSES_CDM_PAIR_TRACK_SCHEMA_VERSION = 1
_OUTPUT_DIRECTORY = re.compile(r"output_(\d{5})$")
_PC_CGS = 3.0856775814913673e18
_MYR_SECONDS = 3.15576e13


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _number(value: str, label: str, *, positive: bool = False) -> float:
    try:
        number = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return number


def _info_records(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read lagRamses info file: {error}") from error
    records: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = (item.strip() for item in line.split("=", 1))
        if key in {"time", "aexp", "unit_l", "unit_t", "boxlen"}:
            if key in records:
                raise ValueError(f"lagRamses info file duplicates {key}")
            records[key] = value
    for key in ("time", "aexp", "unit_l", "unit_t", "boxlen"):
        if key not in records:
            raise ValueError(f"lagRamses info file omits {key}")
    return records


@dataclass(frozen=True)
class SinkState:
    sink_id: int
    mass_code: float
    position_code: tuple[float, float, float]
    velocity_code: tuple[float, float, float]


def _sink_states(path: Path) -> dict[int, SinkState]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read lagRamses sink CSV: {error}") from error
    states: dict[int, SinkState] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 10:
            raise ValueError(f"sink CSV line {line_number} must have 10 fields")
        try:
            sink_id = int(fields[0])
        except ValueError as error:
            raise ValueError(f"sink CSV line {line_number} has an invalid sink ID") from error
        if sink_id <= 0 or sink_id in states:
            raise ValueError(f"sink CSV line {line_number} has a duplicate or invalid sink ID")
        values = tuple(_number(value, f"sink CSV line {line_number}") for value in fields[1:8])
        states[sink_id] = SinkState(
            sink_id=sink_id,
            mass_code=values[0],
            position_code=values[1:4],
            velocity_code=values[4:7],
        )
    return states


def _minimum_image(delta: float, boxlen: float) -> float:
    """Use the half-open [-L/2, L/2) convention for periodic separation."""

    return delta - boxlen * math.floor(delta / boxlen + 0.5)


def _resolve(reference: str, base: Path) -> Path:
    path = Path(reference).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def read_bound_cdm_capture(
    path: str | Path,
    *,
    capture_event_uid: str,
    primary_sink_id: int,
    secondary_sink_id: int,
) -> dict[str, Any]:
    """Read one validated original CDM capture bound to a retained zoom pair.

    This is deliberately shared by the input materializer and the later raw
    orbit extractor.  Both stages therefore prove the same original binary
    identity against the immutable capture-ledger event instead of trusting a
    caller-supplied event label.
    """

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read CDM capture binding: {error}") from error
    if not isinstance(record, Mapping) or set(record) != {
        "schema_version",
        "status",
        "interpretation",
        "capture_event",
        "run_provenance",
        "reasons",
    } or record.get("schema_version") != 1:
        raise ValueError("CDM capture binding schema is unsupported")
    if record.get("status") != "capture_dm_run_provenance_bound":
        raise ValueError("CDM capture binding is not provenance-bound")
    run_provenance = record.get("run_provenance")
    if not isinstance(run_provenance, Mapping) or run_provenance.get("dark_matter_model") != "cdm":
        raise ValueError("CDM pair track cannot use a non-CDM capture binding")
    capture = record.get("capture_event")
    if not isinstance(capture, Mapping) or set(capture) != {
        "event_uid",
        "event_sha256",
        "ledger_path",
    }:
        raise ValueError("CDM capture binding event fields are invalid")
    if capture.get("event_uid") != capture_event_uid:
        raise ValueError("CDM capture binding event UID differs from the requested pair")
    event_sha256 = capture.get("event_sha256")
    if not isinstance(event_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", event_sha256) is None:
        raise ValueError("CDM capture binding event SHA-256 is invalid")
    ledger_reference = capture.get("ledger_path")
    if not isinstance(ledger_reference, str) or not ledger_reference.strip():
        raise ValueError("CDM capture binding ledger_path is invalid")
    ledger_path = _resolve(ledger_reference, source.parent)
    ledger = read_capture_ledger(ledger_path)
    matches = [event for event in ledger.events if event.event_uid == capture_event_uid]
    if len(matches) != 1:
        raise ValueError("CDM capture binding event is absent from its ledger")
    event = matches[0]
    if event.event_sha256 != event_sha256.lower():
        raise ValueError("CDM capture binding event SHA-256 differs from its ledger")
    if event.classification != "BINARY" or {
        member.sink_id for member in event.members
    } != {primary_sink_id, secondary_sink_id}:
        raise ValueError("CDM capture binding does not contain the requested binary pair")
    return {
        "path": str(source),
        "sha256": _file_sha256(source),
        "capture_event_uid": capture_event_uid,
        "capture_event_sha256": event.event_sha256,
        "capture_ledger_path": str(ledger_path),
        "capture_ledger_sha256": _file_sha256(ledger_path),
        "primary_sink_id": primary_sink_id,
        "secondary_sink_id": secondary_sink_id,
    }


@dataclass(frozen=True)
class LagRamsesCDMPairOrbitTrack:
    physics_id: str
    capture_event_uid: str
    primary_sink_id: int
    secondary_sink_id: int
    position_coordinate: str
    time_coordinate: str
    capture_binding: dict[str, Any]
    runtime_identity: dict[str, Any]
    samples: tuple[dict[str, Any], ...]
    source_outputs: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LAGRAMSES_CDM_PAIR_TRACK_SCHEMA_VERSION,
            "status": "raw_relative_orbit_track",
            "interpretation": (
                "instantaneous lagRamses CDM relative states only; orbit averaging, "
                "force decomposition, convergence acceptance, and a physical delay "
                "remain separate steps"
            ),
            "dark_matter_model": "cdm",
            "physics_id": self.physics_id,
            "capture_event_uid": self.capture_event_uid,
            "capture_binding": self.capture_binding,
            "runtime_identity": self.runtime_identity,
            "primary_sink_id": self.primary_sink_id,
            "secondary_sink_id": self.secondary_sink_id,
            "position_coordinate": self.position_coordinate,
            "time_coordinate": self.time_coordinate,
            "samples": list(self.samples),
            "source_outputs": list(self.source_outputs),
        }


def extract_lagramses_cdm_pair_orbit_track(
    runtime_identity_path: str | Path,
) -> LagRamsesCDMPairOrbitTrack:
    """Extract one raw track from the exact outputs in a verified identity.

    Callers cannot substitute an independently chosen output set, case, pair,
    coordinate convention, or capture binding after runtime verification.
    """

    # A local import avoids a module cycle: runtime validation itself checks
    # the original capture event through ``read_bound_cdm_capture`` above.
    from .cdm_zoom_runtime_identity import (
        read_verified_cdm_noncompacting_zoom_runtime_identity,
    )

    runtime = read_verified_cdm_noncompacting_zoom_runtime_identity(runtime_identity_path)
    contract = runtime.contract
    physics_id = contract.case.physics.physics_id
    capture_binding = contract.capture_binding
    capture_event_uid = _nonempty(capture_binding["capture_event_uid"], "capture_event_uid")
    primary_sink_id = capture_binding["primary_sink_id"]
    secondary_sink_id = capture_binding["secondary_sink_id"]
    if (
        isinstance(primary_sink_id, bool)
        or not isinstance(primary_sink_id, int)
        or isinstance(secondary_sink_id, bool)
        or not isinstance(secondary_sink_id, int)
        or primary_sink_id <= 0
        or secondary_sink_id <= 0
        or primary_sink_id == secondary_sink_id
    ):
        raise ValueError("runtime identity capture-pair sink IDs are invalid")
    position_coordinate = contract.plan.position_coordinate
    time_coordinate = contract.plan.time_coordinate

    prepared: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for directory in runtime.output_directories:
        match = _OUTPUT_DIRECTORY.fullmatch(directory.name)
        if match is None:
            raise ValueError("each lagRamses output directory must be output_00000 form")
        output_number = match.group(1)
        complete = directory / "COMPLETE"
        try:
            marker = complete.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ValueError(f"cannot read output COMPLETE marker: {error}") from error
        if marker != output_number:
            raise ValueError("lagRamses output COMPLETE marker is invalid")
        provenance_path = directory / f"dm_run_provenance_{output_number}.txt"
        provenance = read_dark_matter_run_provenance(provenance_path)
        preflight = assess_noncompacting_cdm_zoom_run(provenance)
        if not preflight.ready:
            raise ValueError("lagRamses output is not an accepted non-compacting CDM zoom")
        info_path = directory / f"info_{output_number}.txt"
        sink_path = directory / f"sink_{output_number}.csv"
        info = _info_records(info_path)
        states = _sink_states(sink_path)
        try:
            primary = states[primary_sink_id]
            secondary = states[secondary_sink_id]
        except KeyError as error:
            raise ValueError(
                f"lagRamses output {output_number} does not retain both requested sinks"
            ) from error
        time_code = _number(info["time"], "output time")
        aexp = _number(info["aexp"], "output aexp", positive=True)
        unit_length_cgs = _number(info["unit_l"], "output unit_l", positive=True)
        unit_time_s = _number(info["unit_t"], "output unit_t", positive=True)
        boxlen = _number(info["boxlen"], "output boxlen", positive=True)
        scale_pc = unit_length_cgs / _PC_CGS
        if position_coordinate == "comoving":
            scale_pc *= aexp
        relative_position_code = tuple(
            _minimum_image(second - first, boxlen)
            for first, second in zip(primary.position_code, secondary.position_code)
        )
        relative_position_pc = tuple(component * scale_pc for component in relative_position_code)
        relative_velocity_code = tuple(
            second - first for first, second in zip(primary.velocity_code, secondary.velocity_code)
        )
        separation_pc = math.sqrt(sum(component**2 for component in relative_position_pc))
        if not math.isfinite(separation_pc) or separation_pc <= 0.0:
            raise ValueError("lagRamses output has zero or invalid pair separation")
        time_myr = time_code * unit_time_s / _MYR_SECONDS
        if not math.isfinite(time_myr):
            raise ValueError("lagRamses output proper time is invalid")
        sample = {
            "output_number": output_number,
            "time_myr": time_myr,
            "aexp": aexp,
            "primary_mass_code": primary.mass_code,
            "secondary_mass_code": secondary.mass_code,
            "relative_position_pc": list(relative_position_pc),
            "relative_velocity_code": list(relative_velocity_code),
            "separation_pc": separation_pc,
        }
        source = {
            "output_number": output_number,
            "directory": str(directory),
            "complete_marker": {"path": str(complete), "sha256": _file_sha256(complete)},
            "dm_run_provenance": {
                "path": str(provenance_path),
                "sha256": provenance.source_sha256,
            },
            "info": {"path": str(info_path), "sha256": _file_sha256(info_path)},
            "sink_csv": {"path": str(sink_path), "sha256": _file_sha256(sink_path)},
        }
        prepared.append((time_myr, sample, source))
    if len(prepared) < 3:
        raise ValueError("lagRamses raw relative-orbit track requires at least three outputs")
    prepared.sort(key=lambda item: item[0])
    if any(left[0] >= right[0] for left, right in zip(prepared[:-1], prepared[1:])):
        raise ValueError("lagRamses output proper times must be strictly increasing")
    return LagRamsesCDMPairOrbitTrack(
        physics_id=physics_id,
        capture_event_uid=capture_event_uid,
        primary_sink_id=primary_sink_id,
        secondary_sink_id=secondary_sink_id,
        position_coordinate=position_coordinate,
        time_coordinate=time_coordinate,
        capture_binding=capture_binding,
        runtime_identity={
            "path": str(runtime.source_path),
            "sha256": runtime.source_sha256,
            "contract_path": str(contract.source_path),
            "contract_sha256": contract.source_sha256,
        },
        samples=tuple(item[1] for item in prepared),
        source_outputs=tuple(item[2] for item in prepared),
    )
