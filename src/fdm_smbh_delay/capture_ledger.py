"""Strict ingestion of lagRamses pre-compaction SMBH capture ledgers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from astropy import units as u
import numpy as np

from .lagramses import PairOrbitalState, pair_orbital_state


CAPTURE_LEDGER_SCHEMA_VERSION = 1


class CaptureLedgerError(ValueError):
    """Raised when a capture ledger cannot support physical post-processing."""


@dataclass(frozen=True)
class CaptureMember:
    sink_id: int
    mass_msun: float
    position_pc: np.ndarray
    velocity_pc_myr: np.ndarray
    formation_time_code: float
    spin_magnitude: float
    spin_direction: np.ndarray
    gas_angular_momentum_code: np.ndarray
    accreted_mass_msun: float


@dataclass(frozen=True)
class CapturePair:
    member_ids: tuple[int, int]
    orbital_state: PairOrbitalState | None
    within_numerical_merge_radius: bool
    source_two_body_bound: bool
    legacy_pair_bound: bool


@dataclass(frozen=True)
class CaptureEvent:
    event_uid: str
    classification: str
    nstep_coarse: int
    level: int
    scale_factor: float
    redshift: float
    code_time: float
    proper_time_code: float
    numerical_merge_radius_pc: float
    members: tuple[CaptureMember, ...]
    pairs: tuple[CapturePair, ...]
    event_sha256: str
    source_path: Path
    first_line: int
    last_line: int

    @property
    def binary_orbital_state(self) -> PairOrbitalState | None:
        """Return a state only for an unambiguous two-member event."""

        if self.classification != "BINARY" or len(self.pairs) != 1:
            return None
        return self.pairs[0].orbital_state


@dataclass(frozen=True)
class CaptureLedger:
    source_path: Path
    events: tuple[CaptureEvent, ...]
    duplicate_events: int
    incomplete_event_uids: tuple[str, ...]


@dataclass
class _OpenEvent:
    begin: dict[str, Any]
    first_line: int
    rows: list[dict[str, Any]]
    members: list[dict[str, Any]]
    pairs: list[dict[str, Any]]

    @property
    def uid(self) -> str:
        return str(self.begin.get("event_uid", ""))


def _finite_float(record: dict[str, Any], key: str, *, positive: bool = False) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptureLedgerError(f"{key} must be numeric")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise CaptureLedgerError(f"{key} must be {qualifier}")
    return result


def _vector(record: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(record.get(key), dtype=float)
    if value.shape != (3,) or np.any(~np.isfinite(value)):
        raise CaptureLedgerError(f"{key} must be a finite three-vector")
    return value


def _event_digest(rows: list[dict[str, Any]]) -> str:
    canonical = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unit_scales(begin: dict[str, Any]) -> tuple[float, float, float]:
    length = (_finite_float(begin, "unit_length_cgs", positive=True) * u.cm).to_value(
        u.pc
    )
    velocity = (
        _finite_float(begin, "unit_velocity_cgs", positive=True) * u.cm / u.s
    ).to_value(u.pc / u.Myr)
    mass = (_finite_float(begin, "unit_mass_cgs", positive=True) * u.g).to_value(
        u.Msun
    )
    return float(length), float(velocity), float(mass)


def _build_event(
    block: _OpenEvent,
    end: dict[str, Any],
    *,
    source_path: Path,
    last_line: int,
) -> CaptureEvent:
    begin = block.begin
    uid = block.uid
    if not uid or end.get("event_uid") != uid:
        raise CaptureLedgerError("capture transaction has a missing or mismatched UID")
    if (
        begin.get("schema_version") != CAPTURE_LEDGER_SCHEMA_VERSION
        or end.get("schema_version") != CAPTURE_LEDGER_SCHEMA_VERSION
    ):
        raise CaptureLedgerError(f"{uid}: unsupported ledger schema")
    if begin.get("complete") is not False or end.get("complete") is not True:
        raise CaptureLedgerError(f"{uid}: completion markers are invalid")

    nmember = int(begin.get("nmember", -1))
    expected_pairs = nmember * (nmember - 1) // 2
    classification = str(begin.get("classification", ""))
    if nmember < 2 or len(block.members) != nmember:
        raise CaptureLedgerError(f"{uid}: member count is inconsistent")
    if (
        begin.get("expected_pairs") != expected_pairs
        or len(block.pairs) != expected_pairs
        or end.get("nmember") != nmember
        or end.get("npair") != expected_pairs
    ):
        raise CaptureLedgerError(f"{uid}: pair count is inconsistent")
    expected_classification = "BINARY" if nmember == 2 else "MULTIPLE"
    if classification != expected_classification:
        raise CaptureLedgerError(
            f"{uid}: classification must be {expected_classification}"
        )

    length_per_code, velocity_per_code, mass_per_code = _unit_scales(begin)
    member_rows = sorted(block.members, key=lambda row: int(row["member_index"]))
    if [int(row["member_index"]) for row in member_rows] != list(
        range(1, nmember + 1)
    ):
        raise CaptureLedgerError(f"{uid}: member indices are not contiguous")

    members = []
    member_by_id: dict[int, CaptureMember] = {}
    for row in member_rows:
        sink_id = int(row["sink_id"])
        if sink_id in member_by_id:
            raise CaptureLedgerError(f"{uid}: duplicate sink ID {sink_id}")
        member = CaptureMember(
            sink_id=sink_id,
            mass_msun=_finite_float(row, "mass_code", positive=True) * mass_per_code,
            position_pc=_vector(row, "position_code") * length_per_code,
            velocity_pc_myr=_vector(row, "velocity_code") * velocity_per_code,
            formation_time_code=_finite_float(row, "formation_time_code"),
            spin_magnitude=_finite_float(row, "spin_magnitude"),
            spin_direction=_vector(row, "spin_direction"),
            gas_angular_momentum_code=_vector(row, "gas_angular_momentum_code"),
            accreted_mass_msun=_finite_float(row, "accreted_mass_code")
            * mass_per_code,
        )
        members.append(member)
        member_by_id[sink_id] = member

    box_size_pc = _finite_float(begin, "boxlen", positive=True) * length_per_code
    pairs = []
    seen_pairs: set[tuple[int, int]] = set()
    for row in sorted(block.pairs, key=lambda item: int(item["pair_index"])):
        id1 = int(row["sink_id_1"])
        id2 = int(row["sink_id_2"])
        key = tuple(sorted((id1, id2)))
        if id1 not in member_by_id or id2 not in member_by_id or key in seen_pairs:
            raise CaptureLedgerError(f"{uid}: invalid or duplicate pair {key}")
        seen_pairs.add(key)
        first = member_by_id[id1]
        second = member_by_id[id2]
        try:
            state = pair_orbital_state(
                member_ids=(id1, id2),
                masses_msun=(first.mass_msun, second.mass_msun),
                positions_pc=np.asarray([first.position_pc, second.position_pc]),
                velocities_pc_myr=np.asarray(
                    [first.velocity_pc_myr, second.velocity_pc_myr]
                ),
                periodic_box_pc=box_size_pc,
            )
        except ValueError:
            state = None
        pairs.append(
            CapturePair(
                member_ids=(id1, id2),
                orbital_state=state,
                within_numerical_merge_radius=bool(row["within_rmerge"]),
                source_two_body_bound=bool(row["two_body_bound"]),
                legacy_pair_bound=bool(row["legacy_pair_bound"]),
            )
        )
    if len(seen_pairs) != expected_pairs:
        raise CaptureLedgerError(f"{uid}: pair coverage is incomplete")

    rows = block.rows + [end]
    return CaptureEvent(
        event_uid=uid,
        classification=classification,
        nstep_coarse=int(begin["nstep_coarse"]),
        level=int(begin["ilevel"]),
        scale_factor=_finite_float(begin, "aexp", positive=True),
        redshift=_finite_float(begin, "redshift"),
        code_time=_finite_float(begin, "t_code"),
        proper_time_code=_finite_float(begin, "texp"),
        numerical_merge_radius_pc=(
            _finite_float(begin, "merge_radius_code", positive=True)
            * length_per_code
        ),
        members=tuple(members),
        pairs=tuple(pairs),
        event_sha256=_event_digest(rows),
        source_path=source_path,
        first_line=block.first_line,
        last_line=last_line,
    )


def read_capture_ledger(
    path: str | Path, *, allow_incomplete_tail: bool = False
) -> CaptureLedger:
    """Read complete transactions and deduplicate bitwise restart repeats.

    An event without ``event_end`` is never promoted to a physical input.  It
    may be reported as a censored tail only when ``allow_incomplete_tail`` is
    explicit.  A repeated deterministic UID with different contents is always
    rejected as a provenance conflict.
    """

    resolved = Path(path).expanduser().resolve()
    current: _OpenEvent | None = None
    events: dict[str, CaptureEvent] = {}
    duplicates = 0
    incomplete: list[str] = []

    with resolved.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CaptureLedgerError(
                    f"{resolved}:{line_number}: invalid JSON"
                ) from exc
            record_type = record.get("record_type")
            if record_type == "event_begin":
                if current is not None:
                    raise CaptureLedgerError(
                        f"{current.uid}: missing event_end before line {line_number}"
                    )
                current = _OpenEvent(record, line_number, [record], [], [])
            elif record_type in {"member", "pair"}:
                if current is None or record.get("event_uid") != current.uid:
                    raise CaptureLedgerError(
                        f"{resolved}:{line_number}: orphaned or mismatched {record_type}"
                    )
                current.rows.append(record)
                if record_type == "member":
                    current.members.append(record)
                else:
                    current.pairs.append(record)
            elif record_type == "event_end":
                if current is None:
                    raise CaptureLedgerError(
                        f"{resolved}:{line_number}: orphaned event_end"
                    )
                event = _build_event(
                    current, record, source_path=resolved, last_line=line_number
                )
                previous = events.get(event.event_uid)
                if previous is None:
                    events[event.event_uid] = event
                elif previous.event_sha256 == event.event_sha256:
                    duplicates += 1
                else:
                    raise CaptureLedgerError(
                        f"{event.event_uid}: conflicting deterministic event UID"
                    )
                current = None
            else:
                raise CaptureLedgerError(
                    f"{resolved}:{line_number}: unknown record_type={record_type!r}"
                )

    if current is not None:
        incomplete.append(current.uid)
        if not allow_incomplete_tail:
            raise CaptureLedgerError(f"{current.uid}: incomplete event at end of ledger")
    return CaptureLedger(
        source_path=resolved,
        events=tuple(events.values()),
        duplicate_events=duplicates,
        incomplete_event_uids=tuple(incomplete),
    )
