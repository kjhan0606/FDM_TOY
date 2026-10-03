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
    multiple_members_preserved: bool | None = None
    native_conservation_verified: bool = False

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


def _boolean(record: dict[str, Any], key: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise CaptureLedgerError(f"{key} must be a boolean")
    return value


def _close_code(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=2.0e-12, atol=1.0e-14))


def _periodic_box_code(begin: dict[str, Any]) -> np.ndarray:
    """Use the writer's axis lengths, retaining cubic legacy-ledger support."""

    boxlen = _finite_float(begin, "boxlen", positive=True)
    if "periodic_box_size_code" not in begin:
        return np.full(3, boxlen)
    box = _vector(begin, "periodic_box_size_code")
    if np.any(box <= 0.0):
        raise CaptureLedgerError("periodic_box_size_code must be positive")
    return box


def _minimum_image_code(delta: np.ndarray, box_size_code: np.ndarray) -> np.ndarray:
    # Match the Fortran writer at the exactly half-box tie as well.
    result = delta.copy()
    above = result > 0.5 * box_size_code
    result[above] -= box_size_code[above]
    below = result < -0.5 * box_size_code
    result[below] += box_size_code[below]
    return result


def _validate_native_conservation(
    uid: str,
    begin: dict[str, Any],
    member_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> bool:
    """Independently check full native writer diagnostics when present.

    Minimal legacy/import fixtures lack these fields.  A record that begins
    claiming native conservation diagnostics must provide the entire set;
    partial native records cannot pass as independently checked events.
    """

    event_fields = (
        "factG_code", "total_mass_code", "com_position_code",
        "com_velocity_code", "max_pair_separation_code",
    )
    if not any(field in begin for field in (*event_fields, "periodic_box_size_code")):
        return False
    if any(field not in begin for field in event_fields):
        raise CaptureLedgerError(f"{uid}: incomplete native event conservation diagnostics")
    pair_fields = (
        "delta_position_code", "separation_code", "delta_velocity_code",
        "relative_speed_code", "reduced_mass_code", "relative_kinetic_code",
        "newtonian_potential_1overr_code", "two_body_specific_energy_code",
        "specific_angular_momentum_code", "relative_angular_momentum_code",
        "legacy_binding_proxy_1overr2_code",
    )
    if any(any(field not in row for field in pair_fields) for row in pair_rows):
        raise CaptureLedgerError(f"{uid}: incomplete native pair conservation diagnostics")

    box = _periodic_box_code(begin)
    fact_g = _finite_float(begin, "factG_code", positive=True)
    masses = np.array([_finite_float(row, "mass_code", positive=True) for row in member_rows])
    positions = np.array([_vector(row, "position_code") for row in member_rows])
    velocities = np.array([_vector(row, "velocity_code") for row in member_rows])
    total_mass = float(np.sum(masses))
    if not _close_code(_finite_float(begin, "total_mass_code"), total_mass):
        raise CaptureLedgerError(f"{uid}: native total-mass conservation failed")
    offsets = np.array([_minimum_image_code(pos - positions[0], box) for pos in positions])
    com_position = np.mod(positions[0] + np.sum(masses[:, None] * offsets, axis=0) / total_mass, box)
    com_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    if not np.allclose(_vector(begin, "com_position_code"), com_position, rtol=2.0e-12, atol=1.0e-14):
        raise CaptureLedgerError(f"{uid}: native centre-of-mass position conservation failed")
    if not np.allclose(_vector(begin, "com_velocity_code"), com_velocity, rtol=2.0e-12, atol=1.0e-14):
        raise CaptureLedgerError(f"{uid}: native centre-of-mass velocity conservation failed")

    by_id = {int(row["sink_id"]): (mass, pos, vel)
             for row, mass, pos, vel in zip(member_rows, masses, positions, velocities)}
    max_separation = 0.0
    for pair in pair_rows:
        id1, id2 = int(pair["sink_id_1"]), int(pair["sink_id_2"])
        mass1, pos1, vel1 = by_id[id1]
        mass2, pos2, vel2 = by_id[id2]
        dr = _minimum_image_code(pos2 - pos1, box)
        dv = vel2 - vel1
        separation = float(np.linalg.norm(dr))
        speed = float(np.linalg.norm(dv))
        reduced_mass = mass1 * mass2 / (mass1 + mass2)
        kinetic = 0.5 * reduced_mass * speed**2
        specific_h = np.cross(dr, dv)
        max_separation = max(max_separation, separation)
        vector_checks = (
            ("delta_position_code", dr),
            ("delta_velocity_code", dv),
            ("specific_angular_momentum_code", specific_h),
            ("relative_angular_momentum_code", reduced_mass * specific_h),
        )
        scalar_checks = (
            ("separation_code", separation),
            ("relative_speed_code", speed),
            ("reduced_mass_code", reduced_mass),
            ("relative_kinetic_code", kinetic),
        )
        for field, expected in vector_checks:
            if not np.allclose(_vector(pair, field), expected, rtol=2.0e-12, atol=1.0e-14):
                raise CaptureLedgerError(f"{uid}: native pair {id1}-{id2} {field} invariant failed")
        for field, expected in scalar_checks:
            if not _close_code(_finite_float(pair, field), expected):
                raise CaptureLedgerError(f"{uid}: native pair {id1}-{id2} {field} invariant failed")
        if separation <= np.finfo(float).tiny:
            if any(pair[field] is not None for field in (
                "newtonian_potential_1overr_code", "two_body_specific_energy_code",
                "legacy_binding_proxy_1overr2_code",
            )):
                raise CaptureLedgerError(f"{uid}: singular native pair energies must be null")
            if _boolean(pair, "two_body_bound") or _boolean(pair, "legacy_pair_bound"):
                raise CaptureLedgerError(f"{uid}: singular native pair cannot be bound")
            continue
        potential = -fact_g * mass1 * mass2 / separation
        energy = 0.5 * speed**2 - fact_g * (mass1 + mass2) / separation
        legacy_proxy = fact_g * mass1 * mass2 / separation**2
        for field, expected in (
            ("newtonian_potential_1overr_code", potential),
            ("two_body_specific_energy_code", energy),
            ("legacy_binding_proxy_1overr2_code", legacy_proxy),
        ):
            if not _close_code(_finite_float(pair, field), expected):
                raise CaptureLedgerError(f"{uid}: native pair {id1}-{id2} {field} invariant failed")
        if _boolean(pair, "two_body_bound") != (energy < 0.0):
            raise CaptureLedgerError(f"{uid}: native pair {id1}-{id2} two-body bound invariant failed")
        if _boolean(pair, "legacy_pair_bound") != (kinetic < legacy_proxy):
            raise CaptureLedgerError(f"{uid}: native pair {id1}-{id2} legacy bound invariant failed")
    if not _close_code(_finite_float(begin, "max_pair_separation_code"), max_separation):
        raise CaptureLedgerError(f"{uid}: native maximum pair separation invariant failed")
    return True


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
    if any(
        row.get("schema_version") != CAPTURE_LEDGER_SCHEMA_VERSION
        for row in (*block.rows, end)
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
    multiple_preserved = begin.get("multiple_members_preserved")
    if "multiple_members_preserved" in begin:
        if not isinstance(multiple_preserved, bool):
            raise CaptureLedgerError(f"{uid}: multiple_members_preserved must be boolean")
        if multiple_preserved and classification != "MULTIPLE":
            raise CaptureLedgerError(f"{uid}: multiple_members_preserved requires a MULTIPLE event")

    length_per_code, velocity_per_code, mass_per_code = _unit_scales(begin)
    member_rows = sorted(block.members, key=lambda row: int(row["member_index"]))
    if [int(row["member_index"]) for row in member_rows] != list(
        range(1, nmember + 1)
    ):
        raise CaptureLedgerError(f"{uid}: member indices are not contiguous")

    pair_rows = sorted(block.pairs, key=lambda row: int(row["pair_index"]))
    if [int(row["pair_index"]) for row in pair_rows] != list(
        range(1, expected_pairs + 1)
    ):
        raise CaptureLedgerError(f"{uid}: pair indices are not contiguous")

    members = []
    member_by_id: dict[int, CaptureMember] = {}
    position_code_by_id: dict[int, np.ndarray] = {}
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
        position_code_by_id[sink_id] = _vector(row, "position_code")

    box_size_code = _periodic_box_code(begin)
    box_size_pc = box_size_code * length_per_code
    pairs = []
    seen_pairs: set[tuple[int, int]] = set()
    merge_radius_code = _finite_float(begin, "merge_radius_code", positive=True)
    for row in pair_rows:
        id1 = int(row["sink_id_1"])
        id2 = int(row["sink_id_2"])
        key = tuple(sorted((id1, id2)))
        if id1 not in member_by_id or id2 not in member_by_id or key in seen_pairs:
            raise CaptureLedgerError(f"{uid}: invalid or duplicate pair {key}")
        seen_pairs.add(key)
        first = member_by_id[id1]
        second = member_by_id[id2]
        delta_code = _minimum_image_code(
            position_code_by_id[id2] - position_code_by_id[id1], box_size_code
        )
        source_separation_code = float(np.linalg.norm(delta_code))
        within_merge = _boolean(row, "within_rmerge")
        if within_merge != (source_separation_code <= merge_radius_code):
            raise CaptureLedgerError(f"{uid}: pair {key} within_rmerge contradicts source geometry")
        source_bound = _boolean(row, "two_body_bound")
        legacy_bound = _boolean(row, "legacy_pair_bound")
        if "two_body_specific_energy_code" in row:
            specific_energy_code = row["two_body_specific_energy_code"]
            if specific_energy_code is None:
                if source_bound:
                    raise CaptureLedgerError(f"{uid}: singular pair cannot be two-body bound")
            elif source_bound != (_finite_float(row, "two_body_specific_energy_code") < 0.0):
                raise CaptureLedgerError(f"{uid}: pair {key} two_body_bound contradicts source energy")
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
                within_numerical_merge_radius=within_merge,
                source_two_body_bound=source_bound,
                legacy_pair_bound=legacy_bound,
            )
        )
    if len(seen_pairs) != expected_pairs:
        raise CaptureLedgerError(f"{uid}: pair coverage is incomplete")

    native_conservation_verified = _validate_native_conservation(
        uid, begin, member_rows, pair_rows
    )

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
        multiple_members_preserved=multiple_preserved,
        native_conservation_verified=native_conservation_verified,
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
