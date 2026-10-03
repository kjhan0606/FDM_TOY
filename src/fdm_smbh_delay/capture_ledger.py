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
    censored_batch_uids: tuple[str, ...] = ()


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


@dataclass
class _OpenBatch:
    begin: dict[str, Any]
    events: list[CaptureEvent]

    @property
    def uid(self) -> str:
        return str(self.begin.get("batch_uid", ""))


@dataclass(frozen=True)
class _Attempt:
    resume_step: int
    parent_index: int | None
    parent_batch_cutoff: int | None


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


def _close_code(actual: float, expected: float, *, scale: float = 0.0) -> bool:
    # No fixed absolute floor: code-unit masses and energies may be tiny.
    # Cancellation-dominated quantities use their uncancelled term scale.
    return abs(actual - expected) <= 2.0e-12 * max(
        abs(actual), abs(expected), abs(scale)
    )


def _vector_close(actual: np.ndarray, expected: np.ndarray, scale: np.ndarray) -> bool:
    return all(
        _close_code(float(got), float(want), scale=float(term))
        for got, want, term in zip(actual, expected, scale)
    )


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
    com_delta = _minimum_image_code(_vector(begin, "com_position_code") - com_position, box)
    if not _vector_close(com_delta, np.zeros(3), box):
        raise CaptureLedgerError(f"{uid}: native centre-of-mass position conservation failed")
    velocity_scale = np.sum(np.abs(masses[:, None] * velocities), axis=0) / total_mass
    if not _vector_close(_vector(begin, "com_velocity_code"), com_velocity, velocity_scale):
        raise CaptureLedgerError(f"{uid}: native centre-of-mass velocity conservation failed")

    by_id = {int(row["sink_id"]): (mass, pos, vel)
             for row, mass, pos, vel in zip(member_rows, masses, positions, velocities)}
    max_separation = 0.0
    max_separation_scale = 0.0
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
        separation_scale = float(np.linalg.norm(np.maximum(np.abs(pos1), np.abs(pos2))))
        speed_scale = float(np.linalg.norm(np.maximum(np.abs(vel1), np.abs(vel2))))
        max_separation_scale = max(max_separation_scale, separation_scale)
        h_scale = np.array([
            abs(dr[1] * dv[2]) + abs(dr[2] * dv[1]),
            abs(dr[2] * dv[0]) + abs(dr[0] * dv[2]),
            abs(dr[0] * dv[1]) + abs(dr[1] * dv[0]),
        ])
        vector_checks = (
            ("delta_position_code", dr, np.maximum(np.abs(pos1), np.abs(pos2))),
            ("delta_velocity_code", dv, np.maximum(np.abs(vel1), np.abs(vel2))),
            ("specific_angular_momentum_code", specific_h, h_scale),
            ("relative_angular_momentum_code", reduced_mass * specific_h, reduced_mass * h_scale),
        )
        scalar_checks = (
            ("separation_code", separation, separation_scale),
            ("relative_speed_code", speed, speed_scale),
            ("reduced_mass_code", reduced_mass, 0.0),
            ("relative_kinetic_code", kinetic, reduced_mass * speed * speed_scale),
        )
        for field, expected, scale in vector_checks:
            if not _vector_close(_vector(pair, field), expected, scale):
                raise CaptureLedgerError(f"{uid}: native pair {id1}-{id2} {field} invariant failed")
        for field, expected, scale in scalar_checks:
            if not _close_code(_finite_float(pair, field), expected, scale=scale):
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
        radius_error_factor = separation_scale / separation
        gravity = fact_g * (mass1 + mass2) / separation
        energy_scale = (
            0.5 * speed**2 + gravity
            + speed * speed_scale + gravity * radius_error_factor
        )
        for field, expected, scale in (
            ("newtonian_potential_1overr_code", potential,
             abs(potential) * radius_error_factor),
            ("two_body_specific_energy_code", energy, energy_scale),
            ("legacy_binding_proxy_1overr2_code", legacy_proxy,
             2.0 * abs(legacy_proxy) * radius_error_factor),
        ):
            if not _close_code(_finite_float(pair, field), expected, scale=scale):
                raise CaptureLedgerError(f"{uid}: native pair {id1}-{id2} {field} invariant failed")
        recorded_energy = _finite_float(pair, "two_body_specific_energy_code")
        if (not _close_code(recorded_energy, 0.0, scale=energy_scale)
                and _boolean(pair, "two_body_bound") != (recorded_energy < 0.0)):
            raise CaptureLedgerError(f"{uid}: native pair {id1}-{id2} two-body bound invariant failed")
        recorded_kinetic = _finite_float(pair, "relative_kinetic_code")
        recorded_proxy = _finite_float(pair, "legacy_binding_proxy_1overr2_code")
        if (not _close_code(recorded_kinetic, recorded_proxy,
                            scale=reduced_mass * speed * speed_scale
                            + 2.0 * abs(legacy_proxy) * radius_error_factor)
                and _boolean(pair, "legacy_pair_bound") != (recorded_kinetic < recorded_proxy)):
            raise CaptureLedgerError(f"{uid}: native pair {id1}-{id2} legacy bound invariant failed")
    if not _close_code(_finite_float(begin, "max_pair_separation_code"),
                       max_separation, scale=max_separation_scale):
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
        separation_scale = float(np.linalg.norm(np.maximum(
            np.abs(position_code_by_id[id1]), np.abs(position_code_by_id[id2])
        )))
        if (within_merge != (source_separation_code <= merge_radius_code)
                and not _close_code(source_separation_code, merge_radius_code,
                                    scale=separation_scale)):
            raise CaptureLedgerError(f"{uid}: pair {key} within_rmerge contradicts source geometry")
        source_bound = _boolean(row, "two_body_bound")
        legacy_bound = _boolean(row, "legacy_pair_bound")
        if "two_body_specific_energy_code" in row:
            specific_energy_code = row["two_body_specific_energy_code"]
            if specific_energy_code is None:
                if source_bound:
                    raise CaptureLedgerError(f"{uid}: singular pair cannot be two-body bound")
            else:
                recorded_energy = _finite_float(row, "two_body_specific_energy_code")
                source_speed = _finite_float(row, "relative_speed_code") if "relative_speed_code" in row else 0.0
                kinetic_scale = 0.5 * source_speed**2
                if (source_bound != (recorded_energy < 0.0)
                        and not _close_code(recorded_energy, 0.0,
                                            scale=kinetic_scale + abs(kinetic_scale - recorded_energy))):
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


def _ledger_int(record: dict[str, Any], key: str, *, minimum: int = 0) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CaptureLedgerError(f"{key} must be an integer >= {minimum}")
    return value


def _validate_batched_event_source(block: _OpenEvent) -> None:
    begin = block.begin
    if "periodic_box_size_code" not in begin:
        raise CaptureLedgerError(f"{block.uid}: batched event lacks periodic box extents")
    primary = _ledger_int(begin, "primary_sink_id", minimum=1)
    ordered_members = sorted(block.members, key=lambda row: _ledger_int(row, "member_index", minimum=1))
    masses = [_finite_float(row, "mass_code", positive=True) for row in ordered_members]
    if not masses:
        raise CaptureLedgerError(f"{block.uid}: batched event has no members")
    expected = _ledger_int(ordered_members[masses.index(max(masses))], "sink_id", minimum=1)
    if primary != expected:
        raise CaptureLedgerError(f"{block.uid}: primary sink violates survivor rule")
    for member in block.members:
        if _ledger_int(member, "primary_sink_id", minimum=1) != primary:
            raise CaptureLedgerError(f"{block.uid}: inconsistent member primary sink")
        if _boolean(member, "is_primary") != (member["sink_id"] == primary):
            raise CaptureLedgerError(f"{block.uid}: inconsistent member primary flag")


def read_capture_ledger(
    path: str | Path, *, allow_incomplete_tail: bool = False,
    allow_incomplete_batches: bool = True,
) -> CaptureLedger:
    """Read only committed events on the final restart lineage.

    Legacy bare events remain readable until the first batch/attempt marker.
    A complete ``event_end`` in a new ledger is not enough: its batch must
    commit after sink compaction, and superseded restart branches are excluded.
    A batch interrupted by a valid later restart attempt is censored by
    default. An incomplete final batch still requires ``allow_incomplete_tail``.
    """

    resolved = Path(path).expanduser().resolve()
    current: _OpenEvent | None = None
    batch: _OpenBatch | None = None
    seen_protocol = False
    legacy: list[CaptureEvent] = []
    committed: list[tuple[_OpenBatch, str, int | None]] = []
    attempts: list[_Attempt] = []
    # The third value is the committed-batch count at the checkpoint marker.
    # Ordering, not coarse-step equality, decides what the snapshot contains.
    checkpoints: dict[int, tuple[int, int, int]] = {}
    incomplete: list[str] = []
    censored_batches: list[str] = []

    def censor_open_batch(reason: str) -> None:
        nonlocal current, batch
        if batch is None:
            return
        if not allow_incomplete_batches:
            raise CaptureLedgerError(f"{batch.uid}: {reason}")
        if current is not None:
            incomplete.append(current.uid)
            current = None
        censored_batches.append(batch.uid)
        batch = None

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
            if not isinstance(record, dict):
                raise CaptureLedgerError(f"{resolved}:{line_number}: record is not an object")
            record_type = record.get("record_type")
            if record_type == "attempt_begin":
                if current is not None and batch is None:
                    raise CaptureLedgerError(f"{current.uid}: missing event_end")
                censor_open_batch("missing batch_commit before next attempt")
                restart_output = _ledger_int(record, "restart_output")
                resume_step = _ledger_int(record, "resume_step")
                if record.get("schema_version") != CAPTURE_LEDGER_SCHEMA_VERSION:
                    raise CaptureLedgerError("unsupported attempt schema")
                if restart_output == 0:
                    if attempts or committed or legacy:
                        raise CaptureLedgerError("fresh attempt appended to existing ledger")
                    parent = None
                    parent_cutoff = None
                else:
                    if legacy:
                        raise CaptureLedgerError("legacy bare events cannot prove restart lineage")
                    checkpoint = checkpoints.get(restart_output)
                    if checkpoint is None or checkpoint[1] != resume_step:
                        raise CaptureLedgerError("restart has no matching ledger checkpoint")
                    parent = checkpoint[0]
                    parent_cutoff = checkpoint[2]
                attempts.append(_Attempt(resume_step, parent, parent_cutoff))
                seen_protocol = True
            elif record_type == "checkpoint":
                if current is not None or batch is not None or not attempts:
                    raise CaptureLedgerError("checkpoint outside a completed batch/attempt")
                if record.get("schema_version") != CAPTURE_LEDGER_SCHEMA_VERSION:
                    raise CaptureLedgerError("unsupported checkpoint schema")
                output = _ledger_int(record, "output_number", minimum=1)
                step = _ledger_int(record, "nstep_coarse")
                if step < attempts[-1].resume_step:
                    raise CaptureLedgerError("checkpoint predates active attempt")
                checkpoints[output] = (len(attempts) - 1, step, len(committed))
            elif record_type == "batch_begin":
                if current is not None and batch is None:
                    raise CaptureLedgerError(f"{current.uid}: missing event_end")
                if batch is not None:
                    raise CaptureLedgerError(
                        f"{batch.uid}: missing batch_commit before next batch in same attempt"
                    )
                if not attempts:
                    raise CaptureLedgerError("batch has no run attempt marker")
                if record.get("schema_version") != CAPTURE_LEDGER_SCHEMA_VERSION:
                    raise CaptureLedgerError("unsupported batch schema")
                batch = _OpenBatch(record, [])
                seen_protocol = True
            elif record_type == "event_begin":
                if current is not None:
                    raise CaptureLedgerError(
                        f"{current.uid}: missing event_end before line {line_number}"
                    )
                if seen_protocol and batch is None:
                    raise CaptureLedgerError("bare event after batch protocol began")
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
                    raise CaptureLedgerError(f"{resolved}:{line_number}: orphaned event_end")
                if batch is not None:
                    _validate_batched_event_source(current)
                event = _build_event(
                    current, record, source_path=resolved, last_line=line_number
                )
                if batch is None:
                    legacy.append(event)
                else:
                    batch.events.append(event)
                current = None
            elif record_type == "batch_commit":
                if batch is None or current is not None:
                    raise CaptureLedgerError("batch_commit without complete open batch")
                begin = batch.begin
                uid = batch.uid
                step = _ledger_int(begin, "nstep_coarse")
                level = _ledger_int(begin, "ilevel", minimum=1)
                before = _ledger_int(begin, "nsink_before", minimum=1)
                after = _ledger_int(begin, "nsink_after", minimum=1)
                expected = _ledger_int(begin, "expected_events", minimum=1)
                if (
                    record.get("schema_version") != CAPTURE_LEDGER_SCHEMA_VERSION
                    or uid != f"{step}-{level}-{before}-{after}"
                    or record.get("batch_uid") != uid
                    or _ledger_int(record, "nsink_after", minimum=1) != after
                    or step < attempts[-1].resume_step
                    or before <= after
                    or len(batch.events) != expected
                ):
                    raise CaptureLedgerError(f"{uid}: invalid batch commit or metadata")
                reduction = sum(len(event.members) - 1 for event in batch.events)
                if reduction != before - after:
                    raise CaptureLedgerError(f"{uid}: sink-count conservation failed")
                member_ids: set[int] = set()
                event_uids: set[str] = set()
                for event in batch.events:
                    ids = {member.sink_id for member in event.members}
                    expected_uid = f"{step}-{level}-{min(ids)}-{max(ids)}-{len(ids)}"
                    if (
                        event.event_uid != expected_uid
                        or event.nstep_coarse != step or event.level != level
                        or event.event_uid in event_uids or member_ids & ids
                    ):
                        raise CaptureLedgerError(f"{uid}: inconsistent batched event")
                    event_uids.add(event.event_uid)
                    member_ids.update(ids)
                payload = [begin, [(event.event_uid, event.event_sha256) for event in batch.events], record]
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                committed.append((batch, digest, len(attempts) - 1))
                batch = None
            else:
                raise CaptureLedgerError(
                    f"{resolved}:{line_number}: unknown record_type={record_type!r}"
                )

    if current is not None:
        incomplete.append(current.uid)
        if batch is None and not allow_incomplete_tail:
            raise CaptureLedgerError(f"{current.uid}: incomplete event at end of ledger")
    if batch is not None:
        if not allow_incomplete_tail:
            raise CaptureLedgerError(f"{batch.uid}: incomplete batch at end of ledger")
        censored_batches.append(batch.uid)

    active_cutoffs: dict[int, int | None] = {}
    if attempts:
        index: int | None = len(attempts) - 1
        cutoff: int | None = None
        while index is not None:
            if index in active_cutoffs:
                raise CaptureLedgerError("restart lineage cycle")
            active_cutoffs[index] = cutoff
            cutoff = attempts[index].parent_batch_cutoff
            index = attempts[index].parent_index

    events: dict[str, CaptureEvent] = {}
    batch_digests: dict[str, str] = {}
    duplicates = 0
    def accept(event: CaptureEvent) -> None:
        nonlocal duplicates
        previous = events.get(event.event_uid)
        if previous is None:
            events[event.event_uid] = event
        elif previous.event_sha256 == event.event_sha256:
            duplicates += 1
        else:
            raise CaptureLedgerError(f"{event.event_uid}: conflicting deterministic event UID")

    for event in legacy:
        accept(event)
    for batch_index, (committed_batch, digest, owner) in enumerate(committed):
        if owner not in active_cutoffs:
            continue
        cutoff = active_cutoffs[owner]
        if cutoff is not None and batch_index >= cutoff:
            continue
        previous = batch_digests.get(committed_batch.uid)
        if previous is not None and previous != digest:
            raise CaptureLedgerError(f"{committed_batch.uid}: conflicting deterministic batch UID")
        batch_digests[committed_batch.uid] = digest
        for event in committed_batch.events:
            accept(event)
    return CaptureLedger(
        source_path=resolved, events=tuple(events.values()),
        duplicate_events=duplicates, incomplete_event_uids=tuple(incomplete),
        censored_batch_uids=tuple(censored_batches),
    )
