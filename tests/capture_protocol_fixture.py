"""Committed synthetic capture ledgers for physical-consumer tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from fdm_smbh_delay.constants import G_INTERNAL


def _add_native_diagnostics(begin: dict, members: list[dict], pairs: list[dict]) -> None:
    """Fill the writer's conservation fields, never bypassing reader checks."""
    box = np.asarray(begin["periodic_box_size_code"], dtype=float)
    masses = np.asarray([row["mass_code"] for row in members], dtype=float)
    positions = np.asarray([row["position_code"] for row in members], dtype=float)
    velocities = np.asarray([row["velocity_code"] for row in members], dtype=float)
    total = float(masses.sum())
    offsets = positions - positions[0]
    offsets = np.where(offsets > box / 2, offsets - box, offsets)
    offsets = np.where(offsets < -box / 2, offsets + box, offsets)
    begin.setdefault("factG_code", G_INTERNAL)
    begin.setdefault("total_mass_code", total)
    begin.setdefault("com_position_code", np.mod(positions[0] +
                     np.sum(masses[:, None] * offsets, axis=0) / total, box).tolist())
    begin.setdefault("com_velocity_code", (
        np.sum(masses[:, None] * velocities, axis=0) / total).tolist())
    by_id = {row["sink_id"]: row for row in members}
    largest = 0.0
    for pair in pairs:
        first, second = by_id[pair["sink_id_1"]], by_id[pair["sink_id_2"]]
        dr = np.asarray(second["position_code"], dtype=float) - first["position_code"]
        dr = np.where(dr > box / 2, dr - box, dr)
        dr = np.where(dr < -box / 2, dr + box, dr)
        dv = np.asarray(second["velocity_code"], dtype=float) - first["velocity_code"]
        radius, speed = float(np.linalg.norm(dr)), float(np.linalg.norm(dv))
        largest = max(largest, radius)
        mass1, mass2 = first["mass_code"], second["mass_code"]
        reduced = mass1 * mass2 / (mass1 + mass2)
        angular = np.cross(dr, dv)
        gravity = begin["factG_code"]
        diagnostics = {
            "delta_position_code": dr.tolist(), "separation_code": radius,
            "delta_velocity_code": dv.tolist(), "relative_speed_code": speed,
            "reduced_mass_code": reduced,
            "relative_kinetic_code": 0.5 * reduced * speed**2,
            "newtonian_potential_1overr_code": -gravity * mass1 * mass2 / radius,
            "two_body_specific_energy_code": 0.5 * speed**2 - gravity * (mass1 + mass2) / radius,
            "specific_angular_momentum_code": angular.tolist(),
            "relative_angular_momentum_code": (reduced * angular).tolist(),
            "legacy_binding_proxy_1overr2_code": gravity * mass1 * mass2 / radius**2,
        }
        for field, value in diagnostics.items():
            pair.setdefault(field, value)
    begin.setdefault("max_pair_separation_code", largest)


def committed_capture_rows(rows: list[dict]) -> list[dict]:
    """Wrap one complete legacy-shaped event in the production batch protocol."""
    event_rows = copy.deepcopy(rows)
    begin = event_rows[0]
    assert begin["record_type"] == "event_begin"
    assert event_rows[-1]["record_type"] == "event_end"
    members = [row for row in event_rows if row["record_type"] == "member"]
    assert len(members) == begin["nmember"] >= 2
    step = begin["nstep_coarse"]
    level = begin["ilevel"]
    ids = [row["sink_id"] for row in members]
    uid = f"{step}-{level}-{min(ids)}-{max(ids)}-{len(ids)}"
    primary = max(members, key=lambda row: row["mass_code"])["sink_id"]
    for row in event_rows:
        row["event_uid"] = uid
    begin["periodic_box_size_code"] = [begin["boxlen"]] * 3
    begin["primary_sink_id"] = primary
    for row in members:
        row["primary_sink_id"] = primary
        row["is_primary"] = row["sink_id"] == primary
    pairs = [row for row in event_rows if row["record_type"] == "pair"]
    _add_native_diagnostics(begin, members, pairs)
    before = len(members)
    after = 1
    batch_uid = f"{step}-{level}-{before}-{after}"
    return [
        {"schema_version": 1, "record_type": "attempt_begin",
         "restart_output": 0, "resume_step": 0},
        {"schema_version": 1, "record_type": "batch_begin",
         "batch_uid": batch_uid, "nstep_coarse": step, "ilevel": level,
         "nsink_before": before, "nsink_after": after, "expected_events": 1},
        *event_rows,
        {"schema_version": 1, "record_type": "batch_commit",
         "batch_uid": batch_uid, "nsink_after": after},
    ]


def write_committed_capture(path: Path, rows: list[dict]) -> str:
    committed = committed_capture_rows(rows)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in committed),
        encoding="utf-8",
    )
    return committed[2]["event_uid"]
