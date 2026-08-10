#!/usr/bin/env python3
"""Resolve a lagRamses capture boundary and optional two-SMBH orbital state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fdm_smbh_delay.constants import KM_S_TO_PC_MYR
from fdm_smbh_delay.lagramses import (
    capture_boundary_from_files,
    pair_orbital_state,
)


def _load_event(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    members = raw.get("members")
    if not isinstance(members, list) or len(members) != 2:
        raise ValueError("event.members must contain exactly two SMBHs; retain larger groups as MULTIPLE")
    ids = tuple(int(member["id"]) for member in members)
    masses = tuple(float(member["mass_msun"]) for member in members)
    positions = np.asarray([member["position_pc"] for member in members], dtype=float)
    velocities = np.asarray([member["velocity_km_s"] for member in members], dtype=float)
    box_pc = raw.get("periodic_box_pc")
    state = pair_orbital_state(
        member_ids=ids,
        masses_msun=masses,
        positions_pc=positions,
        velocities_pc_myr=velocities * KM_S_TO_PC_MYR,
        periodic_box_pc=box_pc,
    )
    return raw, state.as_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", required=True, type=Path, help="RAMSES info_*.txt")
    parser.add_argument("--namelist", type=Path, help="archived run namelist")
    parser.add_argument(
        "--box-size-mpc-h",
        required=True,
        type=float,
        help="initial comoving box length in Mpc/h",
    )
    parser.add_argument("--rmerge", type=float, help="override only when the archived value is known")
    parser.add_argument("--event", type=Path, help="pre-compaction two-SMBH JSON record")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    boundary = capture_boundary_from_files(
        info_path=args.info,
        namelist_path=args.namelist,
        box_size_mpc_h=args.box_size_mpc_h,
        rmerge_override=args.rmerge,
    )
    result: dict[str, Any] = {"capture_boundary": boundary.as_dict()}
    if args.event is not None:
        event, state = _load_event(args.event)
        result["event_provenance"] = event.get("provenance", {})
        result["orbital_state"] = state

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
