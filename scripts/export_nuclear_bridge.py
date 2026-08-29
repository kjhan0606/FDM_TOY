#!/usr/bin/env python3
"""Export one complete lagRamses capture event into the bridge contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from fdm_smbh_delay.capture_ledger import read_capture_ledger
from fdm_smbh_delay.nuclear_bridge import EnvironmentSnapshot, NuclearBridgeInput


def _load_environment(path: Path, event_uid: str) -> EnvironmentSnapshot:
    try:
        raw: Any = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"environment snapshot is unreadable: {path}") from error

    if isinstance(raw, Mapping) and "snapshots" in raw:
        candidates = raw["snapshots"]
        if not isinstance(candidates, list):
            raise ValueError("environment snapshots must be a list")
        matches = [
            candidate
            for candidate in candidates
            if isinstance(candidate, Mapping)
            and candidate.get("event_uid") == event_uid
        ]
        if len(matches) != 1:
            raise ValueError(
                f"environment snapshot must contain exactly one record for event_uid={event_uid}"
            )
        raw = matches[0]
    snapshot = EnvironmentSnapshot.from_dict(raw)
    if snapshot.event_uid != event_uid:
        raise ValueError("environment and ledger event_uid values do not agree")
    return snapshot


def export_bridge_input(
    *,
    ledger_path: str | Path,
    environment_path: str | Path,
    event_uid: str,
    run_id: str,
    capture_time_myr: float,
    output_path: str | Path,
    target_semimajor_axis_pc: float = 1.0,
) -> Path:
    """Build one bridge JSON from a complete ledger event and environment."""

    if not event_uid or not run_id:
        raise ValueError("event_uid and run_id are required")
    ledger = read_capture_ledger(ledger_path)
    matches = [event for event in ledger.events if event.event_uid == event_uid]
    if len(matches) != 1:
        raise ValueError(
            f"ledger must contain exactly one complete event_uid={event_uid}"
        )
    event = matches[0]
    environment = _load_environment(Path(environment_path), event_uid)
    bridge = NuclearBridgeInput.from_capture_event(
        event,
        run_id=run_id,
        capture_time_myr=capture_time_myr,
        environment=environment,
        target_semimajor_axis_pc=target_semimajor_axis_pc,
    )
    return bridge.write_json(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export one complete lagRamses event to the nuclear bridge schema."
    )
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--event-uid", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--capture-time-myr", required=True, type=float)
    parser.add_argument("--target-semimajor-axis-pc", type=float, default=1.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = export_bridge_input(
        ledger_path=args.ledger,
        environment_path=args.environment,
        event_uid=args.event_uid,
        run_id=args.run_id,
        capture_time_myr=args.capture_time_myr,
        output_path=args.output,
        target_semimajor_axis_pc=args.target_semimajor_axis_pc,
    )
    bridge = NuclearBridgeInput.read_json(output)
    print(
        json.dumps(
            {
                "status": bridge.status,
                "event_uid": bridge.event_uid,
                "run_id": bridge.run_id,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
