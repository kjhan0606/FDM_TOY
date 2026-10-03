"""Verified end-to-end delay composition for the PTA-facing toy output."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .backreaction import (
    read_verified_backreaction_delay_record,
    read_verified_backreaction_decision,
)
from .delay_budget import (
    DelaySegment,
    TrueMergeEstimate,
    compose_true_merge_time,
    read_verified_delay_segment_record,
)
from .nuclear_bridge import read_ledger_bound_bridge
from .true_time_cli import _fdm_segment


def compose_verified_pta_delay(
    *,
    sink_time_myr: float,
    capture_bridge_path: str | Path,
    backreaction_decision_path: str | Path,
    backreaction_delay_record_path: str | Path,
    fdm_summary_path: str | Path,
    gravitational_wave_record_path: str | Path,
) -> TrueMergeEstimate:
    """Compose kpc, FDM, and GW intervals from verified records only.

    The kpc interval is accepted only after its saved live/frozen decision is
    rebuilt from the current track bytes and its integrated delay record is
    re-hashed and checked against that decision's measured overlap.  The other
    intervals use the same serialized ``DelaySegment`` contract; no scalar
    command-line delay enters this path.
    """

    bridge = read_ledger_bound_bridge(capture_bridge_path)
    if not bridge.ready_for_integration:
        raise ValueError("capture bridge lacks a ready environment")
    if not math.isclose(
        sink_time_myr, bridge.capture_time_myr, rel_tol=1.0e-8, abs_tol=1.0e-6
    ):
        raise ValueError("PTA sink time differs from capture bridge time")
    decision = read_verified_backreaction_decision(backreaction_decision_path)
    if decision.model != "fdm":
        raise ValueError("FDM PTA delay requires an FDM kpc backreaction decision")
    if bridge.environment.channel("fdm").status != "available":
        raise ValueError("FDM PTA delay requires an available capture FDM channel")
    kpc = read_verified_backreaction_delay_record(
        backreaction_delay_record_path,
        decision=decision,
        expected_start_separation_pc=bridge.pair.separation_pc,
    )
    if kpc.name != "kpc_to_pc":
        raise ValueError("backreaction delay record must name kpc_to_pc")
    if kpc.source_case_id != bridge.event_uid:
        raise ValueError("backreaction delay event UID differs from capture bridge")
    summary_path = Path(fdm_summary_path).expanduser().resolve()
    try:
        summary: Any = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read FDM summary: {error}") from error
    if not isinstance(summary, dict):
        raise ValueError("FDM summary must be a JSON object")
    fdm = _fdm_segment(summary)
    if fdm.status == "complete":
        # The legacy analytic-orbit summary has neither a rechecked source
        # artifact nor an accepted q/e/separation table identity. Its
        # diagnostic completion is not a calibrated physical PTA delay.
        fdm = DelaySegment(
            "fdm_pc_to_0p01pc",
            "censored",
            None,
            reason=(
                "legacy FDM toy completion lacks a verified calibration-backed "
                "1 pc to 0.01 pc delay"
            ),
        )
    gw = read_verified_delay_segment_record(
        gravitational_wave_record_path, expected_name="gravitational_wave"
    )
    return compose_true_merge_time(sink_time_myr, kpc, fdm, gw)
