"""Verified end-to-end delay composition for the PTA-facing toy output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .backreaction import (
    read_verified_backreaction_delay_record,
    read_verified_backreaction_decision,
)
from .delay_budget import (
    TrueMergeEstimate,
    compose_true_merge_time,
    read_verified_delay_segment_record,
)
from .true_time_cli import _fdm_segment


def compose_verified_pta_delay(
    *,
    sink_time_myr: float,
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

    decision = read_verified_backreaction_decision(backreaction_decision_path)
    kpc = read_verified_backreaction_delay_record(
        backreaction_delay_record_path, decision=decision
    )
    if kpc.name != "kpc_to_pc":
        raise ValueError("backreaction delay record must name kpc_to_pc")
    summary_path = Path(fdm_summary_path).expanduser().resolve()
    try:
        summary: Any = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read FDM summary: {error}") from error
    if not isinstance(summary, dict):
        raise ValueError("FDM summary must be a JSON object")
    fdm = _fdm_segment(summary)
    gw = read_verified_delay_segment_record(
        gravitational_wave_record_path, expected_name="gravitational_wave"
    )
    return compose_true_merge_time(sink_time_myr, kpc, fdm, gw)
