"""Compose a numerical sink time with explicitly available physical delays."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Sequence

from astropy import units as u

from .delay_budget import (
    DelaySegment,
    compose_true_merge_time,
    cosmic_age_myr,
    read_verified_delay_segment_record,
    redshift_after_delay,
)
from .units import parse_quantity


def _fdm_segment(summary: dict[str, Any]) -> DelaySegment:
    raw_status = summary.get("status")
    if not isinstance(raw_status, str):
        return DelaySegment("fdm_pc_to_0p01pc", "invalid", None)
    status = raw_status.strip().lower().replace("-", "_")
    reason = summary.get("reason")
    if not isinstance(reason, str):
        reason = None
    if status in {"reached_0p01pc", "reached_target"}:
        initial = summary.get("D_initial_pc")
        stop = summary.get("D_stop_pc")
        if (
            isinstance(initial, bool)
            or not isinstance(initial, (int, float))
            or not math.isfinite(initial)
            or not math.isclose(float(initial), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
            or isinstance(stop, bool)
            or not isinstance(stop, (int, float))
            or not math.isfinite(stop)
            or not math.isclose(float(stop), 0.01, rel_tol=0.0, abs_tol=1.0e-14)
        ):
            return DelaySegment(
                "fdm_pc_to_0p01pc",
                "censored",
                None,
                reason=(
                    "FDM summary does not document the required 1 pc to 0.01 pc "
                    "integration interval"
                ),
            )
        flags = summary.get("validity_flags")
        if not isinstance(flags, list) or any(
            not isinstance(flag, str) or not flag.strip() for flag in flags
        ):
            return DelaySegment(
                "fdm_pc_to_0p01pc",
                "invalid",
                None,
                reason="FDM summary validity_flags must be a list of strings",
            )
        delay = summary.get("t_fdm_myr")
        if (
            isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or not math.isfinite(delay)
            or delay < 0.0
        ):
            return DelaySegment("fdm_pc_to_0p01pc", "invalid", None)
        source_case_id = summary.get("source_case_id")
        source_sha256 = summary.get("source_sha256")
        if not isinstance(source_case_id, str) or not source_case_id.strip():
            return DelaySegment(
                "fdm_pc_to_0p01pc",
                "censored",
                None,
                reason="FDM completed summary lacks a source_case_id",
            )
        if (
            not isinstance(source_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", source_sha256) is None
        ):
            return DelaySegment(
                "fdm_pc_to_0p01pc",
                "censored",
                None,
                reason="FDM completed summary lacks a source SHA-256",
            )
        if flags:
            elapsed = summary.get("integration_time_myr", 0.0)
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed < 0.0
            ):
                return DelaySegment("fdm_pc_to_0p01pc", "invalid", None)
            return DelaySegment(
                "fdm_pc_to_0p01pc",
                "censored",
                None,
                float(elapsed),
                reason="FDM validity flags block completed use: " + ", ".join(flags),
                source_case_id=source_case_id,
                source_sha256=source_sha256,
            )
        try:
            return DelaySegment(
                "fdm_pc_to_0p01pc",
                "complete",
                float(delay),
                source_case_id=source_case_id,
                source_sha256=source_sha256,
            )
        except ValueError:
            return DelaySegment(
                "fdm_pc_to_0p01pc",
                "censored",
                None,
                reason="FDM completed summary source SHA-256 is invalid",
            )
    if status in {
        "timeout",
        "censored",
        "stalled",
        "uncalibrated",
        "outside_support",
        "outside_calibration",
        "outside_calibration_domain",
        "outside_calibration_support",
    }:
        elapsed = summary.get("integration_time_myr", 0.0)
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0.0
        ):
            return DelaySegment("fdm_pc_to_0p01pc", "invalid", None)
        segment_status = "timeout" if status == "timeout" else "censored"
        return DelaySegment(
            "fdm_pc_to_0p01pc",
            segment_status,
            None,
            float(elapsed),
            reason=reason or f"FDM summary reported {raw_status!r}",
        )
    return DelaySegment("fdm_pc_to_0p01pc", "invalid", None)


def _optional_complete_segment(
    name: str, value: str | None, record_path: Path | None = None
) -> DelaySegment:
    if value is not None and record_path is not None:
        raise ValueError(f"{name} accepts either a delay record or a bare value, not both")
    if record_path is not None:
        return read_verified_delay_segment_record(record_path, expected_name=name)
    if value is None:
        return DelaySegment(name, "missing", None)
    delay = parse_quantity(value, u.Myr, name)
    if delay < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return DelaySegment(
        name,
        "censored",
        None,
        reason=(
            "bare command-line delay is operator asserted and cannot establish "
            "a physical interval"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fdm-smbh-compose",
        description="Compose an SMBH sink time and three physical inspiral intervals.",
    )
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument("--sink-time", help="cosmic sink time with units")
    time_group.add_argument("--z-sink", type=float, help="numerical sink redshift")
    parser.add_argument("--fdm-summary", required=True, type=Path)
    parser.add_argument(
        "--kpc-to-pc-delay",
        help="operator-asserted delay (always censored; use --kpc-to-pc-record)",
    )
    parser.add_argument(
        "--kpc-to-pc-record",
        type=Path,
        help="hash-bound serialized kpc_to_pc DelaySegment record",
    )
    parser.add_argument(
        "--gw-delay",
        help="operator-asserted delay (always censored; use --gw-record)",
    )
    parser.add_argument(
        "--gw-record",
        type=Path,
        help="hash-bound serialized gravitational_wave DelaySegment record",
    )
    parser.add_argument("--h0", type=float, default=67.66, help="H0 [km/s/Mpc]")
    parser.add_argument("--omega-m", type=float, default=0.3111)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sink_time = (
        parse_quantity(args.sink_time, u.Myr, "sink_time")
        if args.sink_time is not None
        else cosmic_age_myr(
            redshift=args.z_sink,
            h0_km_s_mpc=args.h0,
            omega_m=args.omega_m,
        )
    )
    with args.fdm_summary.expanduser().open("r", encoding="utf-8") as stream:
        fdm_summary = json.load(stream)
    estimate = compose_true_merge_time(
        sink_time,
        _optional_complete_segment(
            "kpc_to_pc", args.kpc_to_pc_delay, args.kpc_to_pc_record
        ),
        _fdm_segment(fdm_summary),
        _optional_complete_segment(
            "gravitational_wave", args.gw_delay, args.gw_record
        ),
    )
    output: dict[str, Any] = asdict(estimate)
    if estimate.status == "complete" and args.z_sink is not None:
        assert estimate.total_delay_myr is not None
        output["z_true"] = redshift_after_delay(
            z_sink=args.z_sink,
            delay_myr=estimate.total_delay_myr,
            h0_km_s_mpc=args.h0,
            omega_m=args.omega_m,
        )
        output["coalesces_by_z0"] = output["z_true"] is not None
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
