"""Compose a numerical sink time with explicitly available physical delays."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from astropy import units as u

from .delay_budget import (
    DelaySegment,
    compose_true_merge_time,
    cosmic_age_myr,
    redshift_after_delay,
)
from .units import parse_quantity


def _fdm_segment(summary: dict[str, Any]) -> DelaySegment:
    status = summary.get("status")
    if status in {"reached_0p01pc", "reached_target"}:
        delay = summary.get("t_fdm_myr")
        if not isinstance(delay, (int, float)):
            return DelaySegment("fdm_pc_to_0p01pc", "invalid", None)
        return DelaySegment("fdm_pc_to_0p01pc", "complete", float(delay))
    if status == "timeout":
        elapsed = summary.get("integration_time_myr", 0.0)
        return DelaySegment(
            "fdm_pc_to_0p01pc", "timeout", None, float(elapsed)
        )
    return DelaySegment("fdm_pc_to_0p01pc", "invalid", None)


def _optional_complete_segment(name: str, value: str | None) -> DelaySegment:
    if value is None:
        return DelaySegment(name, "missing", None)
    delay = parse_quantity(value, u.Myr, name)
    if delay < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return DelaySegment(name, "complete", delay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fdm-smbh-compose",
        description="Compose an SMBH sink time and three physical inspiral intervals.",
    )
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument("--sink-time", help="cosmic sink time with units")
    time_group.add_argument("--z-sink", type=float, help="numerical sink redshift")
    parser.add_argument("--fdm-summary", required=True, type=Path)
    parser.add_argument("--kpc-to-pc-delay", help="resolved or calibrated delay with units")
    parser.add_argument("--gw-delay", help="gravitational-wave delay with units")
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
        _optional_complete_segment("kpc_to_pc", args.kpc_to_pc_delay),
        _fdm_segment(fdm_summary),
        _optional_complete_segment("gravitational_wave", args.gw_delay),
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
