"""Command-line interface for verified PTA delay composition."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from astropy import units as u

from .delay_budget import cosmic_age_myr, redshift_after_delay
from .pta_delay import compose_verified_pta_delay
from .units import parse_quantity


def _write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fdm-pta-delay",
        description=(
            "Compose verified kpc, FDM, and gravitational-wave delay records."
        ),
    )
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument("--sink-time", help="cosmic numerical-capture time with units")
    time_group.add_argument("--z-sink", type=float, help="numerical-capture redshift")
    parser.add_argument("--backreaction-decision", type=Path, required=True)
    parser.add_argument("--backreaction-delay-record", type=Path, required=True)
    parser.add_argument("--fdm-summary", type=Path, required=True)
    parser.add_argument("--gw-record", type=Path, required=True)
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
    estimate = compose_verified_pta_delay(
        sink_time_myr=sink_time,
        backreaction_decision_path=args.backreaction_decision,
        backreaction_delay_record_path=args.backreaction_delay_record,
        fdm_summary_path=args.fdm_summary,
        gravitational_wave_record_path=args.gw_record,
    )
    output: dict[str, Any] = {
        "schema_version": 1,
        "interpretation": (
            "verified record composition only; a complete result requires an "
            "accepted paired backreaction closure and accepted FDM interval"
        ),
        "estimate": asdict(estimate),
    }
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
        _write_json_atomic(args.output.expanduser().resolve(), output)
    print(rendered)
    return 0 if estimate.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
