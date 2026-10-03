"""Compose provenance-gated CDM environmental stages with a Peters GW tail."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from astropy import units as u

from .cdm_coalescence import (
    compose_cdm_coalescence_time,
    peters_gravitational_wave_segment,
    read_cdm_delay_stage_summary,
)
from .delay_budget import cosmic_age_myr, redshift_after_delay
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
        prog="cdm-smbh-compose",
        description=(
            "Compose accepted CDM environmental intervals and a Peters "
            "gravitational-wave tail; HR5 fixed population delays are not inputs."
        ),
    )
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument("--sink-time", help="cosmic numerical-capture time with units")
    time_group.add_argument("--z-sink", type=float, help="numerical-capture redshift")
    parser.add_argument("--capture-to-hard-summary", type=Path, required=True)
    parser.add_argument("--hard-to-gw-summary", type=Path, required=True)
    parser.add_argument("--gw-mass1", required=True, help="primary SMBH mass with units")
    parser.add_argument("--gw-mass2", required=True, help="secondary SMBH mass with units")
    parser.add_argument(
        "--gw-start-separation",
        required=True,
        help="Peters handoff semi-major axis with units",
    )
    parser.add_argument("--gw-eccentricity", type=float, default=0.0)
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
    capture_to_hard = read_cdm_delay_stage_summary(
        args.capture_to_hard_summary,
        expected_stage="capture_to_hard_binary",
    )
    hard_to_gw = read_cdm_delay_stage_summary(
        args.hard_to_gw_summary,
        expected_stage="hard_binary_to_gw_regime",
    )
    mass1 = parse_quantity(args.gw_mass1, u.Msun, "gw_mass1")
    mass2 = parse_quantity(args.gw_mass2, u.Msun, "gw_mass2")
    gw_start = parse_quantity(args.gw_start_separation, u.pc, "gw_start_separation")
    gw = peters_gravitational_wave_segment(
        mass1_msun=mass1,
        mass2_msun=mass2,
        start_separation_pc=gw_start,
        eccentricity=args.gw_eccentricity,
    )
    estimate = compose_cdm_coalescence_time(
        sink_time,
        capture_to_hard,
        hard_to_gw,
        gw,
        gravitational_wave_start_separation_pc=gw_start,
    )
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": estimate.status,
        "dark_matter_model": "cdm",
        "interpretation": (
            "CDM-only composition of accepted resolved-rate integrations and an "
            "orbit-averaged Peters tail; not an HR5 disappearance-history or "
            "fixed population-delay estimate"
        ),
        "estimate": asdict(estimate),
        "environmental_stages": {
            "capture_to_hard_binary": capture_to_hard.as_dict(),
            "hard_binary_to_gw_regime": hard_to_gw.as_dict(),
        },
        "gravitational_wave_initial_orbit": {
            "mass1_msun": mass1,
            "mass2_msun": mass2,
            "semi_major_axis_pc": gw_start,
            "eccentricity": args.gw_eccentricity,
        },
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
    if args.output is not None:
        _write_json_atomic(args.output.expanduser().resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
