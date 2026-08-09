"""Command-line interface for one FDM-delay case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .io import write_result
from .orbit import integrate_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fdm-smbh-delay",
        description="Integrate a parsec-scale SMBH binary in a static FDM soliton.",
    )
    parser.add_argument("config", type=Path, help="YAML case configuration")
    parser.add_argument("--output", type=Path, required=True, help="result directory")
    parser.add_argument(
        "--no-timeseries",
        action="store_true",
        help="write only summary.json and config.yaml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = integrate_case(config)
    output = write_result(
        result,
        config,
        args.output,
        write_timeseries=not args.no_timeseries,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
