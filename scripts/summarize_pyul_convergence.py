#!/usr/bin/env python3
"""Compare live-wave numerical variants over a common resolved interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.convergence import load_convergence_run, summarize_convergence


def _parse_specification(specification: str) -> tuple[str, Path]:
    label, separator, path = specification.partition("=")
    if not separator or not label or not path:
        raise ValueError("each calculation must use LABEL=RUN_DIRECTORY")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "calculations",
        nargs="+",
        help="two or more calculations written as LABEL=RUN_DIRECTORY",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.calculations) < 2:
        parser.error("at least two calculations are required")

    loaded = [
        load_convergence_run(*_parse_specification(specification))
        for specification in args.calculations
    ]
    summary = summarize_convergence(loaded)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
