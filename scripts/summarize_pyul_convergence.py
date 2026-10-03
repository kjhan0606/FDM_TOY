#!/usr/bin/env python3
"""Compare live-wave numerical variants over a common resolved interval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.convergence import load_convergence_run, summarize_convergence
from fdm_smbh_delay.qe_followup_design import (
    read_verified_qe_followup_design,
    verify_qe_design_comparison_runs,
)


def _parse_specification(specification: str) -> tuple[str, Path]:
    label, separator, path = specification.partition("=")
    if not separator or not label or not path:
        raise ValueError("each calculation must use LABEL=RUN_DIRECTORY")
    return label, Path(path)


def _parse_edges(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "separation-bin edges must be comma-separated numbers"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "calculations",
        nargs="+",
        help="two or more calculations written as LABEL=RUN_DIRECTORY",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--separation-bins", type=int)
    parser.add_argument(
        "--separation-bin-edges-pc", type=_parse_edges,
        help="prospectively fixed comma-separated physical separation edges",
    )
    parser.add_argument("--minimum-orbits-per-separation-bin", type=int)
    parser.add_argument("--qe-design", type=Path)
    parser.add_argument("--qe-design-cases", type=Path)
    parser.add_argument("--qe-design-manifest", type=Path)
    args = parser.parse_args()
    if len(args.calculations) < 2:
        parser.error("at least two calculations are required")

    specifications = [_parse_specification(value) for value in args.calculations]
    edges = args.separation_bin_edges_pc
    bins = 8 if args.separation_bins is None else args.separation_bins
    minimum_orbits = (
        8 if args.minimum_orbits_per_separation_bin is None
        else args.minimum_orbits_per_separation_bin
    )
    design_binding = None
    if args.qe_design is not None:
        if (args.qe_design_cases is None or args.qe_design_manifest is None
                or edges is not None or args.separation_bins is not None
                or args.minimum_orbits_per_separation_bin is not None
                or len(specifications) != 2):
            parser.error("registered q/e comparison requires two runs and design-only bin controls")
        design, file_sha256 = read_verified_qe_followup_design(
            args.qe_design,
            physical_cases=args.qe_design_cases,
            run_manifest=args.qe_design_manifest,
        )
        kind = verify_qe_design_comparison_runs(
            design, file_sha256,
            tuple(path.expanduser().resolve() for _, path in specifications),
        )
        edges = tuple(design["separation_bin_edges_pc"])
        bins = len(edges) - 1
        minimum_orbits = design["minimum_orbits_per_bin"]
        design_binding = {
            "path": str(args.qe_design.expanduser().resolve()),
            "physical_cases_path": str(args.qe_design_cases.expanduser().resolve()),
            "run_manifest_path": str(args.qe_design_manifest.expanduser().resolve()),
            "design_sha256": design["design_sha256"],
            "file_sha256": file_sha256,
            "comparison_kind": kind,
            "status": "registered_design_bound_not_a_calibration_release",
        }
    elif args.qe_design_cases is not None or args.qe_design_manifest is not None:
        parser.error("q/e design source files require --qe-design")
    loaded = [load_convergence_run(label, path) for label, path in specifications]
    summary = summarize_convergence(
        loaded,
        separation_bins=bins,
        minimum_orbits_per_separation_bin=minimum_orbits,
        separation_bin_edges_pc=edges,
    )
    if design_binding is not None:
        summary["qe_design_binding"] = design_binding
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if design_binding is not None:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=output.parent, delete=False
                ) as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                    temporary = Path(stream.name)
                os.link(temporary, output)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        else:
            temporary = output.with_name(f".{output.name}.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
