#!/usr/bin/env python3
"""Run one bounded Lageunha relaxation extractor and publish its attestation.

The extractor command is supplied as a token list after ``--extractor``.  The
wrapper appends two positional arguments to it: the verified sample-ledger
path and a private temporary result JSON path.  A validated result is then
published at ``--result`` without overwriting an existing file.  No shell is
used.  For example::

    python scripts/run_dual_soliton_relaxation_extractor.py \
      --sample-ledger results/sample-ledger.json \
      --result results/extractor-result.json \
      --attestation results/extractor-attestation.json \
      --extractor python scripts/my_lageunha_extractor.py

This command performs no GPU work and does not submit a scheduler job.  A
non-zero extractor exit, malformed result, changed source ledger, or missing
result leaves the attestation absent and returns a non-zero status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from fdm_smbh_delay.relaxation_extractor_attestation import (
    ExtractorExecutionError,
    run_dual_soliton_relaxation_extractor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ledger", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument(
        "--working-directory",
        type=Path,
        help="directory in which the extractor command is launched (default: cwd)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="optional positive wall-clock limit; no limit is imposed by default",
    )
    parser.add_argument(
        "--extractor",
        nargs=argparse.REMAINDER,
        required=True,
        help=(
            "extractor command tokens (this must be the final option); the "
            "wrapper appends sample-ledger and result paths as positional arguments"
        ),
    )
    args = parser.parse_args()
    try:
        record = run_dual_soliton_relaxation_extractor(
            sample_ledger_path=args.sample_ledger,
            result_path=args.result,
            attestation_path=args.attestation,
            extractor_command=args.extractor,
            working_directory=args.working_directory,
            timeout_seconds=args.timeout_seconds,
        )
    except (ExtractorExecutionError, OSError, ValueError) as error:
        print(f"extractor attestation failed: {error}", file=sys.stderr)
        if isinstance(error, ExtractorExecutionError):
            return error.returncode if 0 < error.returncode < 256 else 75
        return 75
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
