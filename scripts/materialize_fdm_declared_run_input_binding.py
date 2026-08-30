#!/usr/bin/env python3
"""Join checked all-wave run inputs to a declared FDM zoom seed binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.fdm_zoom_seed_binding import (
    materialize_fdm_declared_run_input_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fdm_capture_seed_zoom_binding", type=Path)
    parser.add_argument("dual_soliton_run_preflight", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    record = materialize_fdm_declared_run_input_binding(
        fdm_capture_seed_zoom_binding_path=args.fdm_capture_seed_zoom_binding,
        dual_soliton_preflight_path=args.dual_soliton_run_preflight,
        output_directory=args.output_directory,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "fdm_declared_run_input_identity_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
