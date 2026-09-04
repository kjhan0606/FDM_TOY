#!/usr/bin/env python3
"""Write a non-submitting FDM capture-to-seed-to-zoom identity decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.fdm_zoom_seed_binding import (
    materialize_fdm_capture_seed_zoom_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_zoom_execution_contract", type=Path)
    parser.add_argument("capture_seed_binding", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    record = materialize_fdm_capture_seed_zoom_binding(
        model_zoom_contract_path=args.model_zoom_execution_contract,
        capture_seed_binding_path=args.capture_seed_binding,
        output_directory=args.output_directory,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "fdm_capture_seed_zoom_identity_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
