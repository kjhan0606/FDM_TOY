#!/usr/bin/env python3
"""Generate a deterministic, resolution-paired galaxy-merger zoom manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.zoom_calibration import load_zoom_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-cases", type=int, default=1000)
    args = parser.parse_args()
    grid = load_zoom_grid(args.specification, maximum_cases=args.maximum_cases)
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(grid.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "galaxy_merger_zoom_manifest_written",
                "cases": len(grid.cases),
                "manifest_sha256": grid.manifest_sha256,
                "output": str(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
