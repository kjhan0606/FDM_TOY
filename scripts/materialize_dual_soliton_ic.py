#!/usr/bin/env python3
"""Materialize a non-submitting lagRamses dual-soliton/dual-SMBH seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdm_smbh_delay.dual_soliton_seed import (
    load_pure_fdm_dual_soliton_seed,
    materialize_pure_fdm_dual_soliton_seed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    manifest = materialize_pure_fdm_dual_soliton_seed(
        load_pure_fdm_dual_soliton_seed(args.seed), args.output_directory
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
