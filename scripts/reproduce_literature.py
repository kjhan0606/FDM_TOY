#!/usr/bin/env python3
"""Print the public Koo/Boey curve cross-check values."""

from __future__ import annotations

import argparse

from fdm_smbh_delay.empirical import BOEY_2025_FITS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=float, default=1.0, help="initial separation [pc]")
    parser.add_argument("--final", type=float, default=0.076, help="final separation [pc]")
    args = parser.parse_args()
    for ratio, fit in BOEY_2025_FITS.items():
        delay = fit.time_between_myr(args.initial, args.final)
        print(f"Boey 2025 {ratio:2d}% fit: {args.initial:g} -> {args.final:g} pc = {delay:.8g} Myr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
