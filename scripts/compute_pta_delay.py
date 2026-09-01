#!/usr/bin/env python3
"""Compose a PTA-facing delay from verified kpc, FDM, and GW records."""

from fdm_smbh_delay.pta_delay_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
