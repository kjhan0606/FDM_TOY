#!/usr/bin/env python3
"""Bind all Poisson gravity shards to a verified FDM relaxation sample ledger.

Run manually on Lageunha; this hashes the complete wave, AMR, and gravity
output sets and submits no simulation or FFT work.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

for _thread_variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

from fdm_smbh_delay.fdm_gravity_binding import materialize_fdm_gravity_source_binding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--particle-density-included", action="store_true")
    args = parser.parse_args()
    if "lageunha" not in socket.gethostname().lower():
        parser.error("full gravity source binding is permitted only on Lageunha")
    record = materialize_fdm_gravity_source_binding(
        args.sample_ledger, args.output,
        particle_density_included=args.particle_density_included,
    )
    print(json.dumps({"status": record["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
