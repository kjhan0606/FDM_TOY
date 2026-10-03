#!/usr/bin/env python3
"""Hash-verify every completed adjacent q/e pilot pair; release no table rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from fdm_smbh_delay.convergence import load_convergence_run
from scripts.audit_qe_bin_occupancy import pair_occupancy
from scripts.reassess_qe_extension import (
    _require_complete,
    _sha256,
    all_adjacent_pairs,
)


def audit_adjacent_pilot(
    manifest: Path,
    torch_root: Path,
    *,
    separation_bins: int = 8,
    minimum_orbits_per_bin: int = 8,
) -> dict:
    """Return pilot occupancy only after verifying source identity twice."""

    manifest = manifest.expanduser().resolve()
    torch_root = torch_root.expanduser().resolve()
    manifest_hash = _sha256(manifest)
    pairs = all_adjacent_pairs(manifest)
    hashes: dict[str, dict[str, str]] = {}
    loaded: dict[str, dict] = {}
    for case_id, fine, coarse in pairs:
        for resolution, run_id in (fine, coarse):
            if run_id not in hashes:
                path = torch_root / run_id
                hashes[run_id] = _require_complete(
                    path, case_id=case_id, resolution=resolution
                )
                loaded[run_id] = load_convergence_run(run_id, path)
    records = []
    for case_id, fine, coarse in pairs:
        diagnostic = pair_occupancy(
            loaded[fine[1]], loaded[coarse[1]],
            separation_bins=separation_bins,
            minimum_orbits_per_bin=minimum_orbits_per_bin,
        )
        records.append({
            "case_id": case_id,
            "fine_resolution": fine[0],
            "coarse_resolution": coarse[0],
            "fine_run": str((torch_root / fine[1]).resolve()),
            "coarse_run": str((torch_root / coarse[1]).resolve()),
            "diagnostic": diagnostic,
            "production_calibration_row_admitted": False,
        })
    if _sha256(manifest) != manifest_hash:
        raise ValueError("q/e manifest changed during adjacent-pair audit")
    for case_id, fine, coarse in pairs:
        for resolution, run_id in (fine, coarse):
            if _require_complete(
                torch_root / run_id, case_id=case_id, resolution=resolution
            ) != hashes[run_id]:
                raise ValueError(f"q/e pilot input changed during audit: {run_id}")
    return {
        "status": "qe_all_adjacent_pilot_design_diagnostic_only",
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "torch_root": str(torch_root),
        "run_input_sha256": hashes,
        "pair_count": len(records),
        "pairs": records,
        "production_calibration_row_admitted": False,
    }


def _write_new_report(output: Path, result: dict) -> None:
    """Publish a complete audit without replacing an existing result."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output.parent, delete=False
        ) as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--torch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--separation-bins", type=int, default=8)
    parser.add_argument("--minimum-orbits-per-bin", type=int, default=8)
    args = parser.parse_args()
    result = audit_adjacent_pilot(
        args.manifest,
        args.torch_root,
        separation_bins=args.separation_bins,
        minimum_orbits_per_bin=args.minimum_orbits_per_bin,
    )
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"refusing existing adjacent-pilot audit: {output}")
        _write_new_report(output, result)


if __name__ == "__main__":
    main()
