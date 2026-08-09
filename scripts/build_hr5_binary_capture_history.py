#!/usr/bin/env python3
"""Build the HR5 binary-capture history from MkAGN snapshot products."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from fdm_smbh_delay.hr5 import (
    HEADER_DTYPE,
    NSTEP_MAX,
    SINK_DTYPE,
    infer_capture_receivers,
    read_mkagn_snapshot,
)


DEFAULT_SNAPSHOT_DIRECTORY = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/SRC(MkAGN)/HR5_AGN_DATA"
)
OUTPUT_PATTERN = re.compile(r"agn\.(\d{5})\.dat$")


def _discover_snapshots(directory: Path) -> list[tuple[int, Path]]:
    snapshots = []
    for path in directory.glob("agn.*.dat"):
        match = OUTPUT_PATTERN.search(path.name)
        if match:
            snapshots.append((int(match.group(1)), path))
    snapshots.sort()
    if not snapshots:
        raise ValueError(f"No MkAGN snapshots were found in {directory}")
    if len(snapshots) > NSTEP_MAX:
        raise ValueError(f"The native HR5 tree supports at most {NSTEP_MAX} outputs")
    return snapshots


def _snapshot_state(records: np.ndarray, dimensionless_hubble: float) -> np.ndarray:
    return np.column_stack(
        [
            records["mass"] / dimensionless_hubble,
            records["x"] / dimensionless_hubble,
            records["y"] / dimensionless_hubble,
            records["z"] / dimensionless_hubble,
            records["vx"],
            records["vy"],
            records["vz"],
        ]
    ).astype(np.float32)


def _disappeared(previous_id: np.ndarray, current_id: np.ndarray) -> np.ndarray:
    position = np.searchsorted(current_id, previous_id)
    survives = position < current_id.size
    survives[survives] &= current_id[position[survives]] == previous_id[survives]
    return np.flatnonzero(~survives)


def build_history(
    snapshot_directory: Path,
    output_tree: Path,
    history_csv: Path,
    provenance_json: Path,
    h0: float,
    omega_m: float,
    omega_lambda: float,
    volume_cmpc_over_h3: float,
    box_size_cmpc: float,
    overwrite: bool,
    allow_output_gaps: bool,
) -> None:
    snapshots = _discover_snapshots(snapshot_directory)
    output_sequence = np.asarray([output for output, _ in snapshots])
    if not allow_output_gaps and np.any(np.diff(output_sequence) != 1):
        missing_after = output_sequence[:-1][np.diff(output_sequence) != 1]
        raise ValueError(
            "MkAGN outputs must be consecutive for disappearance-based capture inference. "
            f"The first gap follows output {int(missing_after[0]):05d}."
        )
    metadata: list[tuple[int, Path, float, float, int, int]] = []
    maximum_id = 0
    dimensionless_hubble = h0 / 100.0
    for output_number, path in snapshots:
        redshift, local_timestep_yr, records = read_mkagn_snapshot(path)
        if records.size and np.any(np.diff(records["sink_id"]) <= 0):
            records.sort(order="sink_id")
        snapshot_maximum_id = int(np.max(records["sink_id"])) if records.size else 0
        maximum_id = max(maximum_id, snapshot_maximum_id)
        metadata.append(
            (output_number, path, redshift, local_timestep_yr, records.size, snapshot_maximum_id)
        )
        print(
            f"Scanned output {output_number:05d} with {records.size:,} sinks at z={redshift:.5f}",
            flush=True,
        )
    if output_tree.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace {output_tree}. Pass --overwrite to replace it.")
    output_tree.parent.mkdir(parents=True, exist_ok=True)
    total_size = HEADER_DTYPE.itemsize + maximum_id * SINK_DTYPE.itemsize
    with output_tree.open("wb") as stream:
        stream.truncate(total_size)

    redshift = np.asarray([item[2] for item in metadata], dtype=np.float64)
    output_number = np.asarray([item[0] for item in metadata], dtype=np.int64)
    header_map = np.memmap(output_tree, mode="r+", dtype=HEADER_DTYPE, shape=(1,))
    header_map["redshift"][0, : redshift.size] = redshift
    header_map["output_number"][0, : output_number.size] = output_number
    header_map["omega_m"][0] = omega_m
    header_map["omega_lambda"][0] = omega_lambda
    header_map["h0"][0] = h0
    header_map["nstep"][0] = len(metadata)
    header_map["nsink"][0] = maximum_id
    header_map.flush()
    del header_map
    sink_map = np.memmap(
        output_tree,
        mode="r+",
        dtype=SINK_DTYPE,
        offset=HEADER_DTYPE.itemsize,
        shape=(maximum_id,),
    )
    sink_map["sink_id"] = np.arange(1, maximum_id + 1, dtype=np.int32)

    active_count = np.zeros(len(metadata), dtype=np.int64)
    birth_count = np.zeros(len(metadata), dtype=np.int64)
    capture_count = np.zeros(len(metadata), dtype=np.int64)
    total_mass = np.zeros(len(metadata), dtype=np.float64)
    previous_records: np.ndarray | None = None
    unmatched_capture_count = 0
    for history_index, (snapshot_meta, output) in enumerate(zip(metadata, output_number)):
        _, path, snapshot_redshift, _, _, _ = snapshot_meta
        _, _, records = read_mkagn_snapshot(path)
        records.sort(order="sink_id")
        sink_id = records["sink_id"].astype(np.int64)
        sink_index = sink_id - 1
        sink_map["state"][sink_index, history_index, :] = _snapshot_state(
            records, dimensionless_hubble
        )
        active_count[history_index] = records.size
        total_mass[history_index] = np.sum(records["mass"], dtype=np.float64) / dimensionless_hubble
        if previous_records is None:
            birth_count[history_index] = records.size
        else:
            previous_id = previous_records["sink_id"].astype(np.int64)
            new_position = np.searchsorted(previous_id, sink_id)
            new_sink = new_position >= previous_id.size
            inside = ~new_sink
            new_sink[inside] = previous_id[new_position[inside]] != sink_id[inside]
            birth_count[history_index] = np.count_nonzero(new_sink)

            disappeared_index = _disappeared(previous_id, sink_id)
            disappeared_records = previous_records[disappeared_index]
            receiver_id = infer_capture_receivers(
                disappeared_records["sink_id"],
                disappeared_records["mass"] / dimensionless_hubble,
                np.column_stack(
                    [disappeared_records["x"], disappeared_records["y"], disappeared_records["z"]]
                ) / dimensionless_hubble,
                sink_id,
                records["mass"] / dimensionless_hubble,
                np.column_stack([records["x"], records["y"], records["z"]])
                / dimensionless_hubble,
                box_size_cmpc_over_h=box_size_cmpc,
            )
            valid_receiver = receiver_id > 0
            disappearing_sink_index = disappeared_records["sink_id"].astype(np.int64) - 1
            sink_map["receiver_id"][disappearing_sink_index[valid_receiver]] = receiver_id[
                valid_receiver
            ]
            sink_map["capture_index"][disappearing_sink_index[valid_receiver]] = history_index
            capture_count[history_index] = np.count_nonzero(valid_receiver)
            unmatched_capture_count += np.count_nonzero(~valid_receiver)
        previous_records = records
        print(
            f"Wrote output {int(output):05d} at z={snapshot_redshift:.5f} with "
            f"{capture_count[history_index]:,} inferred captures",
            flush=True,
        )
    sink_map.flush()
    del sink_map

    cosmology = FlatLambdaCDM(H0=h0, Om0=omega_m, Tcmb0=2.7255)
    cosmic_time_gyr = np.asarray(cosmology.age(redshift).value)
    interval_gyr = np.full(redshift.size, np.nan)
    interval_gyr[1:] = np.diff(cosmic_time_gyr)
    capture_rate = capture_count / (volume_cmpc_over_h3 * interval_gyr)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    with history_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "history_index",
                "output_number",
                "redshift",
                "cosmic_time_gyr",
                "interval_gyr",
                "active_sink_count",
                "seed_birth_count",
                "capture_count",
                "capture_rate_cmpc3_gyr",
                "total_sink_mass_msun",
            )
        )
        writer.writerows(
            zip(
                np.arange(redshift.size),
                output_number,
                redshift,
                cosmic_time_gyr,
                interval_gyr,
                active_count,
                birth_count,
                capture_count,
                capture_rate,
                total_mass,
            )
        )
    provenance = {
        "mkagn_source": str(snapshot_directory),
        "tree_output": str(output_tree),
        "n_output": len(metadata),
        "n_sink_identifier": maximum_id,
        "n_inferred_capture": int(np.sum(capture_count)),
        "n_unmatched_disappearance": int(unmatched_capture_count),
        "receiver_selection": {
            "mass_factor": 2.0,
            "radius_increment_cmpc": 0.002,
            "maximum_radius_cmpc": 0.5,
            "periodic_box_cmpc": box_size_cmpc,
        },
        "event_time_convention": {
            "minor_state": "output i-1",
            "receiver_state": "output i",
            "capture_index": "output i and therefore the time-interval upper bound",
        },
        "units": {
            "mkagn_input_mass": "Msun/h",
            "mkagn_input_position": "cMpc/h",
            "tree_mass": "Msun",
            "tree_position": "cMpc",
            "velocity": "km/s physical",
        },
    }
    provenance_json.parent.mkdir(parents=True, exist_ok=True)
    provenance_json.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-directory", type=Path, default=DEFAULT_SNAPSHOT_DIRECTORY)
    parser.add_argument("--output-tree", type=Path, default=Path("results/hr5/Sink_Merging_Tree.rebuilt.dat"))
    parser.add_argument("--history", type=Path, default=Path("results/hr5/hr5_rebuilt_sink_history.csv"))
    parser.add_argument("--provenance", type=Path, default=Path("results/hr5/hr5_rebuilt_provenance.json"))
    parser.add_argument("--h0", type=float, default=68.4)
    parser.add_argument("--omega-m", type=float, default=0.3)
    parser.add_argument("--omega-lambda", type=float, default=0.7)
    parser.add_argument("--volume-cmpc-over-h3", type=float, default=1.087e7)
    parser.add_argument("--box-size-cmpc", type=float, default=1048.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-output-gaps",
        action="store_true",
        help="Allow diagnostic reconstructions across gaps. Such disappearances are not physical captures.",
    )
    args = parser.parse_args()
    build_history(
        args.snapshot_directory,
        args.output_tree,
        args.history,
        args.provenance,
        args.h0,
        args.omega_m,
        args.omega_lambda,
        args.volume_cmpc_over_h3,
        args.box_size_cmpc,
        args.overwrite,
        args.allow_output_gaps,
    )


if __name__ == "__main__":
    main()
