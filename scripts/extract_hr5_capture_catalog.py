#!/usr/bin/env python3
"""Extract a compact HR5 numerical sink-capture catalog from the legacy tree."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from fdm_smbh_delay.hr5 import HEADER_DTYPE, SINK_DTYPE, read_tree_header


def _read_header(path: Path) -> np.void:
    return read_tree_header(path)


def _collect_sink_statistics(
    path: Path,
    header: np.void,
    chunk_records: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    nstep = int(header["nstep"])
    nsink = int(header["nsink"])
    active_count = np.zeros(nstep, dtype=np.int64)
    birth_count = np.zeros(nstep, dtype=np.int64)
    total_mass = np.zeros(nstep, dtype=np.float64)
    events: dict[str, list[np.ndarray]] = {
        "sink_id": [],
        "receiver_id": [],
        "capture_index": [],
        "minor_mass": [],
        "x": [],
        "y": [],
        "z": [],
        "vx": [],
        "vy": [],
        "vz": [],
    }

    with path.open("rb") as stream:
        stream.seek(HEADER_DTYPE.itemsize)
        records_read = 0
        while records_read < nsink:
            count = min(chunk_records, nsink - records_read)
            block = np.fromfile(stream, dtype=SINK_DTYPE, count=count)
            if block.size != count:
                raise ValueError(f"The HR5 tree ended after {records_read + block.size} sink records")

            mass = block["state"][:, :nstep, 0]
            active = mass > 0.0
            active_count += np.count_nonzero(active, axis=0)
            total_mass += np.sum(mass, axis=0, dtype=np.float64)
            birth_count[0] += np.count_nonzero(active[:, 0])
            birth_count[1:] += np.count_nonzero(active[:, 1:] & ~active[:, :-1], axis=0)

            capture_index = block["capture_index"]
            selected = np.flatnonzero(capture_index > 0)
            if selected.size:
                index = capture_index[selected].astype(np.int64)
                previous_state = block["state"][selected, index - 1, :]
                events["sink_id"].append(block["sink_id"][selected].astype(np.int64))
                events["receiver_id"].append(block["receiver_id"][selected].astype(np.int64))
                events["capture_index"].append(index)
                for column, name in enumerate(("minor_mass", "x", "y", "z", "vx", "vy", "vz")):
                    events[name].append(previous_state[:, column].astype(np.float64))

            records_read += count
            if records_read == nsink or records_read % (chunk_records * 100) == 0:
                print(f"Read {records_read:,} of {nsink:,} sink histories", flush=True)

    compact_events = {
        name: np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        for name, parts in events.items()
    }
    statistics = {
        "active_count": active_count,
        "birth_count": birth_count,
        "total_mass": total_mass,
    }
    return statistics, compact_events


def _read_receiver_masses(
    path: Path,
    header: np.void,
    receiver_id: np.ndarray,
    state_index: np.ndarray,
) -> np.ndarray:
    nsink = int(header["nsink"])
    receiver_index = receiver_id.astype(np.int64) - 1
    if np.any(receiver_index < 0) or np.any(receiver_index >= nsink):
        raise ValueError("At least one HR5 receiver ID lies outside the sink tree")

    masses = np.full(receiver_index.size, np.nan, dtype=np.float64)
    order = np.argsort(receiver_index, kind="stable")
    sorted_receiver = receiver_index[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_receiver)) + 1, order.size]

    with path.open("rb") as stream:
        for group_number, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            sink_index = int(sorted_receiver[begin])
            offset = HEADER_DTYPE.itemsize + sink_index * SINK_DTYPE.itemsize
            stream.seek(offset)
            receiver = np.fromfile(stream, dtype=SINK_DTYPE, count=1)
            if receiver.size != 1:
                raise ValueError(f"Could not read receiver sink record {sink_index + 1}")
            event_rows = order[begin:end]
            steps = state_index[event_rows].astype(np.int64)
            masses[event_rows] = receiver["state"][0, steps, 0]
            if group_number and group_number % 50000 == 0:
                print(f"Read {group_number:,} distinct receiver histories", flush=True)
    return masses


def _write_catalog(path: Path, events: dict[str, np.ndarray]) -> None:
    columns = (
        "sink_id",
        "receiver_id",
        "last_resolved_history_index",
        "assigned_capture_history_index",
        "last_resolved_output",
        "assigned_capture_output",
        "last_resolved_redshift",
        "assigned_capture_redshift",
        "last_resolved_cosmic_time_gyr",
        "assigned_capture_cosmic_time_gyr",
        "capture_interval_gyr",
        "minor_mass_last_resolved_msun",
        "receiver_mass_last_resolved_msun",
        "receiver_mass_assigned_output_msun",
        "mass_ratio_last_resolved",
        "chirp_mass_last_resolved_msun",
        "minor_x_last_resolved_cmpc",
        "minor_y_last_resolved_cmpc",
        "minor_z_last_resolved_cmpc",
        "minor_vx_last_resolved_kms",
        "minor_vy_last_resolved_kms",
        "minor_vz_last_resolved_kms",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        rows = zip(*(events[name] for name in columns))
        writer.writerows(rows)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    levels = (0.05, 0.16, 0.50, 0.84, 0.95)
    return {f"q{int(level * 100):02d}": float(np.quantile(finite, level)) for level in levels}


def extract_catalog(tree: Path, output_dir: Path, volume_cmpc3: float, chunk_records: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    header = _read_header(tree)
    nstep = int(header["nstep"])
    redshift = np.asarray(header["redshift"][:nstep], dtype=np.float64)
    output_number = np.asarray(header["output_number"][:nstep], dtype=np.int64)

    statistics, events = _collect_sink_statistics(tree, header, chunk_records)
    event_index = events["capture_index"].astype(np.int64)
    last_resolved_index = event_index - 1
    events["receiver_mass_last_resolved_msun"] = _read_receiver_masses(
        tree, header, events["receiver_id"], last_resolved_index
    )
    events["receiver_mass_assigned_output_msun"] = _read_receiver_masses(
        tree, header, events["receiver_id"], event_index
    )
    events["assigned_capture_output"] = output_number[event_index]
    events["assigned_capture_redshift"] = redshift[event_index]
    events["last_resolved_output"] = output_number[last_resolved_index]
    events["last_resolved_redshift"] = redshift[last_resolved_index]
    events["last_resolved_history_index"] = last_resolved_index
    events["assigned_capture_history_index"] = event_index
    events["minor_mass_last_resolved_msun"] = events.pop("minor_mass")
    minimum_mass = np.minimum(
        events["minor_mass_last_resolved_msun"],
        events["receiver_mass_last_resolved_msun"],
    )
    maximum_mass = np.maximum(
        events["minor_mass_last_resolved_msun"],
        events["receiver_mass_last_resolved_msun"],
    )
    events["mass_ratio_last_resolved"] = minimum_mass / maximum_mass
    events["chirp_mass_last_resolved_msun"] = (
        events["minor_mass_last_resolved_msun"]
        * events["receiver_mass_last_resolved_msun"]
    ) ** (3.0 / 5.0) / (
        events["minor_mass_last_resolved_msun"]
        + events["receiver_mass_last_resolved_msun"]
    ) ** (1.0 / 5.0)
    for source, target in (
        ("x", "minor_x_last_resolved_cmpc"),
        ("y", "minor_y_last_resolved_cmpc"),
        ("z", "minor_z_last_resolved_cmpc"),
    ):
        events[target] = events.pop(source)
    for source, target in (
        ("vx", "minor_vx_last_resolved_kms"),
        ("vy", "minor_vy_last_resolved_kms"),
        ("vz", "minor_vz_last_resolved_kms"),
    ):
        events[target] = events.pop(source)

    cosmology = FlatLambdaCDM(
        H0=float(header["h0"]),
        Om0=float(header["omega_m"]),
        Tcmb0=2.7255,
    )
    cosmic_time_gyr = np.asarray(cosmology.age(redshift).value)
    events["last_resolved_cosmic_time_gyr"] = cosmic_time_gyr[last_resolved_index]
    events["assigned_capture_cosmic_time_gyr"] = cosmic_time_gyr[event_index]
    events["capture_interval_gyr"] = (
        events["assigned_capture_cosmic_time_gyr"]
        - events["last_resolved_cosmic_time_gyr"]
    )
    interval_gyr = np.full(nstep, np.nan)
    interval_gyr[1:] = cosmic_time_gyr[1:] - cosmic_time_gyr[:-1]
    capture_count = np.bincount(event_index, minlength=nstep)
    capture_rate = capture_count / (volume_cmpc3 * interval_gyr)

    snapshot_path = output_dir / "hr5_sink_history.csv"
    with snapshot_path.open("w", newline="", encoding="utf-8") as stream:
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
                np.arange(nstep),
                output_number,
                redshift,
                cosmic_time_gyr,
                interval_gyr,
                statistics["active_count"],
                statistics["birth_count"],
                capture_count,
                capture_rate,
                statistics["total_mass"],
            )
        )

    _write_catalog(output_dir / "hr5_capture_catalog.csv", events)
    valid_mass = (
        (events["minor_mass_last_resolved_msun"] > 0.0)
        & (events["receiver_mass_last_resolved_msun"] > 0.0)
    )
    summary = {
        "source": str(tree),
        "source_size_bytes": tree.stat().st_size,
        "header_bytes": HEADER_DTYPE.itemsize,
        "sink_record_bytes": SINK_DTYPE.itemsize,
        "n_history_steps": nstep,
        "n_sink_histories": int(header["nsink"]),
        "n_capture_events": int(event_index.size),
        "n_valid_binary_masses": int(np.count_nonzero(valid_mass)),
        "n_outputs_with_captures": int(np.count_nonzero(capture_count)),
        "redshift_range": [float(redshift[-1]), float(redshift[0])],
        "assigned_capture_redshift_range": [
            float(np.min(events["assigned_capture_redshift"])),
            float(np.max(events["assigned_capture_redshift"])),
        ],
        "volume_cmpc3": volume_cmpc3,
        "cosmology": {
            "H0_km_s_Mpc": float(header["h0"]),
            "Omega_m": float(header["omega_m"]),
            "Omega_lambda": float(header["omega_lambda"]),
        },
        "event_time_convention": {
            "progenitor_state": "last resolved output i-1",
            "receiver_selection": "first output without the minor sink i",
            "assigned_capture_time": "cosmic time of output i, the interval upper bound",
            "binary_masses": "minor and receiver masses at the last resolved output i-1",
            "receiver_mass_at_i": "stored separately and not used for the binary chirp mass",
        },
        "minor_mass_last_resolved_msun": _quantiles(
            events["minor_mass_last_resolved_msun"][valid_mass]
        ),
        "receiver_mass_last_resolved_msun": _quantiles(
            events["receiver_mass_last_resolved_msun"][valid_mass]
        ),
        "receiver_mass_assigned_output_msun": _quantiles(
            events["receiver_mass_assigned_output_msun"][valid_mass]
        ),
        "mass_ratio_last_resolved": _quantiles(
            events["mass_ratio_last_resolved"][valid_mass]
        ),
        "chirp_mass_last_resolved_msun": _quantiles(
            events["chirp_mass_last_resolved_msun"][valid_mass]
        ),
    }
    (output_dir / "hr5_capture_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path, help="Path to Sink_Merging_Tree.dat.Updated")
    parser.add_argument("--output-dir", type=Path, default=Path("results/hr5"))
    parser.add_argument(
        "--volume-cmpc3",
        type=float,
        default=1.087e7,
        help="Trimmed HR5 high-resolution comoving volume",
    )
    parser.add_argument("--chunk-records", type=int, default=2048)
    args = parser.parse_args()
    extract_catalog(args.tree, args.output_dir, args.volume_cmpc3, args.chunk_records)


if __name__ == "__main__":
    main()
