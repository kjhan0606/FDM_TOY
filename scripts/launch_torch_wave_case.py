#!/usr/bin/env python3
"""Launch a Torch calculation only after preserving its numerical source."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from snapshot_torch_provenance import _snapshot_run


def _output_argument(arguments: list[str]) -> Path:
    for index, argument in enumerate(arguments):
        if argument == "--output" and index + 1 < len(arguments):
            return Path(arguments[index + 1]).expanduser().resolve()
        if argument.startswith("--output="):
            return Path(argument.split("=", 1)[1]).expanduser().resolve()
    raise ValueError("the Torch launcher requires --output RUN_DIRECTORY")


def _exec_solver(project: Path, arguments: list[str]) -> None:
    runner = project / "scripts" / "run_torch_wave_case.py"
    os.execv(sys.executable, [sys.executable, str(runner), *arguments])


def main() -> int:
    arguments = sys.argv[1:]
    project = Path(__file__).resolve().parents[1]
    if any(argument in {"-h", "--help"} for argument in arguments):
        _exec_solver(project, arguments)
    output = _output_argument(arguments)
    metadata = output / "fdm_adapter_metadata.json"

    if metadata.is_file():
        _snapshot_run(project, output)
        _exec_solver(project, arguments)

    runner = project / "scripts" / "run_torch_wave_case.py"
    process = subprocess.Popen([sys.executable, str(runner), *arguments])

    def forward_signal(signum: int, _frame) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    try:
        while process.poll() is None and not metadata.is_file():
            time.sleep(0.1)
        if metadata.is_file():
            _snapshot_run(project, output)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
