from __future__ import annotations

import json
from pathlib import Path
import signal
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from guard_cuda_process import guard_solver  # noqa: E402


class _Monitor:
    def __init__(self, pids: set[int]) -> None:
        self._pids = pids

    def pids(self) -> set[int]:
        return set(self._pids)


def test_guard_exits_cleanly_after_the_solver_finishes(tmp_path: Path) -> None:
    alive = iter([True, False])
    status = tmp_path / "status.json"
    result = guard_solver(
        _Monitor({41}),
        gpu_index=0,
        solver_pid=41,
        poll_seconds=15.0,
        interrupt_grace_seconds=30.0,
        status_file=status,
        process_exists=lambda _pid: next(alive),
        sleep=lambda _seconds: None,
    )
    assert result == 0
    assert json.loads(status.read_text())["status"] == "solver_exited"


def test_guard_interrupts_owned_solver_when_another_pid_appears(
    tmp_path: Path,
) -> None:
    sent_signals: list[tuple[int, int]] = []
    status = tmp_path / "status.json"
    result = guard_solver(
        _Monitor({41, 99}),
        gpu_index=0,
        solver_pid=41,
        poll_seconds=15.0,
        interrupt_grace_seconds=0.0,
        status_file=status,
        process_exists=lambda _pid: True,
        send_signal=lambda pid, signum: sent_signals.append((pid, signum)),
        sleep=lambda _seconds: None,
        monotonic=lambda: 0.0,
    )
    assert result == 3
    assert sent_signals == [(41, signal.SIGINT), (41, signal.SIGTERM)]
    payload = json.loads(status.read_text())
    assert payload["status"] == "foreign_compute_detected"
    assert payload["unexpected_pids"] == [99]
