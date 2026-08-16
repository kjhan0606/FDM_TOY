from __future__ import annotations

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from run_guarded_syn101_boey_n384 import (  # noqa: E402
    BoeyCase,
    GuardedSequence,
    _foreign_slurm_reason,
    _memory_used_mib,
    _pmon_pid,
)
import run_guarded_syn101_boey_n384  # noqa: E402


def test_pmon_pid_ignores_headers_and_empty_samples() -> None:
    assert _pmon_pid("# gpu pid type sm mem command") is None
    assert _pmon_pid("0 - - - - -") is None
    assert _pmon_pid("") is None
    assert _pmon_pid("0 12345 C 0 4 python") == 12345


def test_foreign_slurm_reason_ignores_owner() -> None:
    assert _foreign_slurm_reason("kjhan 11 RUNNING", "kjhan") is None
    assert _foreign_slurm_reason("", "kjhan") is None
    assert _foreign_slurm_reason("other 12 RUNNING", "kjhan") == (
        "foreign Slurm job 12 owned by other is RUNNING"
    )


def test_memory_used_mib_parses_loop_samples() -> None:
    assert _memory_used_mib("54206") == 54206
    assert _memory_used_mib("0") == 0
    assert _memory_used_mib("memory.used") is None
    assert _memory_used_mib("-1") is None


def test_boey_case_uses_matching_reference_and_output_names() -> None:
    case = BoeyCase("boey_each05pct")
    assert case.reference.name == "boey_each05pct_n384"
    assert case.output.name == "boey_each05pct_n384"
    assert case.duration_myr == 0.8
    assert case.save_number == 2048
    assert case.save_3d_number == 16
    assert case.checkpoint_every_saves == 32


def test_boey_case_supports_a_non_destructive_dt05_recovery_name() -> None:
    case = BoeyCase(
        "boey_each10pct",
        output_run_name="boey_each10pct_n384_dt05",
        time_step_factor=0.5,
    )
    assert case.reference.name == "boey_each10pct_n384"
    assert case.output.name == "boey_each10pct_n384_dt05"
    assert case.pid_marker.name == "fdm_boey_each10pct_n384_dt05_solver.pid"


def test_guarded_sequence_resumes_an_existing_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    reference_root = tmp_path / "reference"
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    monkeypatch.setattr(
        run_guarded_syn101_boey_n384, "REFERENCE_ROOT", reference_root
    )
    monkeypatch.setattr(
        run_guarded_syn101_boey_n384, "OUTPUT_ROOT", output_root
    )
    monkeypatch.setattr(run_guarded_syn101_boey_n384, "LOG_ROOT", log_root)
    case = BoeyCase("boey_each02pct")
    case.output.mkdir(parents=True)
    (case.output / "Checkpoints").mkdir()
    (case.output / "Checkpoints/latest.json").write_text("{}")
    sequence = GuardedSequence(gpu_index=0, poll_seconds=10)
    captured = {}

    def run_command(arguments, *, description, marker=None):
        captured["arguments"] = arguments
        captured["description"] = description
        captured["marker"] = marker
        return 0

    monkeypatch.setattr(sequence, "run_command", run_command)
    assert sequence.evolve(case) == 0
    assert "--resume" in captured["arguments"]
    assert captured["marker"] == case.pid_marker
