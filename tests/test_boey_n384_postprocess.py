from __future__ import annotations

from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[1]
FINALIZER = REPOSITORY / "scripts" / "finalize_boey_n384.sh"
TRIPWIRE = REPOSITORY / "scripts" / "watch_boey_n384_postprocess.sh"


def test_boey_n384_shell_drivers_are_syntactically_valid() -> None:
    for script in (FINALIZER, TRIPWIRE):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_boey_n384_finalizer_dry_run_has_every_required_stage() -> None:
    result = subprocess.run(
        ["bash", str(FINALIZER), "--dry-run", "boey_each02pct"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = result.stdout
    for stage in (
        "provenance",
        "conservation",
        "orbit_averaged_exchange",
        "line_density",
        "resumable_wave_response",
        "wave_exchange_table",
        "matched_n512_n384",
    ):
        assert f"STEP boey_each02pct {stage}:" in output
    assert "boey_each05pct" not in output
    assert "boey_each10pct" not in output


def test_default_finalizer_builds_the_combined_release_last() -> None:
    result = subprocess.run(
        ["bash", str(FINALIZER), "--dry-run"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = result.stdout
    assert output.count("matched_n512_n384") == 3
    assert "STEP combined invalidate_previous_verification:" in output
    assert "STEP combined accepted_subgrid_table:" in output
    assert "STEP combined release_runtime_verification:" in output
    assert output.index("boey_each10pct matched_n512_n384") < output.index(
        "STEP combined invalidate_previous_verification:"
    )
    assert output.index(
        "STEP combined invalidate_previous_verification:"
    ) < output.index(
        "STEP combined accepted_subgrid_table:"
    )
    assert output.index("STEP combined accepted_subgrid_table:") < output.index(
        "STEP combined release_runtime_verification:"
    )


def test_table_only_finalizer_does_not_repeat_case_postprocessing() -> None:
    result = subprocess.run(
        ["bash", str(FINALIZER), "--dry-run", "--build-table-only"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "STEP combined invalidate_previous_verification:" in result.stdout
    assert "STEP combined accepted_subgrid_table:" in result.stdout
    assert "STEP combined release_runtime_verification:" in result.stdout
    assert "matched_n512_n384" not in result.stdout
    assert "resumable_wave_response" not in result.stdout
