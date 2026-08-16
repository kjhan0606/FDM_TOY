from __future__ import annotations

import json

from fdm_smbh_delay.true_time_cli import main


def test_cli_refuses_to_complete_missing_intervals(tmp_path, capsys) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"status": "reached_0p01pc", "t_fdm_myr": 30.0}),
        encoding="utf-8",
    )
    assert main(["--sink-time", "1 Gyr", "--fdm-summary", str(summary)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "incomplete"
    assert result["true_merge_time_myr"] is None
    assert set(result["missing_segments"]) == {"kpc_to_pc", "gravitational_wave"}


def test_cli_composes_completed_intervals(tmp_path, capsys) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"status": "reached_0p01pc", "t_fdm_myr": 30.0}),
        encoding="utf-8",
    )
    assert main(
        [
            "--z-sink", "1.0",
            "--fdm-summary", str(summary),
            "--kpc-to-pc-delay", "20 Myr",
            "--gw-delay", "4 Myr",
            "--z-sink", "1.0",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    assert result["true_merge_time_myr"] == result["sink_time_myr"] + 54.0
    assert 0.0 < result["z_true"] < 1.0


def test_cli_composes_explicit_uncalibrated_summary_as_censored(
    tmp_path, capsys
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "uncalibrated",
                "integration_time_myr": 12.5,
                "reason": "outside accepted q-e-separation support",
            }
        ),
        encoding="utf-8",
    )
    assert main(
        [
            "--sink-time",
            "1 Gyr",
            "--fdm-summary",
            str(summary),
            "--kpc-to-pc-delay",
            "20 Myr",
            "--gw-delay",
            "4 Myr",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "censored"
    assert result["delay_lower_bound_myr"] == 36.5
    assert result["censored_segments"] == ["fdm_pc_to_0p01pc"]
    assert result["missing_segments"] == []
    assert result["segments"][1]["status"] == "censored"
    assert result["segments"][1]["reason"] == (
        "outside accepted q-e-separation support"
    )


def test_cli_accepts_hyphenated_outside_support_status(tmp_path, capsys) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"status": "outside-support", "integration_time_myr": 3.0}),
        encoding="utf-8",
    )
    assert main(
        ["--sink-time", "1 Gyr", "--fdm-summary", str(summary)]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "censored"
    assert result["delay_lower_bound_myr"] == 3.0
