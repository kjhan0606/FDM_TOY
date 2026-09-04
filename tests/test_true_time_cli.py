from __future__ import annotations

import json

from fdm_smbh_delay.true_time_cli import main


def _complete_fdm_summary() -> dict[str, object]:
    return {
        "status": "reached_0p01pc",
        "t_fdm_myr": 30.0,
        "integration_time_myr": 30.0,
        "D_initial_pc": 1.0,
        "D_stop_pc": 0.01,
        "validity_flags": [],
        "source_case_id": "fdm-case-1",
        "source_sha256": "a" * 64,
    }


def _delay_record(path, name: str, delay: float) -> None:
    path.write_text(
        json.dumps(
            {
                "name": name,
                "status": "complete",
                "delay_myr": delay,
                "elapsed_lower_bound_myr": 0.0,
                "reason": None,
                "source_case_id": "case-1",
                "source_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )


def test_cli_refuses_to_complete_missing_intervals(tmp_path, capsys) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(_complete_fdm_summary()), encoding="utf-8")
    assert main(["--sink-time", "1 Gyr", "--fdm-summary", str(summary)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "incomplete"
    assert result["true_merge_time_myr"] is None
    assert set(result["missing_segments"]) == {"kpc_to_pc", "gravitational_wave"}


def test_cli_composes_completed_intervals(tmp_path, capsys) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(_complete_fdm_summary()), encoding="utf-8")
    kpc_record = tmp_path / "kpc.json"
    gw_record = tmp_path / "gw.json"
    _delay_record(kpc_record, "kpc_to_pc", 20.0)
    _delay_record(gw_record, "gravitational_wave", 4.0)
    assert main(
        [
            "--z-sink", "1.0",
            "--fdm-summary", str(summary),
            "--kpc-to-pc-record", str(kpc_record),
            "--gw-record", str(gw_record),
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
    assert result["delay_lower_bound_myr"] == 12.5
    assert set(result["censored_segments"]) == {
        "kpc_to_pc", "fdm_pc_to_0p01pc", "gravitational_wave",
    }
    assert result["missing_segments"] == []
    assert result["segments"][1]["status"] == "censored"
    assert result["segments"][1]["reason"] == (
        "outside accepted q-e-separation support"
    )


def test_bare_delays_are_explicitly_censored(tmp_path, capsys) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(_complete_fdm_summary()), encoding="utf-8")
    assert main(
        [
            "--sink-time", "1 Gyr",
            "--fdm-summary", str(summary),
            "--kpc-to-pc-delay", "20 Myr",
            "--gw-delay", "4 Myr",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "censored"
    assert {item["name"] for item in result["segments"] if item["status"] == "censored"} == {
        "kpc_to_pc", "gravitational_wave",
    }


def test_fdm_summary_flags_and_wrong_interval_are_censored(tmp_path, capsys) -> None:
    summary = tmp_path / "summary.json"
    record = _complete_fdm_summary()
    record["validity_flags"] = ["STATIC_SOLITON_BACKREACTION"]
    summary.write_text(json.dumps(record), encoding="utf-8")
    assert main(["--sink-time", "1 Gyr", "--fdm-summary", str(summary)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "censored"
    assert result["segments"][1]["status"] == "censored"
    assert "validity flags" in result["segments"][1]["reason"]

    record["validity_flags"] = []
    record["D_stop_pc"] = 0.5
    summary.write_text(json.dumps(record), encoding="utf-8")
    assert main(["--sink-time", "1 Gyr", "--fdm-summary", str(summary)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "censored"
    assert "required 1 pc to 0.01 pc" in result["segments"][1]["reason"]


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
