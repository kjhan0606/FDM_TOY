from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from fdm_smbh_delay.dual_soliton_relaxation import (
    assess_dual_soliton_relaxation,
    read_verified_dual_soliton_relaxation_diagnostic_provenance,
)
from fdm_smbh_delay.relaxation_extractor_attestation import (
    ExtractorExecutionError,
    materialize_dual_soliton_relaxation_executed_diagnostic_provenance,
    read_verified_dual_soliton_relaxation_extractor_attestation,
    run_dual_soliton_relaxation_extractor,
)
from test_dual_soliton_relaxation import _evidence, _identity, _sample_ledger


def _diagnostics() -> dict[str, object]:
    return {
        "sample_times_code": [0.1, 0.2, 0.3, 0.4],
        "component_core_mass_code": [[4.0, 3.0]] * 4,
        "component_core_radius_box": [[0.1, 0.1]] * 4,
        "component_centres_box": [
            [[0.35, 0.5, 0.5], [0.65, 0.5, 0.5]],
            [[0.35, 0.5, 0.5], [0.65, 0.5, 0.5]],
            [[0.351, 0.5, 0.5], [0.649, 0.5, 0.5]],
            [[0.352, 0.5, 0.5], [0.648, 0.5, 0.5]],
        ],
        "relative_wave_mass_error": [0.0, 5.0e-4, 6.0e-4, 7.0e-4],
        "relative_hamiltonian_error": [0.0, 4.0e-4, 7.0e-4, 8.0e-4],
        "relative_angular_momentum_error": [0.0, 3.0e-4, 4.0e-4, 5.0e-4],
    }


def _extractor_script(path: Path, *, wrong_ledger: bool = False, exit_code: int = 0) -> None:
    diagnostics = json.dumps(_diagnostics(), sort_keys=True)
    wrong = "wrong_ledger = True" if wrong_ledger else "wrong_ledger = False"
    path.write_text(
        """
import hashlib
import json
from pathlib import Path
import sys

ledger = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
wrong_ledger = WRONG_LEDGER
ledger_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "status": "dual_soliton_relaxation_extractor_result",
    "sample_ledger": {
        "path": str((ledger.parent / "other-ledger.json") if wrong_ledger else ledger),
        "sha256": ledger_hash,
    },
    "extractor_version": "test-extractor-1",
    "diagnostics": DIAGNOSTICS,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
raise SystemExit(EXIT_CODE)
""".replace("WRONG_LEDGER", repr(wrong_ledger))
        .replace("DIAGNOSTICS", diagnostics)
        .replace("EXIT_CODE", str(exit_code)),
        encoding="utf-8",
    )


def _run_attested(tmp_path: Path):
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    ledger = _sample_ledger(identity)
    extractor = tmp_path / "extractor.py"
    _extractor_script(extractor)
    result = tmp_path / "extractor-result.json"
    attestation = tmp_path / "extractor-attestation.json"
    record = run_dual_soliton_relaxation_extractor(
        sample_ledger_path=ledger,
        result_path=result,
        attestation_path=attestation,
        extractor_command=(sys.executable, str(extractor)),
        working_directory=tmp_path,
    )
    return identity, ledger, result, attestation, record


def test_extractor_execution_is_bound_to_result_and_sample_ledger(tmp_path: Path) -> None:
    identity, ledger, result, attestation, record = _run_attested(tmp_path)
    verified = read_verified_dual_soliton_relaxation_extractor_attestation(attestation)
    assert record == verified.as_dict()
    assert verified.sample_ledger_path == ledger.resolve()
    assert verified.result_path == result.resolve()
    assert verified.command[-1] == str(tmp_path / "extractor.py")
    assert verified.environment["OMP_NUM_THREADS"] == "1"

    executed = tmp_path / "executed-diagnostic-provenance.json"
    materialize_dual_soliton_relaxation_executed_diagnostic_provenance(
        sample_ledger_path=ledger,
        extractor_attestation_path=attestation,
        output_path=executed,
    )
    parsed = read_verified_dual_soliton_relaxation_diagnostic_provenance(executed)
    assert parsed.extractor_attestation_path == attestation.resolve()
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(_evidence(identity, ledger, executed)), encoding="utf-8"
    )
    assessment = assess_dual_soliton_relaxation(evidence)
    assert assessment.conditionally_within_thresholds
    assert assessment.status == "relaxation_conservation_executed_series_within_thresholds"


def test_extractor_failure_does_not_publish_attestation(tmp_path: Path) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    ledger = _sample_ledger(identity)
    extractor = tmp_path / "failing-extractor.py"
    _extractor_script(extractor, exit_code=17)
    result = tmp_path / "result.json"
    attestation = tmp_path / "attestation.json"
    with pytest.raises(ExtractorExecutionError, match="return code 17"):
        run_dual_soliton_relaxation_extractor(
            sample_ledger_path=ledger,
            result_path=result,
            attestation_path=attestation,
            extractor_command=(sys.executable, str(extractor)),
            working_directory=tmp_path,
        )
    assert not result.exists()
    assert not attestation.exists()


def test_extractor_result_must_name_exact_ledger(tmp_path: Path) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    ledger = _sample_ledger(identity)
    extractor = tmp_path / "wrong-extractor.py"
    _extractor_script(extractor, wrong_ledger=True)
    result = tmp_path / "result.json"
    attestation = tmp_path / "attestation.json"
    with pytest.raises(ExtractorExecutionError, match="not consumable"):
        run_dual_soliton_relaxation_extractor(
            sample_ledger_path=ledger,
            result_path=result,
            attestation_path=attestation,
            extractor_command=(sys.executable, str(extractor)),
            working_directory=tmp_path,
        )
    assert not attestation.exists()


def test_attestation_rejects_changed_result_or_extractor(tmp_path: Path) -> None:
    _, _, result, attestation, _ = _run_attested(tmp_path)
    result.write_text(result.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extractor result SHA-256"):
        read_verified_dual_soliton_relaxation_extractor_attestation(attestation)

    # Recreate a clean attestation, then mutate a command source.  The result
    # mutation above is intentionally isolated to the first read check.
    second = tmp_path / "second"
    second.mkdir()
    _, _, result, attestation, _ = _run_attested(second)
    command_file = second / "extractor.py"
    command_file.write_text(command_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extractor command file"):
        read_verified_dual_soliton_relaxation_extractor_attestation(attestation)


def test_cli_accepts_command_tokens_and_reports_status(tmp_path: Path) -> None:
    identity = tmp_path / "runtime-identity.json"
    _identity(identity)
    ledger = _sample_ledger(identity)
    extractor = tmp_path / "extractor.py"
    _extractor_script(extractor)
    result = tmp_path / "cli-result.json"
    attestation = tmp_path / "cli-attestation.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_dual_soliton_relaxation_extractor.py",
            "--sample-ledger",
            str(ledger),
            "--result",
            str(result),
            "--attestation",
            str(attestation),
            "--working-directory",
            str(tmp_path),
            "--extractor",
            sys.executable,
            str(extractor),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["status"] == "dual_soliton_relaxation_extractor_executed"
    assert read_verified_dual_soliton_relaxation_extractor_attestation(attestation)
