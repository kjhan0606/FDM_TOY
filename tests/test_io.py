from __future__ import annotations

import csv
import json

from fdm_smbh_delay.io import write_result
from fdm_smbh_delay.orbit import integrate_case


def test_result_bundle_contains_provenance(case_factory, tmp_path) -> None:
    case = case_factory(drag=False, max_time=0.001, output_samples=4)
    result = integrate_case(case)
    output = write_result(result, case, tmp_path / "result")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["input_hash"]) == 64
    assert summary["provenance"]["internal_units"] == {
        "length": "pc",
        "mass": "Msun",
        "time": "Myr",
    }
    with (output / "timeseries.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 5
    assert "power_to_fdm_rest" in rows[0]
