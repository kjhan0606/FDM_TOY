#!/bin/bash
set -euo pipefail

repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
result_root=/gpfs/kjhan/FDM_TOY_RESULTS
n384_root=${result_root}/torch_calibration/tier0_n384
n512_root=${result_root}/torch_calibration/tier0_n512
comparison_root=${result_root}/torch_calibration
log_root=${result_root}/logs
log=${log_root}/fdm_boey_n384_postprocess.log
dry_run=false
requested_cases=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=true
      ;;
    boey_each02pct|boey_each05pct|boey_each10pct)
      requested_cases+=("$1")
      ;;
    *)
      printf 'usage: %s [--dry-run] [boey_each02pct ...]\n' "$0" >&2
      exit 2
      ;;
  esac
  shift
done
if [[ ${#requested_cases[@]} -eq 0 ]]; then
  requested_cases=(boey_each02pct boey_each05pct boey_each10pct)
fi
if [[ ${dry_run} == false && $(hostname -s) != syntax ]]; then
  printf 'This bounded Boey n384 post-processor is restricted to syntax.\n' >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# A 384-cubed response sample is expected to use about 16 GiB. Keep one
# process below a 32 GiB address-space ceiling and let the response runner
# checkpoint after every sparse sample.
ulimit -v 33554432

if [[ ${dry_run} == false ]]; then
  mkdir -p "${log_root}"
  exec 9>"${log_root}/fdm_boey_n384_postprocess.lock"
  if ! flock -n 9; then
    printf 'Another Boey n384 post-processor already holds the lock.\n' >&2
    exit 3
  fi
fi
cd "${repository}"

record() {
  if [[ ${dry_run} == true ]]; then
    printf '%s\n' "$*"
  else
    printf '%s | %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${log}"
  fi
}

run_step() {
  local case_id=$1 name=$2
  shift 2
  if [[ ${dry_run} == true ]]; then
    printf 'STEP %s %s:' "${case_id}" "${name}"
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  record "${case_id} | ${name} starting"
  "$@" >> "${log}" 2>&1
  record "${case_id} | ${name} complete"
}

verify_evolution() {
  local run=$1 expected_case=$2
  python - "${run}" "${expected_case}" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1]).resolve()
expected_case = sys.argv[2]
summary = json.loads((run / "torch_run_summary.json").read_text())
metadata = json.loads((run / "fdm_adapter_metadata.json").read_text())
if summary.get("status") != "complete":
    raise SystemExit("evolution summary is not complete")
if int(metadata.get("resolution", -1)) != 384:
    raise SystemExit("post-processing input is not n384")
if metadata.get("case_id") != expected_case:
    raise SystemExit("post-processing case identity does not match")
PY
}

verify_outputs() {
  local run=$1 reference=$2 comparison=$3
  python - "${run}" "${reference}" "${comparison}" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1]).resolve()
reference = Path(sys.argv[2]).resolve()
comparison_path = Path(sys.argv[3]).resolve()
statuses = {
    "conservation_summary.json": "diagnosed",
    "orbit_averaged_exchange_summary.json": "orbit_averaged",
    "line_density_summary.json": "line_density_diagnosed",
    "wave_response_summary.json": "diagnosed",
    "wave_exchange.summary.json": "dimensionless_exchange_table",
}
for name, expected in statuses.items():
    data = json.loads((run / name).read_text())
    if data.get("status") != expected:
        raise SystemExit(f"{name} has invalid status")
for name in (
    "torch_solver_provenance/manifest.json",
    "conservation_timeseries.csv",
    "orbit_averaged_exchange.csv",
    "line_density_diagnostics.csv",
    "wave_response_timeseries.csv",
    "wave_radial_profiles.csv",
    "wave_exchange.csv",
):
    if not (run / name).is_file() or (run / name).stat().st_size == 0:
        raise SystemExit(f"required output is missing or empty: {name}")
comparison = json.loads(comparison_path.read_text())
if comparison.get("status") != "common_resolved_interval_compared":
    raise SystemExit("matched convergence summary is incomplete")
if comparison.get("reference_label") != "n512":
    raise SystemExit("matched convergence reference is not n512")
runs = {row["label"]: Path(row["run"]).resolve() for row in comparison["runs"]}
if runs != {"n512": reference, "n384": run}:
    raise SystemExit("matched convergence run identity does not match")
matched = comparison.get("matched_separation")
if not isinstance(matched, dict) or not matched.get("bins"):
    raise SystemExit("matched convergence contains no accepted separation bins")
PY
}

for case_id in "${requested_cases[@]}"; do
  run=${n384_root}/${case_id}_n384
  reference=${n512_root}/${case_id}_n512
  comparison=${comparison_root}/${case_id}_spatial_convergence_n384_n512.json
  if [[ ${dry_run} == false ]]; then
    verify_evolution "${run}" "${case_id}"
  fi
  run_step "${case_id}" provenance \
    python scripts/snapshot_torch_provenance.py "${run}"
  run_step "${case_id}" conservation \
    python scripts/analyze_pyul_wave_run.py "${run}"
  run_step "${case_id}" orbit_averaged_exchange \
    python scripts/analyze_pyul_secular_exchange.py "${run}"
  run_step "${case_id}" line_density \
    python scripts/analyze_pyul_line_density.py "${run}"
  run_step "${case_id}" resumable_wave_response \
    bash scripts/run_safe_n384_wave_response.sh "${run}"
  run_step "${case_id}" wave_exchange_table \
    python scripts/build_wave_exchange_table.py "${run}" \
      --output "${run}/wave_exchange.csv"
  run_step "${case_id}" matched_n512_n384 \
    python scripts/summarize_pyul_convergence.py \
      n512="${reference}" n384="${run}" --output "${comparison}"
  if [[ ${dry_run} == false ]]; then
    verify_outputs "${run}" "${reference}" "${comparison}"
  fi
  record "${case_id} | verified post-processing complete"
done
