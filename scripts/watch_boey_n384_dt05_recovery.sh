#!/bin/bash
set -euo pipefail

if [[ $(hostname -s) != syntax ]]; then
  printf 'This recovery finalizer tripwire is restricted to syntax.\n' >&2
  exit 2
fi

repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
result_root=/gpfs/kjhan/FDM_TOY_RESULTS
calibration_root=${result_root}/torch_calibration
n384_root=${calibration_root}/tier0_n384
n512_root=${calibration_root}/tier0_n512
convergence_root=${result_root}/torch_convergence
recovery_run=${n384_root}/boey_each10pct_n384_dt05
recovery_summary=${n384_root}/boey_n384_dt05_recovery_summary.json
release_verification=${calibration_root}/fdm_subgrid_calibration.verification.json
log_root=${result_root}/logs
log=${log_root}/fdm_boey_n384_dt05_recovery_tripwire.log
state=${calibration_root}/boey_n384_dt05_recovery_finalization.json
failure=${calibration_root}/boey_n384_dt05_recovery_finalization.failed
interval_seconds=${FDM_RECOVERY_INTERVAL_SECONDS:-300}
maximum_wait_seconds=${FDM_RECOVERY_MAXIMUM_WAIT_SECONDS:-2592000}

if [[ ! ${interval_seconds} =~ ^[1-9][0-9]*$ ]] \
  || [[ ! ${maximum_wait_seconds} =~ ^[1-9][0-9]*$ ]]; then
  printf 'Recovery wait intervals must be positive integers.\n' >&2
  exit 2
fi

mkdir -p "${log_root}"
exec 9>"${log_root}/fdm_boey_n384_dt05_recovery_tripwire.lock"
if ! flock -n 9; then
  printf 'Another Boey n384 dt05 recovery tripwire holds the lock.\n' >&2
  exit 3
fi
cd "${repository}"

record() {
  printf '%s | %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${log}"
}

record_failure() {
  local status=$?
  trap - ERR
  touch "${failure}"
  record "Boey n384 dt05 recovery finalization failed with status ${status}"
  exit "${status}"
}
trap record_failure ERR

elapsed=0
record "recovery tripwire armed; waiting only on the completion summary file"
until [[ -s ${recovery_summary} ]]; do
  if ((elapsed >= maximum_wait_seconds)); then
    record "recovery tripwire timed out"
    exit 4
  fi
  sleep "${interval_seconds}"
  ((elapsed += interval_seconds))
done

python - "${recovery_run}" "${recovery_summary}" <<'PY'
import json
from pathlib import Path
import sys

from fdm_smbh_delay.run_metadata import validate_torch_calibration_completion

run = Path(sys.argv[1]).resolve()
sequence = json.loads(Path(sys.argv[2]).read_text())
if sequence.get("status") != "complete":
    raise SystemExit("recovery sequence summary is incomplete")
if sequence.get("cases") != [run.name] or sequence.get("time_step_factor") != 0.5:
    raise SystemExit("recovery sequence summary identifies the wrong run")
validate_torch_calibration_completion(
    run,
    expected_case_id="boey_each10pct",
    expected_resolution=384,
    expected_duration_myr=0.8,
    expected_saved_intervals=2048,
    expected_saved_3d_states=17,
    expected_rk4_substeps=9,
    expected_checkpoint_interval=32,
    expected_time_step_factor=0.5,
    expected_run_id=run.name,
)
PY

# Let the evolution process close its final descriptors before CPU analysis.
sleep 30
record "recovery evolution verified; starting bounded post-processing"
bash scripts/finalize_boey_n384.sh \
  --n384-run "${recovery_run}" \
  --expected-time-step-factor 0.5 \
  boey_each10pct >> "${log}" 2>&1

record "refreshing accepted-source convergence summaries with resolved gates"
python scripts/summarize_pyul_convergence.py \
  n512="${n512_root}/boey_each02pct_n512" \
  n384="${n384_root}/boey_each02pct_n384" \
  --output "${calibration_root}/boey_each02pct_spatial_convergence_n384_n512.json" \
  >> "${log}" 2>&1
python scripts/summarize_pyul_convergence.py \
  n512="${n512_root}/boey_each05pct_n512" \
  n384="${n384_root}/boey_each05pct_n384" \
  --output "${calibration_root}/boey_each05pct_spatial_convergence_n384_n512.json" \
  >> "${log}" 2>&1
python scripts/summarize_pyul_convergence.py \
  n512="${result_root}/torch_long_term/koo_n512_cuda_1myr_s2048" \
  n384="${convergence_root}/koo_n384_cuda_1myr_s2048" \
  --output "${convergence_root}/koo_spatial_convergence_n384_n512_1myr.json" \
  >> "${log}" 2>&1

record "building and verifying the accepted-only combined release"
bash scripts/finalize_boey_n384.sh --build-table-only >> "${log}" 2>&1
python - "${release_verification}" "${state}" <<'PY'
import json
import os
from pathlib import Path
import sys

verification = json.loads(Path(sys.argv[1]).read_text())
if verification.get("status") != "subgrid_calibration_release_verified":
    raise SystemExit("combined release verification is incomplete")
state = Path(sys.argv[2])
temporary = state.with_name(f".{state.name}.tmp")
temporary.write_text(
    json.dumps(
        {
            "status": "complete",
            "release_verification": str(Path(sys.argv[1]).resolve()),
            "release_sha256": verification["table"]["sha256"],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
os.replace(temporary, state)
PY
rm -f "${failure}"
trap - ERR
record "Boey n384 dt05 recovery and combined release verification complete"
