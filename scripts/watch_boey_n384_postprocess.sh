#!/bin/bash
set -euo pipefail

if [[ $(hostname -s) != syntax ]]; then
  printf 'This low-impact Boey n384 tripwire is restricted to syntax.\n' >&2
  exit 2
fi

result_root=/gpfs/kjhan/FDM_TOY_RESULTS
n384_root=${result_root}/torch_calibration/tier0_n384
koo_n384=${result_root}/torch_convergence/koo_n384_cuda_1myr_s2048
log_root=${result_root}/logs
log=${log_root}/fdm_boey_n384_tripwire.log
interval_seconds=${FDM_TRIPWIRE_INTERVAL_SECONDS:-300}
maximum_wait_seconds=${FDM_TRIPWIRE_MAXIMUM_WAIT_SECONDS:-604800}

if [[ ! ${interval_seconds} =~ ^[1-9][0-9]*$ ]] \
  || [[ ! ${maximum_wait_seconds} =~ ^[1-9][0-9]*$ ]]; then
  printf 'Tripwire intervals must be positive integers.\n' >&2
  exit 2
fi

mkdir -p "${log_root}"
exec 9>"${log_root}/fdm_boey_n384_tripwire.lock"
if ! flock -n 9; then
  printf 'Another Boey n384 tripwire already holds the lock.\n' >&2
  exit 3
fi

record() {
  printf '%s | %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${log}"
}

inputs_ready() {
  [[ -s ${n384_root}/boey_n384_sequence_summary.json ]] \
    && [[ -s ${n384_root}/boey_each02pct_n384/torch_run_summary.json ]] \
    && [[ -s ${n384_root}/boey_each05pct_n384/torch_run_summary.json ]] \
    && [[ -s ${n384_root}/boey_each10pct_n384/torch_run_summary.json ]] \
    && [[ -s ${koo_n384}/wave_response_summary.json ]]
}

elapsed=0
record "tripwire armed; waiting for the Boey sequence and Koo n384 response"
until inputs_ready; do
  if ((elapsed >= maximum_wait_seconds)); then
    record "tripwire timed out without starting CPU post-processing"
    exit 4
  fi
  sleep "${interval_seconds}"
  ((elapsed += interval_seconds))
done

# Let the Koo response runner publish its final files and release the shared
# n384 FFT lock before starting the first Boey response sample.
sleep 30
record "inputs ready; starting bounded sequential Boey n384 post-processing"
exec /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY/scripts/finalize_boey_n384.sh

