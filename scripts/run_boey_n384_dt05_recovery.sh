#!/bin/bash
set -uo pipefail

if [[ $(hostname -s) != syn101 ]]; then
  printf 'This guarded recovery driver is restricted to syn101.\n' >&2
  exit 2
fi

repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
log_root=/gpfs/kjhan/FDM_TOY_RESULTS/logs
driver_log=${log_root}/fdm_boey_n384_dt05_recovery_driver.log

mkdir -p "${log_root}"
exec 9>"${log_root}/fdm_boey_n384_dt05_recovery_driver.lock"
if ! flock -n 9; then
  printf 'Another Boey n384 dt05 recovery driver holds the lock.\n' >&2
  exit 3
fi
cd "${repository}"

record() {
  printf '%s | %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${driver_log}"
}

record "guarded Boey 10% n384 dt05 recovery armed"
while true; do
  if python scripts/run_guarded_syn101_boey_n384.py \
    --gpu-index 0 \
    --poll-seconds 10 \
    --wait-for-idle \
    --maximum-wait-seconds 2592000 \
    --recovery-10pct-dt05; then
    record "guarded Boey 10% n384 dt05 recovery complete"
    exit 0
  else
    status=$?
  fi
  if [[ ${status} -ne 75 ]]; then
    record "guarded Boey 10% n384 dt05 recovery failed with status ${status}"
    exit "${status}"
  fi
  record "GPU conflict yielded at a checkpoint; rearming the idle wait"
done
