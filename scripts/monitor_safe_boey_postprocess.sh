#!/bin/bash
set -euo pipefail

if [[ $(hostname -s) != syntax ]]; then
  printf 'This file-based CPU monitor is restricted to syntax.\n' >&2
  exit 2
fi

result_root=/gpfs/kjhan/FDM_TOY_RESULTS
progress_log=${result_root}/logs/fdm_boey_cpu_safe.out
error_log=${result_root}/logs/fdm_boey_cpu_safe.err
monitor_log=${result_root}/logs/fdm_boey_cpu_safe_monitor.log
alert_marker=${result_root}/logs/fdm_boey_cpu_safe_monitor.alert
lock=${result_root}/logs/fdm_boey_cpu_safe_monitor.lock
calibration_root=${result_root}/torch_calibration/tier0_n512
poll_seconds=60
stale_seconds=1200

mkdir -p "$(dirname "${monitor_log}")"
exec 9>"${lock}"
if ! flock -n 9; then
  printf '%s | CPU monitor lock is held by another process\n' \
    "$(date '+%F %T %Z')" >> "${monitor_log}"
  exit 3
fi

record() {
  printf '%s | %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${monitor_log}"
}

outputs_complete() {
  local case_id output
  for case_id in \
    boey_each02pct_n512 \
    boey_each05pct_n512 \
    boey_each10pct_n512
  do
    for output in \
      orbit_averaged_exchange_summary.json \
      line_density_summary.json \
      wave_response_summary.json \
      wave_exchange.summary.json
    do
      [[ -s ${calibration_root}/${case_id}/${output} ]] || return 1
    done
  done
  return 0
}

progress_size=$(stat -c '%s' "${progress_log}" 2>/dev/null || echo 0)
last_change_epoch=$(date '+%s')
record "file-based CPU monitor active; interval=${poll_seconds}s stale=${stale_seconds}s"

while true; do
  now=$(date '+%s')
  current_size=$(stat -c '%s' "${progress_log}" 2>/dev/null || echo 0)
  error_size=$(stat -c '%s' "${error_log}" 2>/dev/null || echo 0)

  if (( error_size > 0 )); then
    record "CPU post-processing error log is non-empty (${error_size} bytes)"
    touch "${alert_marker}"
    exit 75
  fi

  if (( current_size != progress_size )); then
    progress_size=${current_size}
    last_change_epoch=${now}
    last_line=$(tail -n 1 "${progress_log}" 2>/dev/null || true)
    record "progress changed (${current_size} bytes): ${last_line}"
  fi

  if outputs_complete; then
    record "all three Boey CPU post-processing outputs verified"
    rm -f "${alert_marker}"
    exit 0
  fi

  if (( now - last_change_epoch >= stale_seconds )); then
    record "CPU post-processing log has not changed for ${stale_seconds} seconds"
    touch "${alert_marker}"
    exit 75
  fi

  sleep "${poll_seconds}"
done
