#!/bin/bash
set -euo pipefail

repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
result_root=/gpfs/kjhan/FDM_TOY_RESULTS
run=${result_root}/torch_long_term/koo_n512_cuda_1myr_s2048
progress_log=${result_root}/logs/fdm_n512_syn101.out
error_log=${result_root}/logs/fdm_n512_syn101.err
monitor_log=${result_root}/logs/fdm_n512_syn101_monitor.log
lock=${result_root}/logs/fdm_n512_tripwire.lock
run_session=fdm_n512_cuda
finalizer_session=fdm_n512_finalize
managed_sessions=(
  "${run_session}"
  "${finalizer_session}"
  fdm_anchor_koo06
  fdm_anchor_koo15
  fdm_anchor_boey02
  fdm_anchor_boey05
  fdm_anchor_boey10
  fdm_conv_n512_rk18
  fdm_conv_n256_dt05
)
poll_seconds=5
owner=$(id -un)
node=$(hostname -s)

if [[ ${node} != syn101 ]]; then
  printf 'This manual GPU monitor is restricted to syn101.\n' >&2
  exit 2
fi

mkdir -p "$(dirname "${monitor_log}")"
exec 9>"${lock}"
if ! flock -n 9; then
  printf '%s | n512 tripwire lock is held by another monitor\n' \
    "$(date '+%F %T %Z')" >> "${monitor_log}"
  exit 3
fi

record() {
  printf '%s | %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${monitor_log}"
}

foreign_slurm_use() {
  local job_user job_id state
  while read -r job_user job_id state; do
    if [[ -n ${job_user} && ${job_user} != "${owner}" ]]; then
      printf 'foreign Slurm job %s owned by %s is %s on %s' \
        "${job_id}" "${job_user}" "${state}" "${node}"
      return 0
    fi
  done < <(
    squeue -h -w "${node}" -t RUNNING,COMPLETING,CONFIGURING \
      -o '%u %i %T' 2>/dev/null || true
  )
  return 1
}

foreign_gpu_use() {
  local process_uuid pid process_owner
  while IFS=',' read -r process_uuid pid; do
    process_uuid=${process_uuid//[[:space:]]/}
    pid=${pid//[[:space:]]/}
    if [[ -z ${pid} ]]; then
      continue
    fi
    process_owner=$(ps -o user= -p "${pid}" 2>/dev/null || true)
    process_owner=${process_owner//[[:space:]]/}
    if [[ -n ${process_owner} && ${process_owner} != "${owner}" ]]; then
      printf 'foreign GPU process %s on %s is owned by %s' \
        "${pid}" "${process_uuid}" "${process_owner}"
      return 0
    fi
  done < <(
    nvidia-smi --query-compute-apps=gpu_uuid,pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  )
  return 1
}

conflict_reason() {
  foreign_slurm_use && return 0
  foreign_gpu_use && return 0
  return 1
}

managed_session_active() {
  local session
  for session in "${managed_sessions[@]}"; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

stop_managed_runs() {
  local reason=$1 attempt session
  local -a active=()
  record "stopping all managed manual work: ${reason}"
  for session in "${managed_sessions[@]}"; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      tmux send-keys -t "${session}" C-c
    fi
  done
  for attempt in $(seq 1 30); do
    if ! managed_session_active; then
      record "all managed manual work released syn101"
      return
    fi
    sleep 1
  done
  for session in "${managed_sessions[@]}"; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      active+=("${session}")
      tmux kill-session -t "${session}" 2>/dev/null || true
    fi
  done
  record "forcibly closed sessions after 30 seconds: ${active[*]}"
}

start_run() {
  record "starting n512 checkpoint continuation manually on syn101 GPU 0"
  tmux new-session -d -s "${run_session}" \
    "bash ${repository}/scripts/run_syn101_n512_1myr.sh >> ${progress_log} 2>> ${error_log}"
}

record_status() {
  local progress checkpoint gpu
  progress=$(tail -n 1 "${progress_log}" 2>/dev/null || true)
  checkpoint=$(tr -d '\n' < "${run}/Checkpoints/latest.json" 2>/dev/null || true)
  gpu=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu,power.draw \
    --format=csv,noheader 2>/dev/null || true)
  record "GPU ${gpu} | progress ${progress} | checkpoint ${checkpoint}"
}

start_finalizer() {
  record "evolution complete; starting sequential finalizer"
  tmux new-session -d -s "${finalizer_session}" \
    "bash ${repository}/scripts/finalize_lageunha_n512.sh >> ${monitor_log} 2>&1"
}

trap 'stop_managed_runs "manual monitor received a termination signal"; exit 130' INT TERM HUP

record "manual n512 monitor active; foreign-use polling interval=${poll_seconds}s"
status_counter=0
while true; do
  reason=''
  if reason=$(conflict_reason); then
    record "foreign use detected: ${reason}"
    stop_managed_runs "${reason}"
    record "manual monitor exiting to release syn101"
    exit 75
  fi

  if [[ -f "${run}/torch_run_summary.json" ]]; then
    if (
      ! tmux has-session -t "${run_session}" 2>/dev/null \
      && [[ ! -f "${run}/Finalization/finalization.complete" ]] \
      && ! tmux has-session -t "${finalizer_session}" 2>/dev/null
    ); then
      start_finalizer
    fi
    if (
      [[ -f "${run}/Finalization/finalization.complete" ]] \
      && ! managed_session_active
    ); then
      record "n512 finalization and all managed manual work complete"
      exit 0
    fi
  elif ! tmux has-session -t "${run_session}" 2>/dev/null; then
    start_run
  fi

  if (( status_counter == 0 )); then
    record_status
  fi
  status_counter=$(( (status_counter + 1) % 360 ))
  sleep "${poll_seconds}"
done
