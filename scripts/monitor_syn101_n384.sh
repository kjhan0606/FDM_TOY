#!/bin/bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  printf 'usage: %s [GPU_INDEX]\n' "$0" >&2
  exit 2
fi
if [[ $(hostname -s) != syn101 ]]; then
  printf 'This manual GPU monitor is restricted to syn101.\n' >&2
  exit 2
fi

gpu_index=${1:-0}
repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
result_root=/gpfs/kjhan/FDM_TOY_RESULTS
run=${result_root}/torch_convergence/koo_n384_cuda_1myr_s2048
progress_log=${result_root}/logs/fdm_n384_syn101.out
error_log=${result_root}/logs/fdm_n384_syn101.err
monitor_log=${result_root}/logs/fdm_n384_syn101_monitor.log
pid_marker=${result_root}/logs/fdm_n384_solver.pid
lock=${result_root}/logs/fdm_n384_tripwire.lock
yield_marker=${result_root}/logs/fdm_n384_yielded
run_session=fdm_n384_cuda
finalizer_session=fdm_n384_finalize
poll_seconds=5
owner=$(id -un)

mkdir -p "$(dirname "${monitor_log}")"
exec 9>"${lock}"
if ! flock -n 9; then
  printf '%s | n384 tripwire lock is held by another monitor\n' \
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
      printf 'foreign Slurm job %s owned by %s is %s' \
        "${job_id}" "${job_user}" "${state}"
      return 0
    fi
  done < <(
    squeue -h -w syn101 -t RUNNING,COMPLETING,CONFIGURING \
      -o '%u %i %T' 2>/dev/null || true
  )
  return 1
}

gpu_uuid=$(nvidia-smi -i "${gpu_index}" --query-gpu=uuid \
  --format=csv,noheader,nounits)

foreign_gpu_use() {
  local app_uuid pid managed_pid=''
  if [[ -f ${pid_marker} ]]; then
    read -r managed_pid < "${pid_marker}" || true
  fi
  while IFS=',' read -r app_uuid pid; do
    app_uuid=${app_uuid//[[:space:]]/}
    pid=${pid//[[:space:]]/}
    if (
      [[ ${app_uuid} == "${gpu_uuid}" ]] \
      && [[ -n ${pid} ]] \
      && [[ ${pid} != "${managed_pid}" ]]
    ); then
      printf 'unmanaged GPU process PID %s' "${pid}"
      return 0
    fi
  done < <(
    nvidia-smi --query-compute-apps=gpu_uuid,pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  )
  return 1
}

stop_session() {
  local session=$1 reason=$2 attempt
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    return
  fi
  printf '%s | %s\n' "$(date '+%F %T %Z')" "${reason}" > "${yield_marker}"
  record "yielding ${session}: ${reason}"
  tmux send-keys -t "${session}" C-c
  for attempt in $(seq 1 5); do
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      record "${session} stopped"
      return
    fi
    sleep 1
  done
  tmux kill-session -t "${session}" 2>/dev/null || true
  record "forcibly closed ${session} after 5 seconds"
}

stop_all() {
  local reason=$1
  stop_session "${run_session}" "${reason}"
  stop_session "${finalizer_session}" "${reason}"
}

trap 'stop_all "n384 monitor received a termination signal"; exit 130' \
  INT TERM HUP

if reason=$(foreign_slurm_use); then
  record "refusing n384 launch: ${reason}"
  exit 75
fi
if reason=$(foreign_gpu_use); then
  record "refusing n384 launch on GPU ${gpu_index}: ${reason}"
  exit 75
fi

record "starting n384 evolution on syn101 GPU ${gpu_index}"
tmux new-session -d -s "${run_session}" \
  "exec bash ${repository}/scripts/run_syn101_n384_1myr.sh ${gpu_index} >> ${progress_log} 2>> ${error_log}"

status_counter=0
while true; do
  if reason=$(foreign_slurm_use); then
    stop_all "${reason}"
    record 'n384 monitor exiting to release syn101'
    exit 75
  fi
  if reason=$(foreign_gpu_use); then
    stop_all "GPU ${gpu_index} collision: ${reason}"
    record 'n384 monitor exiting after GPU collision'
    exit 75
  fi

  if [[ -f ${run}/torch_run_summary.json ]]; then
    if (
      ! tmux has-session -t "${run_session}" 2>/dev/null \
      && [[ ! -f ${run}/Finalization/convergence.complete ]] \
      && [[ ! -f ${run}/Finalization/convergence.failed ]] \
      && ! tmux has-session -t "${finalizer_session}" 2>/dev/null
    ); then
      record 'n384 evolution complete; starting lightweight finalizer'
      tmux new-session -d -s "${finalizer_session}" \
        "exec bash ${repository}/scripts/finalize_syn101_n384.sh"
    fi
    if (
      [[ -f ${run}/Finalization/convergence.complete ]] \
      && ! tmux has-session -t "${finalizer_session}" 2>/dev/null
    ); then
      record 'n384 evolution and convergence finalization complete'
      exit 0
    fi
    if (
      [[ -f ${run}/Finalization/convergence.failed ]] \
      && ! tmux has-session -t "${finalizer_session}" 2>/dev/null
    ); then
      record 'n384 finalizer failure marker found; automatic retry disabled'
      exit 76
    fi
  elif ! tmux has-session -t "${run_session}" 2>/dev/null; then
    record 'n384 evolution stopped without a completion summary; no retry'
    exit 76
  fi

  if (( status_counter == 0 )); then
    progress=$(tail -n 1 "${progress_log}" 2>/dev/null || true)
    gpu=$(nvidia-smi -i "${gpu_index}" \
      --query-gpu=memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null || true)
    record "GPU ${gpu} | progress ${progress}"
  fi
  status_counter=$(( (status_counter + 1) % 60 ))
  sleep "${poll_seconds}"
done
