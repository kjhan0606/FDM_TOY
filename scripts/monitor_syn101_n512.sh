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
declare -A gpu_session=(
  [0]="${run_session}"
  [1]=fdm_anchor_koo06
  [2]=fdm_anchor_koo15
  [3]=fdm_anchor_boey02
  [4]=fdm_anchor_boey05
  [5]=fdm_anchor_boey10
  [6]=fdm_conv_n512_rk18
  [7]=fdm_conv_n256_dt05
)
yield_marker_dir=${result_root}/logs/fdm_syn101_yielded
poll_seconds=5
owner=$(id -un)
node=$(hostname -s)

if [[ ${node} != syn101 ]]; then
  printf 'This manual GPU monitor is restricted to syn101.\n' >&2
  exit 2
fi

mkdir -p "$(dirname "${monitor_log}")" "${yield_marker_dir}"
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

pid_descends_from() {
  local child=$1 ancestor=$2 parent
  while [[ ${child} =~ ^[0-9]+$ ]] && (( child > 1 )); do
    if [[ ${child} == "${ancestor}" ]]; then
      return 0
    fi
    parent=$(ps -o ppid= -p "${child}" 2>/dev/null || true)
    parent=${parent//[[:space:]]/}
    if [[ -z ${parent} || ${parent} == "${child}" ]]; then
      break
    fi
    child=${parent}
  done
  return 1
}

pid_belongs_to_session() {
  local pid=$1 session=$2 pane_pid
  while read -r pane_pid; do
    pane_pid=${pane_pid//[[:space:]]/}
    if [[ -n ${pane_pid} ]] && pid_descends_from "${pid}" "${pane_pid}"; then
      return 0
    fi
  done < <(tmux list-panes -t "${session}" -F '#{pane_pid}' 2>/dev/null || true)
  return 1
}

stop_managed_session() {
  local session=$1 gpu=$2 reason=$3 attempt
  local marker=${yield_marker_dir}/${session}
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    return
  fi
  printf '%s\n' "$(date '+%F %T %Z') | GPU ${gpu} | ${reason}" > "${marker}"
  record "yielding ${session} on GPU ${gpu}: ${reason}"
  tmux send-keys -t "${session}" C-c
  for attempt in $(seq 1 5); do
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      record "${session} released GPU ${gpu}"
      return
    fi
    sleep 1
  done
  tmux kill-session -t "${session}" 2>/dev/null || true
  record "forcibly closed ${session} after 5 seconds"
}

yield_gpu_collisions() {
  local index uuid app_uuid pid session process_owner process_command key
  declare -A uuid_index=()
  declare -A yielded=()

  while IFS=',' read -r index uuid; do
    index=${index//[[:space:]]/}
    uuid=${uuid//[[:space:]]/}
    [[ -n ${index} && -n ${uuid} ]] && uuid_index["${uuid}"]=${index}
  done < <(
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
      2>/dev/null || true
  )

  while IFS=',' read -r app_uuid pid; do
    app_uuid=${app_uuid//[[:space:]]/}
    pid=${pid//[[:space:]]/}
    index=${uuid_index[${app_uuid}]:-}
    session=${gpu_session[${index}]:-}
    if (
      [[ -z ${pid} || -z ${session} || -n ${yielded[${session}]:-} ]] \
      || ! tmux has-session -t "${session}" 2>/dev/null \
      || pid_belongs_to_session "${pid}" "${session}"
    ); then
      continue
    fi
    process_owner=$(ps -o user= -p "${pid}" 2>/dev/null || true)
    process_owner=${process_owner//[[:space:]]/}
    process_command=$(ps -o args= -p "${pid}" 2>/dev/null || true)
    key=${session}
    yielded["${key}"]=1
    stop_managed_session "${session}" "${index}" \
      "unmanaged PID ${pid} owner=${process_owner:-unknown} command=${process_command:-unknown}"
  done < <(
    nvidia-smi --query-compute-apps=gpu_uuid,pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  )
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
  if [[ -f ${yield_marker_dir}/${run_session} ]]; then
    record "not restarting n512: GPU 0 was yielded to another session"
    return
  fi
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
  if reason=$(foreign_slurm_use); then
    record "foreign Slurm use detected: ${reason}"
    stop_managed_runs "${reason}"
    record "manual monitor exiting to release syn101"
    exit 75
  fi
  yield_gpu_collisions

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
  elif (
    ! tmux has-session -t "${run_session}" 2>/dev/null \
    && [[ ! -f ${yield_marker_dir}/${run_session} ]]
  ); then
    start_run
  fi

  if (
    [[ -f ${yield_marker_dir}/${run_session} ]] \
    && ! managed_session_active
  ); then
    record "all remaining work completed or yielded; monitor exiting"
    exit 75
  fi

  if (( status_counter == 0 )); then
    record_status
  fi
  status_counter=$(( (status_counter + 1) % 360 ))
  sleep "${poll_seconds}"
done
