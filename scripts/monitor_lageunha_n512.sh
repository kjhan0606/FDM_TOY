#!/bin/bash
set -euo pipefail

repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
run=/gpfs/kjhan/FDM_TOY_RESULTS/torch_long_term/koo_n512_cuda_1myr_s2048
log=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_n512_monitor.log
movie_directory=/gpfs/kjhan/FDM_TOY_RESULTS/movies

mkdir -p "$(dirname "${log}")" "${movie_directory}"

record_status() {
  local timestamp progress checkpoint gpu
  timestamp=$(date '+%F %T %Z')
  progress=$(tail -n 1 /gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_n512_lageunha.out 2>/dev/null || true)
  checkpoint=$(cat "${run}/Checkpoints/latest.json" 2>/dev/null || true)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw \
    --format=csv,noheader 2>/dev/null || true)
  printf '%s | GPU %s | progress %s | checkpoint %s\n' \
    "${timestamp}" "${gpu}" "${progress}" "${checkpoint}" >> "${log}"
}

while true; do
  if [[ -f "${run}/torch_run_summary.json" ]]; then
    record_status
    cd "${repository}" || exit 1
    python scripts/analyze_pyul_wave_run.py "${run}" >> "${log}" 2>&1
    python scripts/analyze_pyul_secular_exchange.py "${run}" >> "${log}" 2>&1
    python scripts/analyze_pyul_line_density.py "${run}" >> "${log}" 2>&1
    python scripts/build_wave_exchange_table.py "${run}" \
      --output "${run}/wave_exchange.csv" >> "${log}" 2>&1
    python scripts/render_pyul_movie.py "${run}" \
      --output "${movie_directory}/koo_n512_cuda_1myr.mp4" \
      --poster "${movie_directory}/koo_n512_cuda_1myr_poster.png" \
      --zoom-width-pc 1.44 >> "${log}" 2>&1
    printf '%s | analysis and movie complete\n' "$(date '+%F %T %Z')" >> "${log}"
    exit 0
  fi

  if ! tmux has-session -t fdm_n512_cuda 2>/dev/null; then
    printf '%s | evolution session stopped; restarting from checkpoint\n' \
      "$(date '+%F %T %Z')" >> "${log}"
    tmux new-session -d -s fdm_n512_cuda \
      "bash ${repository}/scripts/run_lageunha_n512_1myr.sh >> /gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_n512_lageunha.out 2>> /gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_n512_lageunha.err"
  fi

  record_status
  sleep 1800
done
