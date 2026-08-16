#!/bin/bash
set -euo pipefail

repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
run=/gpfs/kjhan/FDM_TOY_RESULTS/torch_parameter_grid/grid004_n256_box20_0p1myr
log=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_grid004_n256_monitor.log

mkdir -p "$(dirname "${log}")"

while true; do
  timestamp=$(date '+%F %T %Z')
  if [[ -f "${run}/torch_run_summary.json" ]]; then
    printf '%s | evolution complete; starting diagnostics\n' "${timestamp}" >> "${log}"
    cd "${repository}" || exit 1
    python scripts/analyze_pyul_wave_run.py "${run}" >> "${log}" 2>&1
    python scripts/analyze_pyul_secular_exchange.py "${run}" >> "${log}" 2>&1
    python scripts/analyze_pyul_line_density.py "${run}" >> "${log}" 2>&1
    python scripts/build_wave_exchange_table.py "${run}" \
      --output "${run}/wave_exchange.csv" >> "${log}" 2>&1
    printf '%s | diagnostics complete\n' "$(date '+%F %T %Z')" >> "${log}"
    exit 0
  fi

  jobs=$(squeue -h -u "${USER}" -n fdm_grid004_n256 -o '%i:%T:%M:%R')
  if [[ -z "${jobs}" ]]; then
    cd "${repository}" || exit 1
    new_job=$(sbatch --parsable scripts/run_grid004_n256_a40.slurm)
    jobs="${new_job}:SUBMITTED:0:pending"
  fi
  checkpoint=$(tr -d '\n' < "${run}/Checkpoints/latest.json" 2>/dev/null || true)
  printf '%s | jobs %s | checkpoint %s\n' \
    "${timestamp}" "${jobs}" "${checkpoint}" >> "${log}"
  sleep 1800
done
