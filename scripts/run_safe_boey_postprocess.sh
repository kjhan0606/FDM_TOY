#!/bin/bash
set -euo pipefail

if [[ $(hostname -s) != syntax ]]; then
  printf 'This bounded CPU post-processor is restricted to syntax.\n' >&2
  exit 2
fi

repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
result_root=/gpfs/kjhan/FDM_TOY_RESULTS/torch_calibration/tier0_n512
log_root=/gpfs/kjhan/FDM_TOY_RESULTS/logs

mkdir -p "${log_root}"
exec 9>"${log_root}/fdm_boey_cpu_safe.lock"
if ! flock -n 9; then
  printf 'Another bounded Boey post-processor already holds the lock.\n' >&2
  exit 3
fi

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# Bound virtual memory to 64 GiB. The measured 512^3 wave-response RSS is
# approximately 37 GB, so this leaves FFT workspace headroom without allowing
# an unbounded allocation on the shared node.
ulimit -v 67108864

cd "${repository}"

wave_response_complete() {
  local target_run=$1
  if (
    [[ -s ${target_run}/wave_response_summary.json ]] \
    && [[ -s ${target_run}/wave_response_timeseries.csv ]] \
    && [[ -s ${target_run}/wave_radial_profiles.csv ]]
  ); then
    rm -f \
      "${target_run}/wave_response_timeseries.partial.csv" \
      "${target_run}/wave_radial_profiles.partial.csv"
    return 0
  fi
  return 1
}

for case_id in \
  boey_each02pct_n512 \
  boey_each05pct_n512 \
  boey_each10pct_n512
do
  run=${result_root}/${case_id}
  printf '%s | %s | secular exchange\n' "$(date '+%F %T %Z')" "${case_id}"
  python scripts/analyze_pyul_secular_exchange.py "${run}"
  printf '%s | %s | line density\n' "$(date '+%F %T %Z')" "${case_id}"
  python scripts/analyze_pyul_line_density.py "${run}"
  printf '%s | %s | resumable wave response\n' "$(date '+%F %T %Z')" "${case_id}"
  sample_count=$(python -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["saved_3d_states"]))' \
    "${run}/fdm_adapter_metadata.json")
  if [[ ! ${sample_count} =~ ^[1-9][0-9]*$ ]]; then
    printf 'invalid saved_3d_states for %s: %s\n' \
      "${run}" "${sample_count}" >&2
    exit 3
  fi
  for ((invocation = 1; invocation <= sample_count; invocation++))
  do
    if wave_response_complete "${run}"; then
      break
    fi
    python scripts/analyze_pyul_wave_response.py "${run}" \
      --resume --max-new-samples 1
  done
  if ! wave_response_complete "${run}"; then
    printf '%s | %s | wave response did not complete\n' \
      "$(date '+%F %T %Z')" "${case_id}" >&2
    exit 3
  fi
  printf '%s | %s | exchange table\n' "$(date '+%F %T %Z')" "${case_id}"
  python scripts/build_wave_exchange_table.py "${run}" \
    --output "${run}/wave_exchange.csv"
  printf '%s | %s | complete\n' "$(date '+%F %T %Z')" "${case_id}"
done
