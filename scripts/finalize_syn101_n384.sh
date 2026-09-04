#!/bin/bash
set -euo pipefail

if [[ $(hostname -s) != syn101 ]]; then
  printf 'This n384 finalizer is restricted to syn101.\n' >&2
  exit 2
fi

repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
result_root=/gpfs/kjhan/FDM_TOY_RESULTS
run=${result_root}/torch_convergence/koo_n384_cuda_1myr_s2048
n256=${result_root}/torch_convergence/koo_n256_cuda_1myr_s2048
n512=${result_root}/torch_long_term/koo_n512_cuda_1myr_s2048
comparison=${result_root}/torch_convergence/koo_spatial_convergence_3level_1myr.json
state_directory=${run}/Finalization
failure_marker=${state_directory}/convergence.failed
log=${result_root}/logs/fdm_n384_finalizer.log

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

mkdir -p "${state_directory}" "$(dirname "${log}")"
rm -f "${failure_marker}"
cd "${repository}"

record() {
  printf '%s | %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${log}"
}

record_failure() {
  local status=$?
  trap - ERR
  record "n384 finalizer failed with status ${status}; automatic retry disabled"
  touch "${failure_marker}"
  exit "${status}"
}
trap record_failure ERR

summary=${run}/torch_run_summary.json
python -c \
  'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "complete"' \
  "${summary}"

record 'conservation analysis starting'
python scripts/analyze_pyul_wave_run.py "${run}" >> "${log}" 2>&1
record 'orbit-averaged exchange analysis starting'
python scripts/analyze_pyul_secular_exchange.py "${run}" >> "${log}" 2>&1
record 'line-density analysis starting'
python scripts/analyze_pyul_line_density.py "${run}" >> "${log}" 2>&1
record 'three-level spatial comparison starting'
python scripts/summarize_pyul_convergence.py \
  n512="${n512}" \
  n384="${run}" \
  n256="${n256}" \
  --output "${comparison}" >> "${log}" 2>&1

required_outputs=(
  "${run}/conservation_summary.json"
  "${run}/orbit_averaged_exchange_summary.json"
  "${run}/line_density_summary.json"
  "${comparison}"
)
for output in "${required_outputs[@]}"; do
  if [[ ! -s ${output} ]]; then
    record "n384 verification failed: missing or empty ${output}"
    exit 4
  fi
done

touch "${state_directory}/convergence.complete"
rm -f "${failure_marker}"
trap - ERR
record 'n384 lightweight finalization complete'
