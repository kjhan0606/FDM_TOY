#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s GPU_INDEX CASE\n' "$0" >&2
  exit 2
fi
if [[ $(hostname -s) != syn101 ]]; then
  printf 'This manual GPU runner is restricted to syn101.\n' >&2
  exit 2
fi

gpu_index=$1
case_name=$2
repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
result_root=/gpfs/kjhan/FDM_TOY_RESULTS
reference_root=${result_root}/torch_initial
output_root=${result_root}/torch_convergence

cd "${repository}"
export CUDA_VISIBLE_DEVICES="${gpu_index}"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cuda_library=/home/kjhan/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="${cuda_library}:${LD_LIBRARY_PATH:-}"

case "${case_name}" in
  n512_rk18)
    reference=${reference_root}/koo_n512_reference/koo_mbh1.0e8_n512
    output=${output_root}/koo_n512_cuda_rk18_0p1myr_s512
    arguments=(
      --duration-myr 0.1
      --save-number 512
      --movie-frame-number 128
      --save-3d-number 16
      --checkpoint-every-saves 32
      --rk4-substeps 18
      --time-step-factor 1.0
    )
    ;;
  n256_dt05)
    reference=${reference_root}/koo_n256_reference/koo_mbh1.0e8_n256
    output=${output_root}/koo_n256_cuda_dt05_1myr_s2048
    arguments=(
      --duration-myr 1.0
      --save-number 2048
      --movie-frame-number 360
      --save-3d-number 32
      --checkpoint-every-saves 32
      --rk4-substeps 9
      --time-step-factor 0.5
    )
    ;;
  *)
    printf 'unknown convergence case: %s\n' "${case_name}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${output}/torch_run_summary.json" ]]; then
  resume_arguments=()
  if [[ -d "${output}" ]]; then
    resume_arguments=(--resume)
  fi
  python scripts/launch_torch_wave_case.py "${reference}" \
    --output "${output}" \
    --device cuda:0 \
    "${resume_arguments[@]}" \
    "${arguments[@]}"
fi

python scripts/analyze_pyul_wave_run.py "${output}"
python scripts/analyze_pyul_secular_exchange.py "${output}"
python scripts/analyze_pyul_line_density.py "${output}"
python scripts/analyze_pyul_wave_response.py "${output}"
python scripts/build_wave_exchange_table.py "${output}" \
  --output "${output}/wave_exchange.csv"
