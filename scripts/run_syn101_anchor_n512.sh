#!/bin/bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  printf 'usage: %s GPU_INDEX CASE_ID DURATION_MYR SAVE_NUMBER BOX_PC\n' "$0" >&2
  exit 2
fi
if [[ $(hostname -s) != syn101 ]]; then
  printf 'This manual GPU runner is restricted to syn101.\n' >&2
  exit 2
fi

gpu_index=$1
case_id=$2
duration_myr=$3
save_number=$4
box_pc=$5
repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
result_root=/gpfs/kjhan/FDM_TOY_RESULTS
reference_root=${result_root}/torch_initial/tier0_n512
reference_run=${reference_root}/${case_id}_n512
output=${result_root}/torch_calibration/tier0_n512/${case_id}_n512

cd "${repository}"
export CUDA_VISIBLE_DEVICES="${gpu_index}"
export SLURM_CPUS_PER_TASK=4
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cuda_library=/home/kjhan/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="${cuda_library}:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${reference_run}/Outputs/3Wfn/P3D_#000.npy" ]]; then
  python scripts/run_pyul_wave_case.py \
    --pyul-path /gpfs/kjhan/FDM_TOY_DEPS/PyUL_NBody \
    --case-id "${case_id}" \
    --resolution 512 \
    --duration-myr 0.000001 \
    --save-number 1 \
    --save-3d \
    --rk-steps 36 \
    --box-pc "${box_pc}" \
    --output "${reference_root}"
fi

if [[ ! -f "${output}/torch_run_summary.json" ]]; then
  resume_arguments=()
  if [[ -d "${output}" ]]; then
    resume_arguments=(--resume)
  fi
  python scripts/launch_torch_wave_case.py "${reference_run}" \
    --output "${output}" \
    --duration-myr "${duration_myr}" \
    --save-number "${save_number}" \
    --movie-frame-number 128 \
    --save-3d-number 16 \
    --checkpoint-every-saves 32 \
    --rk4-substeps 9 \
    --device cuda:0 \
    "${resume_arguments[@]}"
fi

python scripts/analyze_pyul_wave_run.py "${output}"
python scripts/analyze_pyul_secular_exchange.py "${output}"
python scripts/analyze_pyul_line_density.py "${output}"
python scripts/analyze_pyul_wave_response.py "${output}"
python scripts/build_wave_exchange_table.py "${output}" \
  --output "${output}/wave_exchange.csv"
