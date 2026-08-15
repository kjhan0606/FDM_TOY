#!/bin/bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  printf 'usage: %s [GPU_INDEX]\n' "$0" >&2
  exit 2
fi
if [[ $(hostname -s) != syn101 ]]; then
  printf 'This manual GPU runner is restricted to syn101.\n' >&2
  exit 2
fi

gpu_index=${1:-0}
if [[ ! ${gpu_index} =~ ^[0-7]$ ]]; then
  printf 'GPU_INDEX must lie between 0 and 7.\n' >&2
  exit 2
fi

repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
result_root=/gpfs/kjhan/FDM_TOY_RESULTS
reference_root=${result_root}/torch_initial/koo_n384_reference
reference_run=${reference_root}/koo_mbh1.0e8_n384
output=${result_root}/torch_convergence/koo_n384_cuda_1myr_s2048
pid_marker=${result_root}/logs/fdm_n384_solver.pid

cd "${repository}"
export CUDA_VISIBLE_DEVICES="${gpu_index}"
export SLURM_CPUS_PER_TASK=4
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FDM_SOLVER_PID_FILE="${pid_marker}"
cuda_library=/home/kjhan/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="${cuda_library}:${LD_LIBRARY_PATH:-}"

if [[ ! -f ${reference_run}/Outputs/3Wfn/P3D_#000.npy ]]; then
  if [[ -d ${reference_run} ]]; then
    printf 'Incomplete n384 reference requires inspection: %s\n' \
      "${reference_run}" >&2
    exit 3
  fi
  python scripts/run_pyul_wave_case.py \
    --pyul-path /gpfs/kjhan/FDM_TOY_DEPS/PyUL_NBody \
    --case-id koo_mbh1.0e8 \
    --resolution 384 \
    --duration-myr 0.000001 \
    --save-number 1 \
    --save-3d \
    --rk-steps 36 \
    --box-pc 40 \
    --output "${reference_root}"
fi

if [[ -f ${output}/torch_run_summary.json ]]; then
  printf 'Complete n384 output already exists: %s\n' "${output}"
  exit 0
fi

resume_arguments=()
if [[ -d ${output} ]]; then
  resume_arguments=(--resume)
fi
exec python scripts/launch_torch_wave_case.py "${reference_run}" \
  --output "${output}" \
  --duration-myr 1.0 \
  --save-number 2048 \
  --movie-frame-number 360 \
  --save-3d-number 32 \
  --checkpoint-every-saves 32 \
  --rk4-substeps 9 \
  --time-step-factor 1.0 \
  --device cuda:0 \
  "${resume_arguments[@]}"
