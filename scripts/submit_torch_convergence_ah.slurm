#!/bin/bash
#SBATCH --job-name=fdm_conv
#SBATCH --partition=a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_conv-%A_%a.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_conv-%A_%a.err

set -euo pipefail

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

reference_root=/gpfs/kjhan/FDM_TOY_RESULTS/torch_initial
result_root=/gpfs/kjhan/FDM_TOY_RESULTS/torch_convergence

run_cuda_case() {
  local reference=$1
  local output=$2
  shift 2

  if [[ -f "${output}/torch_run_summary.json" ]]; then
    printf 'complete output already exists: %s\n' "${output}"
    return
  fi

  local resume_arguments=()
  if [[ -d "${output}" ]]; then
    resume_arguments=(--resume)
  fi

  srun python scripts/launch_torch_wave_case.py "${reference}" \
    --output "${output}" \
    --device cuda:0 \
    "${resume_arguments[@]}" \
    "$@"
}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    reference_directory="${reference_root}/koo_n256_reference"
    reference_run="${reference_directory}/koo_mbh1.0e8_n256"
    if [[ ! -f "${reference_run}/Outputs/3Wfn/P3D_#000.npy" ]]; then
      python scripts/run_pyul_wave_case.py \
        --pyul-path /gpfs/kjhan/FDM_TOY_DEPS/PyUL_NBody \
        --case-id koo_mbh1.0e8 \
        --resolution 256 \
        --duration-myr 0.000001 \
        --save-number 1 \
        --save-3d \
        --rk-steps 36 \
        --box-pc 40 \
        --output "${reference_directory}"
    fi
    run_cuda_case \
      "${reference_run}" \
      "${result_root}/koo_n256_cuda_1myr_s2048" \
      --duration-myr 1.0 \
      --save-number 2048 \
      --movie-frame-number 360 \
      --save-3d-number 32 \
      --checkpoint-every-saves 32 \
      --rk4-substeps 9 \
      --time-step-factor 1.0
    ;;
  1)
    run_cuda_case \
      "${reference_root}/koo_n512_reference/koo_mbh1.0e8_n512" \
      "${result_root}/koo_n512_cuda_dt05_0p1myr_s512" \
      --duration-myr 0.1 \
      --save-number 512 \
      --movie-frame-number 128 \
      --save-3d-number 16 \
      --checkpoint-every-saves 32 \
      --rk4-substeps 9 \
      --time-step-factor 0.5
    ;;
  *)
    printf 'unknown array index: %s\n' "${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac
