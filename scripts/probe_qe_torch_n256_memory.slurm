#!/bin/bash
#SBATCH --job-name=qe-wave-mem-n256
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/logs/memory_probe_n256-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/logs/memory_probe_n256-%j.err

set -euo pipefail
module purge
module load cuda/13.0.2

source_root="${FDM_PROBE_SOURCE_ROOT:-/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY}"
probe_variant="${FDM_PROBE_VARIANT:-optimized}"
if [[ ! "${probe_variant}" =~ ^[a-z0-9_]+$ ]]; then
  printf 'invalid memory-probe variant: %s\n' "${probe_variant}" >&2
  exit 1
fi
cd "${source_root}"
export PYTHONPATH=.:src
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

reference=/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/pyul_initial/qe_q100_e000_a020_n256
output="/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/torch/memory_probe_n256_${probe_variant}_${SLURM_JOB_ID}"
if [[ -e "${output}" ]]; then
  printf 'refusing existing memory-probe output: %s\n' "${output}" >&2
  exit 1
fi

# A single wave step exercises FFTs, the live SMBH force, and both energy
# diagnostics. This is only a memory probe, never a calibration trajectory.
printf 'solver_revision=%s output=%s\n' "$(git rev-parse HEAD)" "${output}"
profile_args=()
if [[ "${FDM_PROBE_STAGE_PROFILE:-0}" == "1" ]]; then
  profile_args=(--profile-memory-stages)
fi
srun /home/kjhan/miniconda3/bin/python scripts/launch_torch_wave_case.py \
  "${reference}" \
  --output "${output}" \
  --duration-myr 0.000001 \
  --save-number 1 \
  --movie-frame-number 0 \
  --save-3d-number 0 \
  --checkpoint-every-saves 0 \
  --rk4-substeps 9 \
  --time-step-factor 1.0 \
  --device cuda:0 \
  "${profile_args[@]}"
