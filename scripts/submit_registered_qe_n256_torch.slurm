#!/bin/bash
#SBATCH --job-name=qe-reg-n256-torch
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/logs/qe-reg-n256-torch-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/logs/qe-reg-n256-torch-%j.err

set -euo pipefail
module purge
module load cuda/13.0.2

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
export PYTHONPATH=.:src
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0" || ! "${SLURM_JOB_GPUS:-}" =~ ^[0-9]+$ ]]; then
  printf 'unsupported Slurm GPU mapping: visible=%s allocated=%s\n' \
    "${CUDA_VISIBLE_DEVICES:-<unset>}" "${SLURM_JOB_GPUS:-<unset>}" >&2
  exit 78
fi

printf 'source_commit=%s job_id=%s visible_gpu=%s physical_gpu=%s\n' \
  "$(git rev-parse HEAD)" "${SLURM_JOB_ID}" \
  "${CUDA_VISIBLE_DEVICES}" "${SLURM_JOB_GPUS}"

srun /home/kjhan/miniconda3/bin/python scripts/run_guarded_qe_plan.py \
  --manifest results/wave_calibration_qe_followup_q100e000/run_manifest.csv \
  --cases results/wave_calibration_qe_extension/physical_cases.csv \
  --qe-design results/wave_calibration_qe_followup_q100e000/design.json \
  --initial-root /gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q100e000_v1/pyul_initial \
  --torch-root /gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q100e000_v1/torch \
  --pyul-path /gpfs/kjhan/FDM_TOY_DEPS/PyUL_NBody \
  --log-root /gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q100e000_v1/logs \
  --gpu-index 0 \
  --run-id qe_q100_e000_a020_n256 \
  --expected-stage torch \
  --maximum-gpu-stages 1
