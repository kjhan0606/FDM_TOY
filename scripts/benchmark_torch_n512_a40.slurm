#!/bin/bash
#SBATCH --job-name=fdm_n512_a40_bench
#SBATCH --partition=a40
#SBATCH --nodelist=syn07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_n512_a40-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_n512_a40-%j.err

set -euo pipefail

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

output="/gpfs/kjhan/FDM_TOY_RESULTS/torch_validation/koo_n512_a40_benchmark64_${SLURM_JOB_ID}"
srun python scripts/launch_torch_wave_case.py \
  /gpfs/kjhan/FDM_TOY_RESULTS/torch_initial/koo_n512_reference/koo_mbh1.0e8_n512 \
  --output "${output}" \
  --duration-myr 0.000063 \
  --save-number 1 \
  --movie-frame-number 0 \
  --save-3d-number 0 \
  --checkpoint-every-saves 0 \
  --rk4-substeps 9 \
  --device cuda:0
