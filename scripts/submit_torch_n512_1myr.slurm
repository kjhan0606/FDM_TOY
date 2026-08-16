#!/bin/bash
#SBATCH --job-name=fdm_n512_1myr
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_n512_1myr-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_n512_1myr-%j.err

set -euo pipefail

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python scripts/launch_torch_wave_case.py \
  /gpfs/kjhan/FDM_TOY_RESULTS/torch_initial/koo_n512_reference/koo_mbh1.0e8_n512 \
  --output /gpfs/kjhan/FDM_TOY_RESULTS/torch_long_term/koo_n512_cuda_1myr_s2048 \
  --duration-myr 1.0 \
  --save-number 2048 \
  --movie-frame-number 360 \
  --save-3d-number 32 \
  --checkpoint-every-saves 32 \
  --rk4-substeps 9 \
  --device cuda:0
