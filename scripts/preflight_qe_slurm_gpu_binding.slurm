#!/bin/bash
#SBATCH --job-name=qe-gpu-binding
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/logs/qe-gpu-binding-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/logs/qe-gpu-binding-%j.err

set -euo pipefail
module purge
module load cuda/13.0.2

printf 'job_id=%s node=%s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'SLURM_JOB_GPUS=%s\n' "${SLURM_JOB_GPUS:-<unset>}"
nvidia-smi --query-gpu=index,uuid,memory.total --format=csv,noheader
