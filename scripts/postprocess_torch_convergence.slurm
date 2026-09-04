#!/bin/bash
#SBATCH --job-name=fdm_conv_analysis
#SBATCH --partition=a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_conv_analysis-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_conv_analysis-%j.err

set -euo pipefail

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

n256=/gpfs/kjhan/FDM_TOY_RESULTS/torch_convergence/koo_n256_cuda_1myr_s2048
n512=/gpfs/kjhan/FDM_TOY_RESULTS/torch_long_term/koo_n512_cuda_1myr_s2048
n512_dt05=/gpfs/kjhan/FDM_TOY_RESULTS/torch_convergence/koo_n512_cuda_dt05_0p1myr_s512
comparison_root=/gpfs/kjhan/FDM_TOY_RESULTS/torch_convergence

python scripts/analyze_pyul_wave_run.py "${n512_dt05}"
python scripts/analyze_pyul_secular_exchange.py "${n512_dt05}"
python scripts/analyze_pyul_line_density.py "${n512_dt05}"
python scripts/build_wave_exchange_table.py "${n512_dt05}" \
  --output "${n512_dt05}/wave_exchange.csv"

# Refresh the evolving 512^3 baseline before selecting the common interval.
python scripts/analyze_pyul_wave_run.py "${n512}"
python scripts/analyze_pyul_secular_exchange.py "${n512}"

python scripts/summarize_pyul_convergence.py \
  n512="${n512}" \
  n256="${n256}" \
  --output "${comparison_root}/koo_spatial_convergence_partial.json"

python scripts/summarize_pyul_convergence.py \
  dt1="${n512}" \
  dt05="${n512_dt05}" \
  --output "${comparison_root}/koo_temporal_convergence_0p1myr.json"

python scripts/summarize_pyul_convergence.py \
  n512_dt1="${n512}" \
  n512_dt05="${n512_dt05}" \
  n256_dt1="${n256}" \
  --output "${comparison_root}/koo_joint_convergence_0p1myr.json"
