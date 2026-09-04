#!/bin/bash
#SBATCH --job-name=fdm_grid004_n256
#SBATCH --partition=a40
#SBATCH --nodelist=syn07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_grid004_n256-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/logs/fdm_grid004_n256-%j.err

set -euo pipefail

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

reference_root=/gpfs/kjhan/FDM_TOY_RESULTS/torch_initial/grid004_n256_box20_reference
reference_run="${reference_root}/grid_004_n256"
if [[ ! -f "${reference_run}/Outputs/3Wfn/P3D_#000.npy" ]]; then
  python scripts/run_pyul_wave_case.py \
    --pyul-path /gpfs/kjhan/FDM_TOY_DEPS/PyUL_NBody \
    --case-id grid_004 \
    --resolution 256 \
    --duration-myr 0.000001 \
    --save-number 1 \
    --save-3d \
    --rk-steps 36 \
    --box-pc 20 \
    --output "${reference_root}"
fi

output=/gpfs/kjhan/FDM_TOY_RESULTS/torch_parameter_grid/grid004_n256_box20_0p1myr
resume_arguments=()
if [[ -d "${output}" ]]; then
  resume_arguments=(--resume)
fi

python scripts/launch_torch_wave_case.py "${reference_run}" \
  --output "${output}" \
  --duration-myr 0.1 \
  --save-number 512 \
  --movie-frame-number 128 \
  --save-3d-number 16 \
  --checkpoint-every-saves 32 \
  --rk4-substeps 9 \
  --device cuda:0 \
  "${resume_arguments[@]}"
