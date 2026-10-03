#!/bin/bash
set -euo pipefail

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY

cuda_library=/home/kjhan/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="${cuda_library}:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2

output=/gpfs/kjhan/FDM_TOY_RESULTS/torch_long_term/koo_n512_cuda_1myr_s2048
resume_arguments=()
if [[ -d "${output}" ]]; then
  resume_arguments=(--resume)
fi

exec python scripts/launch_torch_wave_case.py \
  /gpfs/kjhan/FDM_TOY_RESULTS/torch_initial/koo_n512_reference/koo_mbh1.0e8_n512 \
  --output "${output}" \
  --duration-myr 1.0 \
  --save-number 2048 \
  --movie-frame-number 360 \
  --save-3d-number 32 \
  --checkpoint-every-saves 32 \
  --rk4-substeps 9 \
  --device cuda:0 \
  "${resume_arguments[@]}"
