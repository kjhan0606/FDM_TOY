#!/bin/bash
#SBATCH --job-name=FDMTOY_offset_seed
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --array=0-1%1
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/periodic-offset-seed-%A_%a.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/periodic-offset-seed-%A_%a.err

set -euo pipefail
module purge
module load cuda/13.0.2

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
export PYTHONPATH=.:src:scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ "${SLURM_JOB_PARTITION:-}" != a10 || "${CUDA_VISIBLE_DEVICES:-}" != 0 ||
      ! "${SLURM_JOB_GPUS:-}" =~ ^[0-9]+$ ||
      ! "${SLURM_MEM_PER_NODE:-}" =~ ^[0-9]+$ ]] ||
   (( SLURM_MEM_PER_NODE < 16384 )); then
  printf 'one isolated a10 GPU and 16 GiB host memory are required\n' >&2
  exit 78
fi
if ! git ls-files --error-unmatch \
  scripts/derive_periodic_offset_seed.py \
  src/fdm_smbh_delay/pyul.py > /dev/null ||
   ! git diff --quiet HEAD -- \
  scripts/derive_periodic_offset_seed.py \
  src/fdm_smbh_delay/pyul.py; then
  printf 'offset seed source changed after registration\n' >&2
  exit 78
fi

labels=(whole_x half_x)
shifts=(1.0 0.5)
task_index=${SLURM_ARRAY_TASK_ID:-}
if [[ ! "$task_index" =~ ^[01]$ ]]; then
  printf 'unexpected offset task index: %s\n' "$task_index" >&2
  exit 78
fi
label=${labels[$task_index]}
shift_cells=${shifts[$task_index]}
reference=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/pyul_initial/qe_q030_e030_a020_n256
root=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_offset_seeds_v1
if [[ -e "$root/$label" ||
      ! -f "$reference/Outputs/3Wfn/P3D_#000.npy" ||
      ! -f "$reference/Outputs/NBody/NTM_#000.npy" ]]; then
  printf 'offset output exists or parent seed is missing\n' >&2
  exit 78
fi
/home/kjhan/miniconda3/bin/python -c 'from guard_cuda_process import NvmlComputeProcesses; m=NvmlComputeProcesses(0); n=m.visible_device_count(); p=m.pids(); m.close(); assert n == 1, f"ambiguous NVML GPU mapping: {n} visible devices"; assert not p, f"GPU already has compute processes: {sorted(p)}"'

mkdir -p "$root"
source_commit=$(git rev-parse HEAD)
printf 'project=FDM_TOY\njob_id=%s\narray_task_id=%s\nowner=%s\nsource_commit=%s\npurpose=periodic_offset_seed_%s\n' \
  "$SLURM_JOB_ID" "$task_index" "$USER" "$source_commit" "$label" \
  > "$root/OWNER_$label.txt"
/home/kjhan/miniconda3/bin/python scripts/derive_periodic_offset_seed.py \
  "$reference" --output "$root/$label" --axis x \
  --shift-cells "$shift_cells" --source-commit "$source_commit" \
  --device cuda:0 &
solver_pid=$!
guard_code=0
/home/kjhan/miniconda3/bin/python scripts/guard_cuda_process.py \
  --gpu-index 0 --solver-pid "$solver_pid" --poll-seconds 5 \
  --interrupt-grace-seconds 15 --status-file "$root/guard_$label.json" || guard_code=$?
if (( guard_code != 0 )); then
  kill -INT "$solver_pid" 2>/dev/null || true
fi
solver_code=0
wait "$solver_pid" || solver_code=$?
if (( guard_code != 0 || solver_code != 0 )); then
  printf 'offset seed stopped: label=%s guard=%s solver=%s\n' \
    "$label" "$guard_code" "$solver_code" >&2
  exit 75
fi
