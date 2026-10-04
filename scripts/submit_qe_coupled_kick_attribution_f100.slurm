#!/bin/bash
#SBATCH --job-name=FDMTOY_kick_attr
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/coupled-kick-attribution-f100-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/coupled-kick-attribution-f100-%j.err

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
  scripts/probe_coupled_kick_attribution.py \
  scripts/probe_wave_only_momentum_control.py \
  src/fdm_smbh_delay/periodic_mesh_coupling.py \
  src/fdm_smbh_delay/torch_wave.py \
  src/fdm_smbh_delay/pyul.py > /dev/null ||
   ! git diff --quiet HEAD -- \
  scripts/probe_coupled_kick_attribution.py \
  scripts/probe_wave_only_momentum_control.py \
  src/fdm_smbh_delay/periodic_mesh_coupling.py \
  src/fdm_smbh_delay/torch_wave.py \
  src/fdm_smbh_delay/pyul.py; then
  printf 'kick attribution numerical source changed after registration\n' >&2
  exit 78
fi

root=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/coupled_kick_attribution_v1
run=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_tsc_strang_momentum_step_trace_v1/f100
if [[ -e "$root" || ! -f "$run/Checkpoints/wave_000010.npy" ||
      ! -f "$run/Checkpoints/state_000010.npz" ]]; then
  printf 'attribution output exists or candidate checkpoint is missing\n' >&2
  exit 78
fi
/home/kjhan/miniconda3/bin/python -c 'from guard_cuda_process import NvmlComputeProcesses; m=NvmlComputeProcesses(0); n=m.visible_device_count(); p=m.pids(); m.close(); assert n == 1, f"ambiguous NVML GPU mapping: {n} visible devices"; assert not p, f"GPU already has compute processes: {sorted(p)}"'

mkdir "$root"
source_commit=$(git rev-parse HEAD)
printf 'project=FDM_TOY\njob_id=%s\nowner=%s\nsource_commit=%s\npurpose=coupled_kick_attribution_f100\n' \
  "$SLURM_JOB_ID" "$USER" "$source_commit" > "$root/OWNER.txt"
/home/kjhan/miniconda3/bin/python scripts/probe_coupled_kick_attribution.py \
  "$run" --output "$root/f100.json" --source-commit "$source_commit" \
  --device cuda:0 &
solver_pid=$!
guard_code=0
/home/kjhan/miniconda3/bin/python scripts/guard_cuda_process.py \
  --gpu-index 0 --solver-pid "$solver_pid" --poll-seconds 5 \
  --interrupt-grace-seconds 15 --status-file "$root/guard_f100.json" || guard_code=$?
if (( guard_code != 0 )); then
  kill -INT "$solver_pid" 2>/dev/null || true
fi
solver_code=0
wait "$solver_pid" || solver_code=$?
if (( guard_code != 0 || solver_code != 0 )); then
  printf 'kick attribution stopped: guard=%s solver=%s\n' \
    "$guard_code" "$solver_code" >&2
  exit 75
fi
