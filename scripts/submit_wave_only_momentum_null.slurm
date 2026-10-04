#!/bin/bash
#SBATCH --job-name=FDMTOY_wave_null
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/wave-only-momentum-null-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/wave-only-momentum-null-%j.err

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
  scripts/probe_wave_only_momentum_control.py \
  src/fdm_smbh_delay/torch_wave.py \
  src/fdm_smbh_delay/pyul.py > /dev/null ||
   ! git diff --quiet HEAD -- \
  scripts/probe_wave_only_momentum_control.py \
  src/fdm_smbh_delay/torch_wave.py \
  src/fdm_smbh_delay/pyul.py; then
  printf 'wave-only numerical worktree changed after registration\n' >&2
  exit 78
fi

root=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/wave_only_momentum_null_v1
seed=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/pyul_initial/qe_q030_e030_a020_n256
candidate=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_tsc_strang_momentum_step_trace_v1
source_commit=$(git rev-parse HEAD)
if [[ -e "$root" || ! -f "$seed/Outputs/3Wfn/P3D_#000.npy" ||
      ! -f "$seed/Outputs/NBody/NTM_#000.npy" ||
      ! -f "$candidate/f0125/fdm_adapter_metadata.json" ]]; then
  printf 'null output exists or required seed/candidate run is missing\n' >&2
  exit 78
fi
/home/kjhan/miniconda3/bin/python -c 'from guard_cuda_process import NvmlComputeProcesses; m=NvmlComputeProcesses(0); n=m.visible_device_count(); p=m.pids(); m.close(); assert n == 1, f"ambiguous NVML GPU mapping: {n} visible devices"; assert not p, f"GPU already has compute processes: {sorted(p)}"'

/home/kjhan/miniconda3/bin/python scripts/probe_wave_only_momentum_control.py \
  "$seed" "$candidate" --output "$root" --device cuda:0 \
  --source-commit "$source_commit" &
solver_pid=$!
guard_code=0
/home/kjhan/miniconda3/bin/python scripts/guard_cuda_process.py \
  --gpu-index 0 --solver-pid "$solver_pid" --poll-seconds 5 \
  --interrupt-grace-seconds 15 \
  --status-file "/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/wave-only-null-guard-${SLURM_JOB_ID}.json" || guard_code=$?
if (( guard_code != 0 )); then
  kill -INT "$solver_pid" 2>/dev/null || true
fi
solver_code=0
wait "$solver_pid" || solver_code=$?
if (( guard_code != 0 || solver_code != 0 )); then
  printf 'wave-only null stopped: guard=%s solver=%s\n' \
    "$guard_code" "$solver_code" >&2
  exit 75
fi
