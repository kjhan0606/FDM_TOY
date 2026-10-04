#!/bin/bash
#SBATCH --job-name=FDMTOY_qe_q030_step_trace
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/step-trace-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/step-trace-%j.err

set -euo pipefail
module purge
module load cuda/13.0.2

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
export PYTHONPATH=.:src:scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

if [[ "${SLURM_JOB_PARTITION:-}" != a10 || "${CUDA_VISIBLE_DEVICES:-}" != 0 ||
      ! "${SLURM_JOB_GPUS:-}" =~ ^[0-9]+$ ||
      ! "${SLURM_MEM_PER_NODE:-}" =~ ^[0-9]+$ ]] ||
   (( SLURM_MEM_PER_NODE < 16384 )); then
  printf 'one exclusive a10 GPU and 16 GiB host memory are required\n' >&2
  exit 78
fi

root=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/startup_step_trace_v1
seed=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/pyul_initial/qe_q030_e030_a020_n256
if [[ -e "$root" || ! -f "$seed/Outputs/3Wfn/P3D_#000.npy" ]]; then
  printf 'trace output already exists or registered seed is missing\n' >&2
  exit 78
fi
mkdir "$root"
printf 'project=FDM_TOY\nrepo=%s\npurpose=q/e first-prefix per-step energy trace only\njob_id=%s\nowner=%s\nDO_NOT_CANCEL=other sessions must coordinate with this job owner\n' \
  "$PWD" "$SLURM_JOB_ID" "$USER" > "$root/OWNER.txt"
printf 'source_commit=%s job_id=%s physical_gpu=%s\n' \
  "$(git rev-parse HEAD)" "$SLURM_JOB_ID" "$SLURM_JOB_GPUS"

/home/kjhan/miniconda3/bin/python -c 'from guard_cuda_process import NvmlComputeProcesses; m=NvmlComputeProcesses(0); n=m.visible_device_count(); p=m.pids(); m.close(); assert n == 1, f"ambiguous NVML GPU mapping: {n} visible devices"; assert not p, f"GPU already has compute processes: {sorted(p)}"'

output="$root/n256_factor1_every_step"
/home/kjhan/miniconda3/bin/python scripts/run_torch_wave_case.py "$seed" \
  --output "$output" --duration-myr 0.1 --save-number 58500 \
  --save-3d-number 0 --movie-frame-number 0 --rk4-substeps 9 \
  --time-step-factor 1.0 --checkpoint-every-saves 50 \
  --diagnostic-stop-after-save 50 --device cuda:0 &
solver_pid=$!
guard_code=0
/home/kjhan/miniconda3/bin/python scripts/guard_cuda_process.py \
  --gpu-index 0 --solver-pid "$solver_pid" --poll-seconds 5 \
  --interrupt-grace-seconds 15 --status-file "$root/guard.json" || guard_code=$?
if (( guard_code != 0 )); then
  kill -INT "$solver_pid" 2>/dev/null || true
fi
solver_code=0
wait "$solver_pid" || solver_code=$?
if (( guard_code != 0 || solver_code != 0 )); then
  printf 'trace stopped: guard=%s solver=%s\n' "$guard_code" "$solver_code" >&2
  exit 75
fi
/home/kjhan/miniconda3/bin/python scripts/snapshot_torch_provenance.py "$output"
