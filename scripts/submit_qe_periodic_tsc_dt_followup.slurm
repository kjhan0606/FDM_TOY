#!/bin/bash
#SBATCH --job-name=FDMTOY_qe_tsc_dt
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/periodic-tsc-dt-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/periodic-tsc-dt-%j.err

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

root=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_tsc_dt_followup_v1
seed=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/pyul_initial/qe_q030_e030_a020_n256
if [[ -e "$root" || ! -f "$seed/Outputs/3Wfn/P3D_#000.npy" ||
      ! -f "$seed/Outputs/NBody/NTM_#000.npy" ]]; then
  printf 'follow-up output already exists or registered seed is missing\n' >&2
  exit 78
fi
mkdir "$root"
printf 'project=FDM_TOY\nrepo=%s\npurpose=periodic reciprocal TSC half/quarter-step diagnostic only\njob_id=%s\nowner=%s\nsource_commit=%s\n' \
  "$PWD" "$SLURM_JOB_ID" "$USER" "$(git rev-parse HEAD)" > "$root/OWNER.txt"

for factor in 0.5 0.25; do
  case "$factor" in
    0.5) label=f050 ;;
    0.25) label=f025 ;;
  esac
  /home/kjhan/miniconda3/bin/python -c 'from guard_cuda_process import NvmlComputeProcesses; m=NvmlComputeProcesses(0); n=m.visible_device_count(); p=m.pids(); m.close(); assert n == 1, f"ambiguous NVML GPU mapping: {n} visible devices"; assert not p, f"GPU already has compute processes: {sorted(p)}"'
  output="$root/$label"
  marker="$root/solver_$label.pid"
  FDM_SOLVER_PID_FILE="$marker" \
  /home/kjhan/miniconda3/bin/python scripts/launch_torch_wave_case.py "$seed" \
    --output "$output" --duration-myr 0.1 --save-number 58500 \
    --save-3d-number 0 --movie-frame-number 0 --rk4-substeps 9 \
    --time-step-factor "$factor" --checkpoint-every-saves 10 \
    --diagnostic-stop-after-save 10 --wave-smbh-coupling periodic_tsc_reciprocal \
    --device cuda:0 &
  launcher_pid=$!
  for attempt in {1..100}; do
    if [[ -s "$marker" ]]; then
      break
    fi
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if [[ ! -s "$marker" ]]; then
    wait "$launcher_pid" || true
    printf 'launcher did not publish the owned solver PID: %s\n' "$label" >&2
    exit 75
  fi
  read -r solver_pid < "$marker"
  guard_code=0
  /home/kjhan/miniconda3/bin/python scripts/guard_cuda_process.py \
    --gpu-index 0 --solver-pid "$solver_pid" --poll-seconds 5 \
    --interrupt-grace-seconds 15 \
    --status-file "$root/guard_$label.json" || guard_code=$?
  if (( guard_code != 0 )); then
    kill -INT "$solver_pid" 2>/dev/null || true
  fi
  solver_code=0
  wait "$launcher_pid" || solver_code=$?
  if (( guard_code != 0 || solver_code != 0 )); then
    printf 'diagnostic stopped: factor=%s guard=%s solver=%s\n' \
      "$factor" "$guard_code" "$solver_code" >&2
    exit 75
  fi
  /home/kjhan/miniconda3/bin/python scripts/snapshot_torch_provenance.py "$output"
done
