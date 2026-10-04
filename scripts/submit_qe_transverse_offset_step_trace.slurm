#!/bin/bash
#SBATCH --job-name=FDMTOY_transverse_step
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%1
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/transverse-offset-step-%A_%a.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/transverse-offset-step-%A_%a.err

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
if ! printf '%s\n' \
  'b5de6d1f9e964e3b8e0eca130b8e88bc4e54d6c7bdfc32e5f651527fd24c1156  scripts/run_torch_wave_case.py' \
  '3b4952cea5b552265af854068ef08f9fb1c3d19b2863a925bb9af37606272d06  src/fdm_smbh_delay/torch_wave.py' \
  'fbf928849179bb701d7819d6cfd60d988bb2a82f81b2fa451e0e5e7a656f09a9  src/fdm_smbh_delay/periodic_mesh_coupling.py' \
  'e531203ac43809df3c13c0b05c87cf7534197207fab53660c683b4e2afe786d1  src/fdm_smbh_delay/pyul.py' |
  sha256sum --check --status; then
  printf 'numerical solver source differs from the registered baseline\n' >&2
  exit 78
fi

axes=(y z)
task_index=${SLURM_ARRAY_TASK_ID:-}
if [[ ! "$task_index" =~ ^[01]$ ]]; then
  printf 'unexpected transverse step task index: %s\n' "$task_index" >&2
  exit 78
fi
axis=${axes[$task_index]}
seed_root=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_transverse_offset_seeds_v1
root=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_transverse_offset_step_trace_v1
seed="$seed_root/half_$axis"
label=half_${axis}_f0125
output="$root/$label"
if [[ -e "$output" || ! -f "$seed/offset_seed_manifest.json" ]]; then
  printf 'transverse output exists or source-bound seed is missing\n' >&2
  exit 78
fi
case "$axis" in
  y) expected_seed_sha256=b28e7fa66f090c2f77989d0f64b4d7aff53dd2d013b8dd5e484925afeb72bd5c ;;
  z) expected_seed_sha256=a882c64a2db538ddb8c5cf4b58ddf98355fd22a95800e32791dfbf10e66b5387 ;;
esac
if ! printf '%s  %s\n' "$expected_seed_sha256" "$seed/offset_seed_manifest.json" |
  sha256sum --check --status; then
  printf 'transverse seed manifest differs from registered input\n' >&2
  exit 78
fi
/home/kjhan/miniconda3/bin/python -c 'from guard_cuda_process import NvmlComputeProcesses; m=NvmlComputeProcesses(0); n=m.visible_device_count(); p=m.pids(); m.close(); assert n == 1, f"ambiguous NVML GPU mapping: {n} visible devices"; assert not p, f"GPU already has compute processes: {sorted(p)}"'

mkdir -p "$root"
source_commit=$(git rev-parse HEAD)
printf 'project=FDM_TOY\njob_id=%s\narray_task_id=%s\nowner=%s\nsource_commit=%s\nseed_manifest_sha256=%s\npurpose=transverse_offset_step_%s\n' \
  "$SLURM_JOB_ID" "$task_index" "$USER" "$source_commit" \
  "$expected_seed_sha256" "$label" > "$root/OWNER_$label.txt"
marker="$root/solver_$label.pid"
FDM_SOLVER_PID_FILE="$marker" \
/home/kjhan/miniconda3/bin/python scripts/launch_torch_wave_case.py "$seed" \
  --output "$output" --duration-myr 0.1 --save-number 468000 \
  --save-3d-number 0 --movie-frame-number 0 --rk4-substeps 1 \
  --time-step-factor 0.125 --checkpoint-every-saves 10 \
  --diagnostic-stop-after-save 80 \
  --wave-smbh-coupling periodic_tsc_strang_momentum --device cuda:0 &
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
  printf 'transverse launcher did not publish owned solver PID: %s\n' "$label" >&2
  exit 75
fi
read -r solver_pid < "$marker"
guard_code=0
/home/kjhan/miniconda3/bin/python scripts/guard_cuda_process.py \
  --gpu-index 0 --solver-pid "$solver_pid" --poll-seconds 5 \
  --interrupt-grace-seconds 15 --status-file "$root/guard_$label.json" || guard_code=$?
if (( guard_code != 0 )); then
  kill -INT "$solver_pid" 2>/dev/null || true
fi
solver_code=0
wait "$launcher_pid" || solver_code=$?
if (( guard_code != 0 || solver_code != 0 )); then
  printf 'transverse step stopped: axis=%s guard=%s solver=%s\n' \
    "$axis" "$guard_code" "$solver_code" >&2
  exit 75
fi
/home/kjhan/miniconda3/bin/python scripts/snapshot_torch_provenance.py "$output"
