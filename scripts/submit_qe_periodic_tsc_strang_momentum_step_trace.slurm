#!/bin/bash
#SBATCH --job-name=FDMTOY_qe_mom_step
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/periodic-tsc-strang-momentum-step-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/logs/periodic-tsc-strang-momentum-step-%j.err

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

expected_commit=d6709d2368ddd56ade596c6cbe79e28b6a749d15
if ! printf '%s\n' \
  'b5de6d1f9e964e3b8e0eca130b8e88bc4e54d6c7bdfc32e5f651527fd24c1156  scripts/run_torch_wave_case.py' \
  '3b4952cea5b552265af854068ef08f9fb1c3d19b2863a925bb9af37606272d06  src/fdm_smbh_delay/torch_wave.py' \
  'fbf928849179bb701d7819d6cfd60d988bb2a82f81b2fa451e0e5e7a656f09a9  src/fdm_smbh_delay/periodic_mesh_coupling.py' \
  'e531203ac43809df3c13c0b05c87cf7534197207fab53660c683b4e2afe786d1  src/fdm_smbh_delay/pyul.py' |
  sha256sum --check --status; then
  printf 'numerical source changed since diagnostic registration\n' >&2
  exit 78
fi

root=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_tsc_strang_momentum_step_trace_v1
seed=/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/pyul_initial/qe_q030_e030_a020_n256
if [[ -e "$root" || ! -f "$seed/Outputs/3Wfn/P3D_#000.npy" ||
      ! -f "$seed/Outputs/NBody/NTM_#000.npy" ]]; then
  printf 'momentum-force trace output already exists or registered seed is missing\n' >&2
  exit 78
fi
mkdir "$root"
printf 'project=FDM_TOY\nrepo=%s\npurpose=experimental spectral-momentum Strang short-prefix diagnostic only\njob_id=%s\nowner=%s\nsource_commit=%s\n' \
  "$PWD" "$SLURM_JOB_ID" "$USER" "$expected_commit" > "$root/OWNER.txt"

for factor in 1.0 0.5 0.25 0.125; do
  case "$factor" in
    1.0) label=f100; saves=58500; stop=10 ;;
    0.5) label=f050; saves=117000; stop=20 ;;
    0.25) label=f025; saves=234000; stop=40 ;;
    0.125) label=f0125; saves=468000; stop=80 ;;
  esac
  /home/kjhan/miniconda3/bin/python -c 'from guard_cuda_process import NvmlComputeProcesses; m=NvmlComputeProcesses(0); n=m.visible_device_count(); p=m.pids(); m.close(); assert n == 1, f"ambiguous NVML GPU mapping: {n} visible devices"; assert not p, f"GPU already has compute processes: {sorted(p)}"'
  output="$root/$label"
  marker="$root/solver_$label.pid"
  FDM_SOLVER_PID_FILE="$marker" \
  /home/kjhan/miniconda3/bin/python scripts/launch_torch_wave_case.py "$seed" \
    --output "$output" --duration-myr 0.1 --save-number "$saves" \
    --save-3d-number 0 --movie-frame-number 0 --rk4-substeps 1 \
    --time-step-factor "$factor" --checkpoint-every-saves 10 \
    --diagnostic-stop-after-save "$stop" \
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
    printf 'momentum-force trace stopped: factor=%s guard=%s solver=%s\n' \
      "$factor" "$guard_code" "$solver_code" >&2
    exit 75
  fi
  /home/kjhan/miniconda3/bin/python scripts/snapshot_torch_provenance.py "$output"
done
