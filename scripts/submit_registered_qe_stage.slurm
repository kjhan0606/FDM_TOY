#!/bin/bash
#SBATCH --job-name=qe-registered-stage
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/logs/qe-registered-stage-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/logs/qe-registered-stage-%j.err

set -euo pipefail
module purge
module load cuda/13.0.2

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
export PYTHONPATH=.:src
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

: "${QE_DESIGN_DIR:?set a registered q/e design directory}"
: "${QE_RESULT_ROOT:?set a distinct q/e result root}"
: "${QE_RUN_ID:?set one design-bound run ID}"
: "${QE_STAGE:?set seed or torch}"

if [[ "${QE_STAGE}" != seed && "${QE_STAGE}" != torch ]]; then
  printf 'invalid QE_STAGE=%s\n' "${QE_STAGE}" >&2
  exit 78
fi
if [[ ! "${QE_RUN_ID}" =~ _n(256|384|768)$ ]]; then
  printf 'unsupported QE_RUN_ID=%s\n' "${QE_RUN_ID}" >&2
  exit 78
fi
resolution="${BASH_REMATCH[1]}"
case "${resolution}" in
  256) required_mem_mib=16384; required_partition=a10 ;;
  384) required_mem_mib=32768; required_partition=a10 ;;
  768) required_mem_mib=196608; required_partition=large_gpu ;;
esac
if [[ ! "${SLURM_MEM_PER_NODE:-}" =~ ^[0-9]+$ ]] ||
   (( SLURM_MEM_PER_NODE < required_mem_mib )); then
  printf 'n%s requires at least %s MiB Slurm memory; allocated=%s\n' \
    "${resolution}" "${required_mem_mib}" "${SLURM_MEM_PER_NODE:-<unset>}" >&2
  exit 78
fi
if [[ "${required_partition}" == a10 && "${SLURM_JOB_PARTITION:-}" != a10 ]] ||
   [[ "${required_partition}" == large_gpu && ! "${SLURM_JOB_PARTITION:-}" =~ ^(a100|h100|h200)$ ]]; then
  printf 'unsupported n%s partition=%s\n' \
    "${resolution}" "${SLURM_JOB_PARTITION:-<unset>}" >&2
  exit 78
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0" || ! "${SLURM_JOB_GPUS:-}" =~ ^[0-9]+$ ]]; then
  printf 'unsupported Slurm GPU mapping: visible=%s allocated=%s\n' \
    "${CUDA_VISIBLE_DEVICES:-<unset>}" "${SLURM_JOB_GPUS:-<unset>}" >&2
  exit 78
fi
if [[ ! -f "${QE_DESIGN_DIR}/design.json" || ! -f "${QE_DESIGN_DIR}/run_manifest.csv" ||
      "${QE_RESULT_ROOT}" != /gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_* ]]; then
  printf 'registered design or owned result root is invalid\n' >&2
  exit 78
fi

printf 'source_commit=%s job_id=%s run_id=%s stage=%s visible_gpu=%s physical_gpu=%s\n' \
  "$(git rev-parse HEAD)" "${SLURM_JOB_ID}" "${QE_RUN_ID}" "${QE_STAGE}" \
  "${CUDA_VISIBLE_DEVICES}" "${SLURM_JOB_GPUS}"

srun /home/kjhan/miniconda3/bin/python scripts/run_guarded_qe_plan.py \
  --manifest "${QE_DESIGN_DIR}/run_manifest.csv" \
  --cases results/wave_calibration_qe_extension/physical_cases.csv \
  --qe-design "${QE_DESIGN_DIR}/design.json" \
  --initial-root "${QE_RESULT_ROOT}/pyul_initial" \
  --torch-root "${QE_RESULT_ROOT}/torch" \
  --pyul-path /gpfs/kjhan/FDM_TOY_DEPS/PyUL_NBody \
  --log-root "${QE_RESULT_ROOT}/logs" \
  --gpu-index 0 \
  --run-id "${QE_RUN_ID}" \
  --expected-stage "${QE_STAGE}" \
  --maximum-gpu-stages 1
