#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s RUN_DIRECTORY\n' "$0" >&2
  exit 2
fi
if [[ $(hostname -s) != syntax ]]; then
  printf 'This bounded n384 wave-response runner is restricted to syntax.\n' >&2
  exit 2
fi

repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
result_root=/gpfs/kjhan/FDM_TOY_RESULTS
log_root=${result_root}/logs
run=$(readlink -f "$1")
case "${run}" in
  "${result_root}"/*) ;;
  *)
    printf 'Run must lie below %s: %s\n' "${result_root}" "${run}" >&2
    exit 2
    ;;
esac

mkdir -p "${log_root}"
exec 9>"${log_root}/fdm_n384_wave_response_safe.lock"
if ! flock -n 9; then
  printf 'Another bounded n384 wave-response process holds the lock.\n' >&2
  exit 3
fi

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# A 384-cubed response sample is expected to use about 16 GiB from the
# measured 512-cubed peak. The 32 GiB address-space limit leaves FFT workspace
# while preventing an unbounded allocation.
ulimit -v 33554432

cd "${repository}"
summary=${run}/torch_run_summary.json
metadata=${run}/fdm_adapter_metadata.json
python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "complete"' \
  "${summary}"
resolution=$(python -c \
  'import json,sys; print(int(json.load(open(sys.argv[1]))["resolution"]))' \
  "${metadata}")
if [[ ${resolution} -ne 384 ]]; then
  printf 'Expected a 384-cubed run, found %s: %s\n' \
    "${resolution}" "${run}" >&2
  exit 2
fi
sample_count=$(python -c \
  'import json,sys; print(int(json.load(open(sys.argv[1]))["saved_3d_states"]))' \
  "${metadata}")
if [[ ! ${sample_count} =~ ^[1-9][0-9]*$ ]]; then
  printf 'Invalid saved_3d_states for %s: %s\n' \
    "${run}" "${sample_count}" >&2
  exit 2
fi

wave_response_complete() {
  [[ -s ${run}/wave_response_summary.json ]] \
    && [[ -s ${run}/wave_response_timeseries.csv ]] \
    && [[ -s ${run}/wave_radial_profiles.csv ]]
}

for ((invocation = 1; invocation <= sample_count; invocation++))
do
  if wave_response_complete; then
    break
  fi
  printf '%s | %s | wave-response invocation %d/%d\n' \
    "$(date '+%F %T %Z')" "${run}" "${invocation}" "${sample_count}"
  python scripts/analyze_pyul_wave_response.py "${run}" \
    --resume --max-new-samples 1
done

if ! wave_response_complete; then
  printf 'Wave response did not complete after %s invocations: %s\n' \
    "${sample_count}" "${run}" >&2
  exit 4
fi
rm -f \
  "${run}/wave_response_timeseries.partial.csv" \
  "${run}/wave_radial_profiles.partial.csv"
printf '%s | %s | wave response complete\n' \
  "$(date '+%F %T %Z')" "${run}"
