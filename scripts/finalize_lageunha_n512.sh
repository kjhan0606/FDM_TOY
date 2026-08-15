#!/bin/bash
set -euo pipefail

repository=/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY
result_root=/gpfs/kjhan/FDM_TOY_RESULTS
run=${result_root}/torch_long_term/koo_n512_cuda_1myr_s2048
n256=${result_root}/torch_convergence/koo_n256_cuda_1myr_s2048
n512_dt05=${result_root}/torch_convergence/koo_n512_cuda_dt05_0p1myr_s512
comparison_root=${result_root}/torch_convergence
movie_directory=${result_root}/movies
state_directory=${run}/Finalization
log=${result_root}/logs/fdm_n512_monitor.log
dry_run=false

if [[ ${1:-} == "--dry-run" ]]; then
  dry_run=true
elif [[ $# -ne 0 ]]; then
  printf 'usage: %s [--dry-run]\n' "$0" >&2
  exit 2
fi

if [[ "${dry_run}" == false ]]; then
  mkdir -p "${state_directory}" "${movie_directory}" "$(dirname "${log}")"
fi
cd "${repository}"

record() {
  printf '%s | %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${log}"
}

run_step() {
  local name=$1
  shift
  local marker=${state_directory}/${name}.complete
  if [[ -f "${marker}" ]]; then
    record "tripwire step ${name} already complete; skipping"
    return
  fi
  if [[ "${dry_run}" == true ]]; then
    printf 'STEP %s:' "${name}"
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  record "tripwire step ${name} starting"
  "$@" >> "${log}" 2>&1
  touch "${marker}"
  record "tripwire step ${name} complete"
}

if [[ "${dry_run}" == false ]]; then
  summary=${run}/torch_run_summary.json
  if [[ ! -f "${summary}" ]]; then
    record "tripwire finalizer refused: evolution summary is missing"
    exit 3
  fi
  python -c \
    'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "complete"' \
    "${summary}"
fi

run_step provenance \
  python scripts/snapshot_torch_provenance.py "${run}"
run_step conservation \
  python scripts/analyze_pyul_wave_run.py "${run}"
run_step orbit_averaged_exchange \
  python scripts/analyze_pyul_secular_exchange.py "${run}"
run_step line_density \
  python scripts/analyze_pyul_line_density.py "${run}"
run_step wave_response \
  python scripts/analyze_pyul_wave_response.py "${run}"
run_step wave_exchange_table \
  python scripts/build_wave_exchange_table.py "${run}" \
    --output "${run}/wave_exchange.csv"
run_step spatial_convergence \
  python scripts/summarize_pyul_convergence.py \
    n512="${run}" \
    n256="${n256}" \
    --output "${comparison_root}/koo_spatial_convergence_final.json"
run_step temporal_convergence \
  python scripts/summarize_pyul_convergence.py \
    dt1="${run}" \
    dt05="${n512_dt05}" \
    --output "${comparison_root}/koo_temporal_convergence_0p1myr_final.json"
run_step joint_convergence \
  python scripts/summarize_pyul_convergence.py \
    n512_dt1="${run}" \
    n512_dt05="${n512_dt05}" \
    n256_dt1="${n256}" \
    --output "${comparison_root}/koo_joint_convergence_0p1myr_final.json"
run_step movie \
  python scripts/render_pyul_movie.py "${run}" \
    --output "${movie_directory}/koo_n512_cuda_1myr.mp4" \
    --poster "${movie_directory}/koo_n512_cuda_1myr_poster.png" \
    --zoom-width-pc 1.44

if [[ "${dry_run}" == true ]]; then
  exit 0
fi

required_outputs=(
  "${run}/torch_solver_provenance/manifest.json"
  "${run}/conservation_summary.json"
  "${run}/orbit_averaged_exchange_summary.json"
  "${run}/line_density_summary.json"
  "${run}/wave_response_summary.json"
  "${run}/wave_exchange.summary.json"
  "${comparison_root}/koo_spatial_convergence_final.json"
  "${comparison_root}/koo_temporal_convergence_0p1myr_final.json"
  "${comparison_root}/koo_joint_convergence_0p1myr_final.json"
  "${movie_directory}/koo_n512_cuda_1myr.mp4"
  "${movie_directory}/koo_n512_cuda_1myr.mp4.json"
  "${movie_directory}/koo_n512_cuda_1myr_poster.png"
)
for output in "${required_outputs[@]}"; do
  if [[ ! -s "${output}" ]]; then
    record "tripwire verification failed: missing or empty ${output}"
    exit 4
  fi
done

touch "${state_directory}/finalization.complete"
record "tripwire finalization and output verification complete"
