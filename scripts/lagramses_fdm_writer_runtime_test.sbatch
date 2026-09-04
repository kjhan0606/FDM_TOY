#!/bin/bash
#SBATCH --job-name=fdm-writer-rt
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/kjhan/FDM_WRITER_RUNTIME_TEST/slurm-%j.out
#SBATCH --error=/gpfs/kjhan/FDM_WRITER_RUNTIME_TEST/slurm-%j.err

# Run a short, isolated FDM writer integration test.  This script never
# changes the lagRamses checkout and never claims a physical merger result.
set -euo pipefail

# Keep this smoke test bounded even when the executable initializes threaded
# FFTW/OpenMP libraries.  The allocation is four MPI ranks × two CPUs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export FFTW_NUM_THREADS=1

# The tested executable is linked against Intel MPI.  Slurm's PMI2 plugin plus
# the system PMI2 library is required for a real multi-rank launch on syn01;
# without this setting each task can silently fall back to a singleton.
if [[ -z "${I_MPI_PMI_LIBRARY:-}" && -r /usr/lib64/libpmi2.so ]]; then
  export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
fi

repo=${RAMSES_REPO:-/home/kjhan/BACKUP/lagRamses}
executable=${RAMSES_EXECUTABLE:-${repo}/bin/ramses_final3d}
run_base=${FDM_WRITER_TEST_ROOT:-/gpfs/kjhan/FDM_WRITER_RUNTIME_TEST}
run_dir=${run_base}/${SLURM_JOB_ID}

test -x "${executable}"
test -f "${repo}/tests/dynamic_fdm.nml"
mkdir -p "${run_dir}"
cp "${repo}/tests/dynamic_fdm.nml" "${run_dir}/cosmo.nml"

# Only the copied namelist is edited.  The production source and test input
# remain untouched, while the output writer is forced to emit its raw ledger.
if grep -q 'fdm_outer_ledger' "${run_dir}/cosmo.nml"; then
  sed -i 's/fdm_outer_ledger[[:space:]]*=.false./fdm_outer_ledger=.true./' "${run_dir}/cosmo.nml"
else
  sed -i '/fdm_use_hjm/a fdm_outer_ledger=.true.' "${run_dir}/cosmo.nml"
fi
# The tiny dynamic-FDM fixture starts at aexp much below its production
# output target (aout=1).  Request one output on the final smoke step so the
# writer's COMPLETE/sidecar/raw-provenance path is exercised deterministically
# without changing the tracked input or extending the run.
sed -i -E 's/^[[:space:]]*foutput[[:space:]]*=.*/foutput=2/' "${run_dir}/cosmo.nml"
# Full FDM field shards are part of this writer smoke contract.  The current
# HDF5 path intentionally fails closed until it has an equivalent FDM-field
# writer, so exercise the original per-rank backup path here.
sed -i -E "s/^[[:space:]]*outformat[[:space:]]*=.*/outformat='original'/" "${run_dir}/cosmo.nml"
# The current full lagRamses build includes the separate Phase-0 stellar
# production gate.  This writer-only test has no hydro or sinks, so keep that
# unrelated channel out of the smoke path without changing the tracked input.
sed -i -E "s/^[[:space:]]*feedback_mode[[:space:]]*=.*/feedback_mode='legacy'/" "${run_dir}/cosmo.nml"
: > "${run_dir}/jobcontrol.txt"
cp "${executable}" "${run_dir}/ramses_final3d.tested"
sha256sum "${executable}" > "${run_dir}/executable.sha256.before"
sha256sum "${run_dir}/ramses_final3d.tested" > "${run_dir}/executable.sha256.copied"
sha256sum "${repo}/patch/lagRamses/output_amr.kjhan.f90" > "${run_dir}/writer-source.sha256"

cd "${run_dir}"
# A PMI2-enabled Intel-MPI launch is required for nproc to equal the Slurm
# task count.  The post-run check below rejects singleton fallbacks.
srun --kill-on-bad-exit=1 --mpi=pmi2 -n "${SLURM_NTASKS}" \
  -c "${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
  "${run_dir}/ramses_final3d.tested" cosmo.nml 2>&1 | tee run.log

grep -Fq 'Run completed' run.log
test -f output_00001/COMPLETE
mapfile -t sidecars < <(find output_* -maxdepth 1 -type f -name 'dm_run_provenance_*.txt' | sort)
test "${#sidecars[@]}" -eq 1
sidecar=${sidecars[0]}
test -f output_00001/fdm_outer_wave_provenance_00001.txt

nproc_values=$(sed -nE 's/.*Working with nproc = *([0-9]+) for ndim = *3.*/\1/p' run.log)
nproc_count=$(printf '%s\n' "${nproc_values}" | sed '/^$/d' | wc -l)
nproc_unique=$(printf '%s\n' "${nproc_values}" | sed '/^$/d' | sort -u | wc -l)
nproc_value=$(printf '%s\n' "${nproc_values}" | sed '/^$/d' | head -n 1)
test "${nproc_count}" -ge 1
test "${nproc_unique}" -eq 1
test "${nproc_value}" -ge 2
completed_count=$(grep -Ec '^[[:space:]]*Run completed[[:space:]]*$' run.log)
if [[ "${nproc_count}" -eq 1 && "${completed_count}" -eq 1 ]]; then
  : # master-only RAMSES transcript
else
  test "${nproc_count}" -eq "${nproc_value}"
  test "${completed_count}" -eq "${nproc_value}"
fi

printf '%s\n' "${run_dir}/${sidecar#./}" > runtime-sidecar.path
printf '%s\n' "${run_dir}" > runtime-run.path
printf 'writer_runtime_test_status = passed\n'
printf 'runtime_run_directory = %s\n' "${run_dir}"
printf 'runtime_sidecar = %s\n' "${run_dir}/${sidecar#./}"
