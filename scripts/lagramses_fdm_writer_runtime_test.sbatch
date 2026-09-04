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
: > "${run_dir}/jobcontrol.txt"
cp "${executable}" "${run_dir}/ramses_final3d.tested"
sha256sum "${executable}" > "${run_dir}/executable.sha256.before"
sha256sum "${run_dir}/ramses_final3d.tested" > "${run_dir}/executable.sha256.copied"
sha256sum "${repo}/patch/lagRamses/output_amr.kjhan.f90" > "${run_dir}/writer-source.sha256"

cd "${run_dir}"
# This cluster exposes PMIx v4; PMI2 causes a handshake hang on syn01.
srun --kill-on-bad-exit=1 --mpi=pmix_v4 -n "${SLURM_NTASKS}" \
  -c "${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
  "${run_dir}/ramses_final3d.tested" cosmo.nml 2>&1 | tee run.log

grep -Fq 'Run completed' run.log
test -f output_00001/COMPLETE
sidecar=$(find output_* -maxdepth 3 -type f -name 'dm_run_provenance_*.txt' | sort | tail -n 1)
test -n "${sidecar}"
printf '%s\n' "${run_dir}/${sidecar#./}" > runtime-sidecar.path
printf '%s\n' "${run_dir}" > runtime-run.path
printf 'writer_runtime_test_status = passed\n'
printf 'runtime_run_directory = %s\n' "${run_dir}"
printf 'runtime_sidecar = %s\n' "${run_dir}/${sidecar#./}"
