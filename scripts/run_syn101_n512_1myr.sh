#!/bin/bash
set -euo pipefail

if [[ $(hostname -s) != syn101 ]]; then
  printf 'This manual GPU runner is restricted to syn101.\n' >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0
exec bash /home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY/scripts/run_lageunha_n512_1myr.sh
