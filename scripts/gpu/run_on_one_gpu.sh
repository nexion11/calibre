#!/bin/bash
# Everything in order on ONE machine with a GPU (the home PC, a rented cloud GPU, or an interactive CARC session).
# Safe to stop and start again: finished pieces are skipped.
#   bash scripts/gpu/run_on_one_gpu.sh            (Part C, then Phase 4 main runs, then the ablations)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
uv run python -m calibre.experiments.final_detectors train 0 1 2 3 4
uv run python -m calibre.experiments.final_detectors label
uv run python -m calibre.experiments.final_detectors score
uv run python -m calibre.experiments.per_second_finetune prepare
for seed in 0 1 2; do uv run python -m calibre.experiments.per_second_finetune train --variant all --seed "$seed"; done
uv run python -m calibre.experiments.per_second_finetune train --variant frozen --seed 0
for v in no_time no_tools no_blood no_muscle; do uv run python -m calibre.experiments.per_second_finetune train --variant "$v" --seed 0; done
