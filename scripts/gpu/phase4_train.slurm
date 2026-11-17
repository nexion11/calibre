#!/bin/bash
# Phase 4: one encoder per array task. "all" and "frozen" need 25 encoders (array 0-24), the other variants 5 (0-4).
#   VARIANT=all SEED=0 sbatch --array=0-24 --export=ALL scripts/gpu/phase4_train.slurm
#   uv run python -m calibre.experiments.per_second_finetune jobs --variant all     (lists what each index trains)
#SBATCH --job-name=calibre-p4
#SBATCH --partition=gpu
#SBATCH --account=EDIT_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/phase4_%x_%a_%j.txt
source scripts/gpu/common.sh
uv run python -m calibre.experiments.per_second_finetune train --variant "${VARIANT:-all}" --seed "${SEED:-0}" --job "$SLURM_ARRAY_TASK_ID"
