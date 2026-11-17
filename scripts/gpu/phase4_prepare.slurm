#!/bin/bash
# Phase 4: shrink every frame to 288x160 and write the per-second targets (CPU only, ~10 min).
#   sbatch scripts/gpu/phase4_prepare.slurm
#SBATCH --job-name=calibre-p4prep
#SBATCH --partition=main
#SBATCH --account=EDIT_ME
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/phase4_prepare_%j.txt
source scripts/gpu/common.sh
uv run python -m calibre.experiments.per_second_finetune prepare
