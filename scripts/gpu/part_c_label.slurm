#!/bin/bash
# Part C: label every fold's unseen surgeons, then score the 5-fold detectors. Run after all training jobs:
#   sbatch --dependency=afterok:<train job id> scripts/gpu/part_c_label.slurm
#SBATCH --job-name=calibre-label
#SBATCH --partition=gpu
#SBATCH --account=EDIT_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/partc_label_%j.txt
source scripts/gpu/common.sh
uv run python -m calibre.experiments.final_detectors label
uv run python -m calibre.experiments.final_detectors score
