#!/bin/bash
# Part C: train the final detector (YOLO11s, 960 px, rare-tool fix) for one surgeon fold per array task.
#   sbatch --array=1-4 scripts/gpu/part_c_train.slurm      (fold 0 is copied from the PC; use 0-4 to retrain it)
#SBATCH --job-name=calibre-partc
#SBATCH --partition=gpu
#SBATCH --account=EDIT_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/partc_fold%a_%j.txt
source scripts/gpu/common.sh
uv run python -m calibre.experiments.final_detectors train "$SLURM_ARRAY_TASK_ID"
