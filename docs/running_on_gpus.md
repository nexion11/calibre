# Running the GPU stages (Part C and Phase 4)

Everything here is written and smoke-tested on the Mac in `--quick` mode. The real runs need a GPU. They work the same on the home PC, a rented cloud GPU, or a cluster such as USC CARC. Every step can be stopped and started again; finished pieces are skipped.

## What runs where

| Step | Command | Needs | Rough time on an RTX 4060 | Rough time on one A100 |
|---|---|---|---|---|
| Part C: final detector, folds 1–4 | `final_detectors train 1 2 3 4` | GPU | ~8–9 h per fold | ~2–3 h per fold (folds run in parallel on a cluster) |
| Part C: label unseen surgeons + score | `final_detectors label`, then `score` | GPU | ~1 h | ~20 min |
| Phase 4: shrink frames, targets | `per_second_finetune prepare` | CPU | ~10 min | ~10 min |
| Phase 4: main method, 3 seeds | `per_second_finetune train --variant all --seed 0/1/2` | GPU | 25 encoders per seed, ~15–25 min each | ~5–10 min each (array jobs run in parallel) |
| Phase 4: "frozen" baseline | `--variant frozen` | GPU | 25 encoders, faster (backbone frozen) | |
| Phase 4: one task removed at a time | `--variant no_time / no_tools / no_blood / no_muscle` | GPU | 5–25 encoders each | |

The times are estimates, not measurements. The first real encoder prints its own time in its `.json` file.

Order of importance if GPU time is short: Part C first, then Phase 4 `all` seed 0, then `frozen`, then `no_time`, then the extra seeds and the other ablations.

## On a Slurm cluster (USC CARC)

1. Copy the repo and the frames. The CSVs are in git; the frames (3 GB) are not.
   - `git clone https://github.com/nexion11/calibre.git`
   - copy `data/frames/JPEGImages` from the Mac (31,443 files), and the PC's fold-0 detector to `models/detector_search/s_960_rare/weights/best.pt` (otherwise fold 0 is retrained too).
2. `bash scripts/gpu/setup.sh` on a login node.
3. Put the account name in `scripts/gpu/common.sh` and in the `#SBATCH --account` line of each `.slurm` file. Check the partition and GPU names against CARC's own docs.
4. Submit from the repo folder:
   - `sbatch --array=1-4 scripts/gpu/part_c_train.slurm`
   - `sbatch --dependency=afterok:<that job id> scripts/gpu/part_c_label.slurm`
   - `sbatch scripts/gpu/phase4_prepare.slurm`
   - `VARIANT=all SEED=0 sbatch --array=0-24 --export=ALL scripts/gpu/phase4_train.slurm` (then seeds 1 and 2, `frozen` 0-24, the `no_*` variants: `no_time` 0-4, the others 0-24)

## On one GPU machine (home PC or a rented GPU)

`bash scripts/gpu/run_on_one_gpu.sh` runs everything in order. On Windows, run the same `uv run python -m ...` lines from that file one by one in PowerShell.

## What to bring back to the Mac

- `data/auto_boxes_v2.csv`, `data/auto_boxes_v2_low/`, `results/final_detectors/`
- `data/finetune/` (the encoders' per-second outputs, about 40 MB per encoder; the network weights themselves are not saved)

## Then, on the Mac

1. `uv run python -m calibre.experiments.track_cleanup tune --v2` (label set `clean2`)
2. `uv run python -m calibre.experiments.clean_vs_auto clean clean2` (do the new labels improve outcome prediction?)
3. `uv run python -m calibre.experiments.run_experiments clean2` and `CALIBRE_LABELS=clean2 uv run python -m calibre.experiments.phase1`
4. `CALIBRE_LABELS=clean2 uv run python -m calibre.experiments.phase3`
5. `CALIBRE_LABELS=clean2 uv run python -m calibre.experiments.per_second_finetune evaluate`
