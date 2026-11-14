# CALIBRE experiments: building on SOCAL

All numbers come from `results/` and can be rerun with `uv run python -m calibre.experiments.run_experiments human` (or `auto`).

## How everything is tested

- 143 SOCAL trials with video, tool labels and outcomes, from 73 surgeons.
- **Whole surgeons are held out.** A surgeon is never in both training and testing, and the 5-fold split is repeated 10 times.
- 95% ranges come from resampling surgeons. "Gap vs A" is a paired test on the same surgeons. A range that doesn't cross 0 means the gain is real.
- Severe bleed means more than 500 mL, matching SOCAL. Failure means the bleed was not controlled within 5 minutes.
- Everything uses only the first 60 seconds of video unless a time is stated.

## What SOCAL reported

| | SOCAL result | How it was tested |
|---|---|---|
| Tool detection (RetinaNet) | mAP50 0.67, by tool 0.91 / 0.77 / 0.52 / 0.50 / 0.25 (suction / grasper / cottonoid / string / muscle) | 14 test trials, split not described |
| SOCALNet outcome | 85% correct (17/20) on success, blood loss R² 0.74 | 20 random test videos, not split by surgeon, needs human tool labels as input |
| Expert surgeons | 69% correct, caught 47.5% of bleeds over 500 mL | same 20 videos |

## Results (human tool labels, held-out surgeons)

| Method | Severe-bleed AUC [95%] | Failure AUC [95%] | False alarms at 75% caught | Severe gap vs A [95%] | Failure gap vs A [95%] |
|---|---|---|---|---|---|
| A. Original CALIBRE (7 features) | 0.757 [0.67–0.84] | 0.636 [0.58–0.77] | 34% | | |
| B. + 10 new bleed-handling features | 0.781 [0.70–0.86] | 0.652 [0.59–0.77] | 27% | +0.023 [−0.01, +0.05] | +0.006 [−0.06, +0.07] |
| C. B with survival (rate × time) model | 0.781 [0.70–0.86] | 0.693 [0.60–0.78] | 28% | +0.021 [−0.01, +0.05] | +0.016 [−0.04, +0.07] |
| **D. B + surgeon's previous attempt** | **0.813 [0.74–0.89]** | **0.792 [0.71–0.87]** | **25%** | **+0.056 [+0.01, +0.10]** | **+0.117 [+0.04, +0.19]** |
| E. B + all surgeon information | 0.819 [0.75–0.89] | 0.745 [0.68–0.86] | 27% | +0.064 [+0.01, +0.12] | +0.094 [−0.01, +0.19] |
| F. DINOv2 frame features only | 0.786 [0.71–0.86] | 0.467 | 28% | +0.031 [−0.04, +0.10] | worse |
| **G. DINOv2 + B (video only)** | **0.810 [0.73–0.89]** | 0.553 | **23%** | **+0.055 [+0.01, +0.10]** | −0.025 [−0.09, +0.04] |
| H. SOCALNet-style ResNet + LSTM (our re-build) | 0.740 [0.66–0.83] | 0.638 [0.58–0.74] | 39% | −0.010 [−0.10, +0.08] | −0.017 [−0.10, +0.07] |

**Honest "best of" score.** Picking the best of methods A–E inside each training set (nested cross-validation) gives severe AUC **0.807** and failure AUC **0.792**. So the gains are not from cherry-picking.

## Main findings

1. **Individualized prediction works.** Adding how the surgeon's previous attempt went (D) is the only change that clearly improves both severe-bleed and failure prediction. A surgeon's first and second attempts are strongly linked (Spearman 0.52 for blood loss, p < 0.00001). SOCALNet's paper lists individualized models as an open limitation.
2. **Whole-frame features help severity.** Adding self-supervised DINOv2 features to our handling features (G) clearly improves severe-bleed AUC to 0.81 and cuts false alarms to 23%, using video only. They do not help predict failure.
3. **New handling features help a little.** These cover red-out time, suction clearing, time to first cottonoid, and two-handed work. They raise severe AUC from 0.76 to 0.78, but the gain is not significant on its own.
4. **A SOCALNet-style deep model is not better under fair testing.** Our ResNet + LSTM rebuild, given the same tool-presence input SOCALNet used, scored 0.74 / 0.64. That is no better than the simple model, though it has the best blood-loss R² (0.43). The architecture matches their paper's description, but we do not have their trained weights.
5. **20-video tests swing a lot.** Testing the same model (B) on 500 random sets of 20 videos gave anywhere from 50% to 85% correct on success (5th–95th percentile). A score on 20 videos cannot be compared directly with ours.
6. **Moving to new course sites costs about 0.03–0.05 AUC.** Trained on 6 SOCAL sites and tested on the 7th, the individualized model still reaches 0.76 severe and 0.77 failure AUC.
7. **Early warning.** The individualized model already reaches 0.71 severe and 0.76 failure AUC at 15 seconds, and peaks around 45–60 seconds.

| Seconds watched | A severe / failure | B severe / failure | D severe / failure |
|---|---|---|---|
| 15 | 0.68 / 0.67 | 0.66 / 0.64 | 0.71 / 0.76 |
| 30 | 0.68 / 0.64 | 0.70 / 0.61 | 0.75 / 0.76 |
| 45 | 0.71 / 0.67 | 0.74 / 0.67 | 0.79 / 0.81 |
| 60 | 0.75 / 0.64 | 0.77 / 0.66 | 0.80 / 0.78 |
| 90 | 0.71 / 0.63 | 0.73 / 0.64 | 0.77 / 0.78 |
| 120 | 0.64 / 0.56 | 0.72 / 0.54 | 0.72 / 0.71 |

## Our models scored the way SOCALNet was scored

This uses 20 random test videos and trains on the rest. The split is by video, so the same surgeon can be in both training and testing. It is repeated 200 times (40 for H), and each cell shows the median with the 5th–95th percentile range.

| Method | Success accuracy | Blood loss R² | RMSE (mL) | Bleeds > 500 mL caught | Splits reaching 85% accuracy |
|---|---|---|---|---|---|
| SOCALNet (published, one split) | 85% | 0.74 | 295 | 80% | |
| A. Original CALIBRE | 65% [50–85] | 0.06 [−0.56, 0.43] | 356 [233–494] | 38% | 6% |
| B. + handling features | 70% [50–80] | 0.13 [−0.45, 0.50] | 349 [227–480] | 33% | 4% |
| D. B + previous attempt | 75% [60–90] | 0.24 [−0.23, 0.55] | 320 [217–450] | 50% | 23% |
| G. DINOv2 + B | 65% [50–80] | 0.25 [−0.32, 0.56] | 318 [222–438] | 43% | 4% |
| H. SOCALNet-style re-build | 68% [50–80] | 0.12 [−0.32, 0.44] | 343 [259–480] | 50% | 0% |

What this shows:
- **None of our models, including our SOCALNet re-build, reach SOCALNet's published R² of 0.74 on any of the 200 splits.** The 85% accuracy is reached on 23% of splits for model D, and rarely for the others.
- SOCALNet's published RMSE of 295 mL falls inside the range our best models reach (D and G, medians about 320 mL).
- So the easier test protocol does not fully explain their published numbers. Possible reasons include differences in their exact model and training (we only have the paper's description), their specific 20 test videos, or how R² was calculated. **We cannot claim to beat the published SOCALNet. Comparing on their exact test videos with their trained model is the next step.**
- A single 20-video test swings widely. For the same model, accuracy ranges from 50% to 90% and R² from below 0 to 0.55. This is why we report surgeon-held-out results on all 143 trials with 95% ranges.

## Fully automatic (no human tool labels)

Five tool detectors are being trained, one per surgeon fold. Each labels only the surgeons it never saw. Rerunning everything on those labels (`run_experiments auto`) answers SOCALNet's other stated limitation, that it needs human tool annotations. *Results pending.*

## Limitations

- These are cadaver simulation trials only, with 143 trials from 73 surgeons, so the 95% ranges are wide.
- The previous-attempt feature only exists for second attempts (70 of 143 trials). It fits a training course, where a trainee's last attempt is known.
- The blood measurement is a hand-tuned color rule.
- The SOCALNet comparison uses our rebuild, not their trained model.
