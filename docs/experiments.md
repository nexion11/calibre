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
| SOCALNet outcome | 85% correct (17/20) on success, blood loss RMSE 295 mL, R² 0.74, caught 8 of 10 bleeds over 500 mL | 20 test videos picked to balance outcomes (11 controlled, 9 failed), not split by surgeon, needs human tool labels as input |
| Expert surgeons | 69% correct (55/80), RMSE 351 mL, caught 47.5% of bleeds over 500 mL | same 20 videos |

SOCALNet's numbers and test design are according to PubMed (Pangal et al., Sci Rep 2022, [DOI](https://doi.org/10.1038/s41598-022-11549-2)). The paper reports blood loss with RMSE and Pearson's correlation, so its R² is a squared correlation and not 1 − error/variance.

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
- So the easier test protocol does not fully explain their published numbers. Phase 1 below looks at two reasons, how R² was calculated and how the 20 test videos were picked. **We cannot claim to beat the published SOCALNet. Comparing on their exact test videos with their trained model is the next step.**
- A single 20-video test swings widely. For the same model, accuracy ranges from 50% to 90% and R² from below 0 to 0.55. This is why we report surgeon-held-out results on all 143 trials with 95% ranges.

## Phase 1: a fairer comparison with SOCALNet and a better blood-loss estimate

Rerun with `uv run python -m calibre.experiments.phase1`. Numbers are in `results/human_phase1_*.csv`. The plan and the success bars are in `docs/improvement_plan.md`.

### 1.1 Why SOCALNet's R² of 0.74 cannot be compared with ours

R² (as 1 − error/variance) and RMSE together fix how spread out the test videos' blood loss must be. An R² of 0.74 at an RMSE of 295 mL needs a spread (standard deviation) of 579 mL. We drew 10,000 test sets of 20 videos from our 143 trials, both at random and the way SOCALNet's was built (11 controlled, 9 failed).

| 20-video test set | Spread of blood loss, 5th / median / 95th percentile | Sets with a spread of 579 mL or more | R² that an RMSE of 295 mL gives on the median set |
|---|---|---|---|
| Random | 250 / 371 / 496 mL | 0.1% | 0.37 |
| 11 controlled + 9 failed | 277 / 395 / 509 mL | 0.2% | 0.44 |

- Balancing the test set by outcome does not explain it. It raises the spread only a little.
- A squared correlation is always at least as high as R², because it ignores predictions that run too high or too low. For our best blood-loss model the median squared correlation on 20 videos is 0.43 and the R² is 0.37. 1% of test sets reach a squared correlation of 0.74.
- **RMSE is the number that can be compared fairly.** The R² gap is a question for the lab, not evidence that either result is wrong.

### 1.2 and 1.3 What the blood-loss model learns

Our model learned log(blood loss), which predicts a typical value and runs low on big bleeds. Same features as D, surgeons held out:

| The model learns | RMSE (mL) | R² in mL | Severe-bleed AUC | RMSE gap vs log [95%] |
|---|---|---|---|---|
| log (what we had) | 338 | 0.24 | 0.813 | |
| log, corrected to the right average | 342 | 0.22 | 0.813 | +1 [−20, +28] |
| **square root** | **322** | **0.30** | 0.803 | **−14 [−25, −1]** |
| square root, corrected | 320 | 0.31 | 0.803 | −17 [−33, +4] |
| plain mL | 324 | 0.30 | 0.789 | −13 [−33, +13] |
| gamma model | 325 | 0.29 | 0.795 | −14 [−28, 0] |
| Huber loss | 331 | 0.27 | 0.796 | −6 [−21, +11] |
| two-part (controlled vs. failed) | 333 | 0.26 | 0.792 | −6 [−24, +14] |

- **The square root is the only change whose gain is clear**, and it holds on the video-only features too (B: 360 → 335 mL, gap −23 [−34, −9]). It costs nothing real in ranking.
- The two-part model (1.3) did not beat it. That is a negative result.

### New models, surgeons held out

All blood-loss models below learn the square root unless marked. "Prior from all 365 trials" means the surgeon part of the model is fitted on every trial in `socal_outcomes.csv`, including the 222 with no video, and never on the test surgeons.

| Method | Severe-bleed AUC [95%] | Failure AUC [95%] | False alarms at 75% caught | RMSE (mL) | R² in mL |
|---|---|---|---|---|---|
| B. Video only (log) | 0.781 [0.70–0.86] | 0.652 [0.59–0.77] | 27% | 360 | 0.13 |
| D. B + previous attempt (log) | 0.813 [0.74–0.89] | 0.792 [0.71–0.87] | 25% | 338 | 0.24 |
| D1. D, square root | 0.803 [0.73–0.87] | 0.792 | 28% | 322 | 0.30 |
| D2. D, two-part | 0.792 | 0.792 | 33% | 333 | 0.26 |
| P1. B + previous-attempt prior from all 365 trials | 0.801 | 0.706 [0.66–0.82] | 26% | 322 | 0.31 |
| P2. D + that prior | 0.803 | 0.792 [0.72–0.88] | 27% | 323 | 0.30 |
| P3. B + all-surgeon-information prior from all 365 trials | 0.790 | 0.723 | 31% | 323 | 0.30 |
| E1. B + all surgeon information (143 trials) | 0.804 | 0.745 | 31% | 332 | 0.26 |
| G1. DINOv2 + B (video only) | 0.803 | 0.553 | 25% | 318 | 0.33 |
| **DG. DINOv2 + D** | **0.818 [0.75–0.89]** | 0.738 [0.69–0.86] | **25%** | **312** | **0.35** |
| PG. DINOv2 + P1 | 0.812 | 0.635 | 25% | 310 | 0.36 |
| H. SOCALNet-style re-build | 0.740 [0.66–0.83] | 0.638 [0.58–0.74] | 38% | 349 | 0.19 |

- **Best blood-loss estimate: DG.** Against D it lowers RMSE by 24 mL [−39, −8] with no real change in severe AUC (+0.008 [−0.02, +0.04]) or failure AUC (−0.018 [−0.05, +0.01]).
- **Best failure prediction is still D's classifier** (0.792). DINOv2 features do not help failure, as before.
- So the model we carry forward estimates blood loss with DG and failure risk with D.

### 1.4 Turning the numbers into yes/no calls

SOCALNet is scored on yes/no calls, and we caught few big bleeds because we cut a low-running mL estimate at 500. "Tuned cutoff" picks the cutoff inside the training surgeons only.

| Call | Rule | Accuracy | Caught | False alarms |
|---|---|---|---|---|
| Failure (D) | risk above 0.5 | 76% [70–83] | 53% | 12% |
| Failure (D) | tuned cutoff | 75% | 54% | 13% |
| Over 500 mL (D, log) | estimate above 500 mL | 73% | 45% [33–56] | 12% |
| Over 500 mL (D, log) | tuned cutoff | 75% [68–81] | 72% [62–81] | 24% [15–33] |
| Over 500 mL (D1, square root) | estimate above 500 mL | 75% | 58% | 15% |
| Over 500 mL (D1, square root) | tuned cutoff | 74% | 70% | 24% |
| Over 500 mL (D) | separate classifier, tuned cutoff | 70% | 68% | 30% |

- A tuned cutoff catches 70–72% of bleeds over 500 mL (up from 45%) at about the same accuracy. The price is twice the false alarms.
- A separate over-500 classifier is no better than cutting the mL estimate.
- Accuracy stays at 75–76%. The bar of 78% was not met.

### 1.5 Using all 365 SOCAL trials

`socal_outcomes.csv` holds 365 trials from 177 surgeons. These tests use outcomes and surgeon information only, with no video.

| Test | Trials | Severe-bleed AUC [95%] | Failure AUC [95%] | Spearman | RMSE (mL) |
|---|---|---|---|---|---|
| All trials, previous attempt only | 365 | 0.706 | 0.543 | 0.45 | 326 |
| All trials, all surgeon information | 365 | 0.741 [0.69–0.80] | 0.700 | 0.52 | 320 |
| Second and third attempts, previous attempt only | 188 | 0.837 [0.76–0.91] | see note | 0.59 | 283 |
| **Trained on surgeons with video, tested on 103 surgeons with no video** | 115 | **0.862 [0.77–0.94]** | **0.722 [0.57–0.85]** | 0.56 [0.42–0.69] | 241 |

- **The previous-attempt finding replicates on surgeons we never used before.** A model fitted on the 73 later attempts by surgeons with video predicts the later attempts of 103 other surgeons at 0.86 severe AUC.
- Note on the third row: only 21 of those 188 trials failed, so the failure classifier falls back to a near-constant guess whose level changes from fold to fold. Its failure AUC (0.42) is an artifact of that and says nothing either way.
- **Adding this prior to the video model did not raise failure AUC** (P2 0.792 vs D 0.792, gap +0.005 [−0.004, +0.014]). It lowers RMSE by 13 mL [−25, −1]. The bar of 0.82 was not met.

### 1.6 Our models against the SOCALNet-style re-build, paired

| Ours vs H | Severe-bleed AUC gap [95%] | Failure AUC gap [95%] | RMSE gap (mL) [95%] |
|---|---|---|---|
| D | +0.066 [−0.02, +0.15] | **+0.134 [+0.03, +0.23]** | −2 [−32, +23] |
| P2 | | **+0.138 [+0.04, +0.23]** | |
| **DG** | **+0.074 [+0.01, +0.15]** | **+0.115 [+0.02, +0.20]** | **−27 [−50, −4]** |

DG is clearly better than the re-build on all three scores when both are tested on the same held-out surgeons. H is our re-build from the paper's description, not their trained model.

### Scored SOCALNet's way, with both kinds of test set

200 test sets of 20 videos (40 for H), split by video. Each cell is the median with the 5th–95th percentile; "caught" is the average.

**Random 20 videos**

| Method | Success accuracy | RMSE (mL) | R² | Squared correlation | Bleeds > 500 mL caught | Sets with RMSE ≤ 295 |
|---|---|---|---|---|---|---|
| SOCALNet (published, one balanced set) | 85% | 295 | | 0.74 | 80% | |
| B. Video only (log) | 70% [50–80] | 349 [227–480] | 0.13 | 0.31 | 36% | 28% |
| D. B + previous attempt (log) | 75% [60–90] | 320 [217–450] | 0.24 | 0.38 | 47% | 38% |
| D1. D, square root | 75% [60–90] | 307 [202–443] | 0.29 | 0.37 | 58% (69% tuned) | 45% |
| DG. DINOv2 + D | 70% [55–90] | 299 [201–412] | 0.37 | 0.43 | 58% | 48% |
| **Carried forward (DG blood loss, D success)** | **75% [60–90]** | **299 [201–412]** | **0.37** | **0.43** | **58%** | **48%** |
| H. SOCALNet-style re-build | 68% [50–80] | 343 [259–480] | 0.12 | 0.28 | 46% | 10% |

**11 controlled + 9 failed, as SOCALNet's test set was built**

| Method | Success accuracy | RMSE (mL) | R² | Squared correlation | Bleeds > 500 mL caught | Sets with RMSE ≤ 295 |
|---|---|---|---|---|---|---|
| SOCALNet (published) | 85% | 295 | | 0.74 | 80% | |
| B. Video only (log) | 60% [50–75] | 378 [259–536] | 0.04 | 0.28 | 36% | 15% |
| D. B + previous attempt (log) | 70% [55–85] | 357 [236–500] | 0.17 | 0.37 | 46% | 22% |
| D1. D, square root | 70% [55–85] | 338 [219–476] | 0.27 | 0.39 | 58% (70% tuned) | 27% |
| DG. DINOv2 + D | 65% [55–80] | 328 [217–448] | 0.33 | 0.45 | 57% | 32% |
| **Carried forward (DG blood loss, D success)** | **70% [55–85]** | **328 [217–448]** | **0.33** | **0.45** | **57%** | **32%** |
| H. SOCALNet-style re-build | 60% [50–70] | 362 [231–447] | 0.19 | 0.36 | 48% | 25% |

- **On random test sets our blood-loss error now matches SOCALNet's** (299 vs 295 mL, down from 320), and about half of all test sets come in at or under 295.
- **On test sets balanced like theirs we are still behind** (328 mL, 70% accuracy). A balanced set has more failed trials (45% instead of 34%), and every one of our models still runs low on those (about 90 mL low on average). Their 85% accuracy is reached on 10% of balanced sets.
- Our re-build of their design scores worse than our models under both designs, so the gap to the published numbers is not explained by test design alone.

### Phase 1 scorecard

| # | Bar | Result |
|---|---|---|
| 1.1 | Explain the R² gap with numbers | Done. RMSE is the comparable score |
| 1.2 | RMSE under their protocol below 310 mL | **Met on random test sets (299)**, not on balanced ones (328) |
| 1.3 | Two-part model beats the best of 1.2 | Not met |
| 1.4 | Over-500 caught ≥ 70% and accuracy ≥ 78% | Caught met with a tuned cutoff (70–72%), accuracy not (75–76%) |
| 1.5 | Failure AUC ≥ 0.82 | Not met (0.792). The 365-trial data did confirm the previous-attempt finding on 103 new surgeons |
| 1.6 | Gap vs H excludes 0 | **Met for DG on all three scores** |

## Fully automatic (no human tool labels)

SOCALNet's paper lists its need for human tool annotations as a limitation. Here no human label is used at test time. Five YOLO11n tool detectors were trained, one per surgeon fold, each on the other four folds. Every surgeon's video is then labeled only by the detector that never saw that surgeon, and every experiment above is rerun on those labels (`run_experiments auto`, and `CALIBRE_LABELS=auto` for phase 1).

### The detectors, scored on surgeons they never saw

Each fold's detector is scored on every hand-labeled frame of its 15 unseen surgeons, so all 75 surgeons and 31,443 frames are scored once (`uv run python -m calibre.experiments.detector_check`, numbers in `results/auto_detector_unseen_surgeons.csv`).

| Fold | Frames | mAP50 | Suction | Grasper | Cottonoid | String | Muscle |
|---|---|---|---|---|---|---|---|
| 0 | 6,062 | 0.797 | 0.954 | 0.917 | 0.694 | 0.812 | 0.610 |
| 1 | 6,201 | 0.784 | 0.946 | 0.911 | 0.719 | 0.848 | 0.497 |
| 2 | 6,239 | 0.725 | 0.900 | 0.876 | 0.624 | 0.731 | 0.496 |
| 3 | 6,649 | 0.794 | 0.966 | 0.947 | 0.728 | 0.836 | 0.493 |
| 4 | 6,292 | 0.775 | 0.955 | 0.905 | 0.632 | 0.834 | 0.548 |
| **Average of 5 folds** | 31,443 | **0.775** | **0.944** | **0.911** | **0.679** | **0.812** | **0.529** |
| SOCAL RetinaNet (published) | 14 test trials | 0.67 | 0.91 | 0.77 | 0.52 | 0.50 | 0.25 |

- **Every fold beats the published 0.67, and every tool is higher on average.** The weakest fold is 0.725. Muscle is still the hardest tool (0.53) but is twice the published score.
- This is a stricter test than the 0.75 we reported before, which came from one split. The plan's bar for this check (2.1, at least 0.72) is met.
- The published number is from a different split that the paper does not describe, so this is not yet a same-data comparison. Training their RetinaNet on our folds (plan item 2.4) would make it one.
- These detectors trained for 20 rounds on the smallest YOLO11 model. Phase 2 of the plan tests larger models and longer training.

### Outcome prediction with automatic labels

Same 143 trials, 73 surgeons and tests as above. "Human" uses the hand-drawn tool boxes. "Auto" uses only boxes from the detector that never saw that surgeon. Ranges are 95%, from resampling surgeons.

| Method | Severe AUC, human | Severe AUC, auto [95%] | Failure AUC, human | Failure AUC, auto [95%] | RMSE mL, auto |
|---|---|---|---|---|---|
| A. Original CALIBRE | 0.757 | 0.744 [0.66–0.82] | 0.636 | 0.615 [0.56–0.75] | 380 |
| B. + handling features | 0.781 | 0.742 [0.66–0.83] | 0.652 | 0.589 [0.51–0.71] | 376 |
| C. B with survival model | 0.781 | 0.746 [0.66–0.83] | 0.693 | 0.680 [0.58–0.77] | 365 |
| D. B + previous attempt | 0.813 | 0.785 [0.71–0.86] | 0.792 | 0.764 [0.70–0.85] | 351 |
| E. B + all surgeon information | 0.819 | 0.808 [0.74–0.88] | 0.745 | 0.772 [0.69–0.87] | 352 |
| F. DINOv2 only (uses no tool labels) | 0.786 | 0.786 [0.71–0.86] | 0.467 | 0.467 | 344 |
| G. DINOv2 + B | 0.810 | 0.798 [0.72–0.87] | 0.553 | 0.535 | 348 |
| H. SOCALNet-style re-build | 0.740 | 0.742 [0.66–0.83] | 0.638 | 0.633 [0.56–0.73] | 350 |

The phase 1 models, rerun on automatic labels (`results/auto_phase1_held_out.csv`):

| Method | Severe AUC, human → auto | Failure AUC, human → auto | RMSE mL, human → auto | R² in mL, human → auto |
|---|---|---|---|---|
| B. Video only (log) | 0.781 → 0.742 | 0.652 → 0.589 | 360 → 376 | 0.13 → 0.05 |
| D. B + previous attempt (log) | 0.813 → 0.785 | 0.792 → 0.764 | 338 → 351 | 0.24 → 0.17 |
| D1. D, square root | 0.803 → 0.774 | 0.792 → 0.764 | 322 → 334 | 0.30 → 0.25 |
| P3. B + all-surgeon-information prior from all 365 trials | 0.790 → 0.770 | 0.723 → 0.703 | 323 → 326 | 0.30 → 0.29 |
| E1. B + all surgeon information (143 trials) | 0.804 → 0.789 | 0.745 → 0.772 | 332 → 339 | 0.26 → 0.23 |
| G1. DINOv2 + B (video only) | 0.803 → 0.784 | 0.553 → 0.535 | 318 → 323 | 0.33 → 0.30 |
| **DG. DINOv2 + D** | **0.818 → 0.799 [0.73–0.88]** | 0.738 → 0.732 | **312 → 316** | **0.35 → 0.33** |
| PG. DINOv2 + P1 | 0.812 → 0.796 | 0.635 → 0.586 | 310 → 316 | 0.36 → 0.33 |
| H. SOCALNet-style re-build | 0.740 → 0.742 | 0.638 → 0.633 | 349 → 350 | 0.19 → 0.18 |

What this shows:
- **The model we carry forward (DG for blood loss, D for failure) works with no human labels: severe AUC 0.80, failure AUC 0.76, RMSE 316 mL.** With human labels it scores 0.82, 0.79 and 312. So removing the human annotator costs about 0.02 to 0.03 AUC and 4 mL.
- **It still beats the SOCALNet-style re-build on all three scores, and all three ranges exclude zero.** DG against H on automatic labels: severe AUC +0.059 [+0.002, +0.123], failure AUC +0.120 [+0.037, +0.208], RMSE −25 mL [−45, −4]. D's failure classifier against H: +0.134 [+0.040, +0.225].
- **The cost falls on the tool-handling features.** With human labels the 10 new handling features lift A to B (0.757 to 0.781 severe AUC). With automatic labels they add nothing (0.744 to 0.742), and B's failure AUC drops from 0.652 to 0.589. These features count cottonoid and muscle use, the two tools the detector finds least well (0.68 and 0.53). Better detection of those two is the clearest way to close the gap, which is what phase 2 of the plan tests.
- Models that lean on the surgeon's history or on DINOv2 frame features lose the least (E: 0.819 to 0.808, DG: 0.818 to 0.799). The square-root target still lowers error on automatic labels (D to D1: −15 mL [−32, +2]), and DG is still the best blood-loss model (−35 mL against D [−58, −10]).

Other checks, human → auto:

| Check | Human labels | Automatic labels |
|---|---|---|
| Best-of chosen inside training (nested), severe / failure AUC | 0.807 / 0.792 | 0.790 / 0.771 |
| Leave one course site out, D, severe / failure AUC | 0.762 / 0.772 | 0.716 / 0.768 |
| Leave one course site out, E, severe / failure AUC | 0.775 / 0.759 | 0.759 / 0.776 |
| D after only 15 s of video, severe / failure AUC | 0.713 / 0.758 | 0.732 / 0.755 |
| D after 45 s of video, severe / failure AUC | 0.788 / 0.806 | 0.756 / 0.798 |
| D failure call (risk above 0.5): accuracy / caught / false alarms | 76% / 54% / 12% | 73% / 45% / 12% |
| D over 500 mL, cutoff tuned in training: accuracy / caught / false alarms | 75% / 72% / 24% | 70% / 68% / 29% |

### Automatic labels, scored SOCALNet's way

Same 200 test sets of 20 videos as in phase 1 (`results/auto_phase1_protocol.csv`). Cells show the median with the 5th–95th percentile range.

| Test sets | Method | Success accuracy | RMSE (mL) | R² | Squared correlation | Bleeds > 500 mL caught | Sets at or under 295 mL |
|---|---|---|---|---|---|---|---|
| | SOCALNet (published, one set) | 85% | 295 | | 0.74 | 80% | |
| Random | D, human labels | 75% [60–90] | 320 [217–450] | 0.24 | 0.38 | 47% | 38% |
| Random | D, automatic | 75% [55–90] | 336 [224–465] | 0.17 | 0.30 | 47% | 27% |
| Random | DG, human labels | 70% [55–90] | 299 [201–412] | 0.37 | 0.43 | 58% | 48% |
| Random | **DG, automatic** | 70% [55–85] | **304 [202–411]** | 0.35 | 0.41 | 58% | 44% |
| Random | H re-build, automatic | 65% [50–75] | 352 [258–473] | 0.17 | 0.29 | 45% | 15% |
| Balanced like theirs | D, human labels | 70% [55–85] | 357 [236–500] | 0.17 | 0.37 | 46% | 22% |
| Balanced like theirs | D, automatic | 70% [55–80] | 372 [245–492] | 0.12 | 0.32 | 48% | 17% |
| Balanced like theirs | DG, human labels | 65% [55–80] | 328 [217–448] | 0.34 | 0.45 | 57% | 32% |
| Balanced like theirs | **DG, automatic** | 60% [55–80] | **323 [215–439]** | 0.34 | 0.44 | 58% | 31% |
| Balanced like theirs | H re-build, automatic | 60% [50–70] | 347 [244–450] | 0.21 | 0.38 | 49% | 20% |

- **Scored their way, the fully automatic model gives 75% accuracy (D) and 304 mL RMSE (DG) on random test sets, and 70% and 323 mL on sets balanced like theirs.** That is within 5 mL of the human-label result in both designs.
- The comparison with the published SOCALNet numbers is unchanged by removing human labels: blood-loss error is close to theirs on random test sets (304 vs 295 mL), and accuracy (70–75% vs 85%) and big bleeds caught (58% vs 80%) are still behind.
- With a cutoff tuned in training, the automatic D1 model catches 73% of bleeds over 500 mL at 33% false alarms.

### Phase 2, Part A: cleaning up the tool tracks

The raw automatic labels use one confidence cutoff (0.35, the app's) for every tool and take every frame on its own. Part A re-ran each fold's detector on the surgeons it never saw, keeping weak boxes too, and picked three settings per tool. The settings are a confidence cutoff, the longest gap in a tool's track to fill in (0, 1 or 2 seconds), and the shortest run of detections to keep (1 or 2 seconds). They are judged by how well "is this tool on screen this second?" matches the human labels. **The settings used for each fold's surgeons were picked only on the other four folds' surgeons**, so no surgeon's human labels touched their own cleanup (`calibre/experiments/track_cleanup.py`, settings in `results/clean_track_settings.csv`).

The picked settings were nearly the same in all 5 folds. Suction keeps a 0.25 cutoff and no smoothing. Cottonoid drops to 0.10–0.15 with 1–2 second gaps filled and one-second blips removed. Muscle uses 0.15–0.20 with 2-second gaps filled and blips removed. String drops to 0.10 with blips removed.

**Per-second agreement with human labels (F1), all 22,064 seconds:**

| Tool | Seconds on screen (human) | App setting (0.35) | Cleaned |
|---|---|---|---|
| Suction | 15,712 | 0.938 | 0.938 |
| Grasper | 10,983 | 0.916 | 0.915 |
| Cottonoid | 7,225 | 0.789 | **0.842** |
| Muscle | 3,144 | 0.631 | **0.693** |
| String | 7,976 | 0.899 | **0.911** |

**Outcome prediction, same trials, paired over surgeons (`results/clean_vs_auto_paired.csv`):**

| Model | Severe AUC raw → cleaned (gap [95%]) | Failure AUC raw → cleaned (gap [95%]) | RMSE mL raw → cleaned (gap [95%]) |
|---|---|---|---|
| A. Original 7 features | 0.744 → 0.760 (+0.021 [+0.002, +0.045]) | 0.615 → 0.617 (+0.003 [−0.021, +0.032]) | 380 → 380 (0 [−5, +7]) |
| B. + handling features | 0.742 → 0.762 (+0.024 [+0.001, +0.048]) | 0.589 → 0.547 (−0.026 [−0.070, +0.021]) | 376 → 365 (−10 [−20, −1]) |
| D. B + previous attempt | 0.785 → 0.806 (+0.018 [−0.006, +0.045]) | 0.764 → 0.774 (0.000 [−0.019, +0.021]) | 351 → 341 (−10 [−19, −2]) |
| E. B + all surgeon info | 0.808 → 0.816 (+0.010 [−0.005, +0.026]) | 0.772 → 0.759 (−0.006 [−0.028, +0.018]) | 352 → 352 (0 [−5, +4]) |
| G. DINOv2 + B | 0.798 → 0.801 (+0.005 [−0.015, +0.025]) | 0.535 → 0.504 (−0.042 [−0.121, +0.044]) | 348 → 343 (−4 [−15, +5]) |
| DG. DINOv2 + D (phase 1 best) | 0.799 → 0.804 | 0.732 → 0.744 | 316 → 318 |

The raw and cleaned columns are averages over the 10 reshuffles. The gap is from the averaged predictions, so it can differ from the plain subtraction by about 0.003. DG comes from `results/clean_phase1_held_out.csv`. Its main table is `results/clean_main.csv`.

- **Cleanup lifts the weak tools' tracks.** Cottonoid agreement rose 0.79 → 0.84 and muscle 0.63 → 0.69. The tools that were already good did not move.
- **It gives a small, real gain in blood-loss prediction for the plain feature models.** Severe AUC rose about 0.02 for A and B (ranges exclude 0), and RMSE fell about 10 mL for B and D (ranges exclude 0). With cleaned labels, D comes within 0.007 AUC and 3 mL of the same model on human labels (0.813, 338 mL).
- **It does not change failure prediction or the best model.** DG stays at about 0.80 severe AUC and 316–318 mL, because its DINOv2 image features carry most of the signal.
- **The 10 handling features still add nothing over the original 7 with automatic labels** (B vs A severe gap +0.004 [−0.031, +0.038]). The feature check below shows why.

**How close each feature from automatic labels is to the same feature from human labels** (correlation across the 143 trials, `results/clean_feature_agreement.csv`): the blood features are identical (r = 1.00), because blood is measured from color, not from tool labels. Suction and cottonoid time, suction clearing and two-handed time agree well (r 0.82–0.94). The weak ones are **first second a cottonoid appears** (r 0.40 raw, 0.48 cleaned), **first second a grasper appears** (0.48, 0.42) and **tool switches per second** (0.66, 0.67). A single early false detection sets a "first second" feature wrong, and every blip counts as a switch. These are where better detection, or features less sensitive to single frames, would help most.

Decision: the cleaned labels become the automatic default (`clean` label set), since they are never worse in a way the ranges can show, and they are better for blood loss in the plain models. Part B (the detector search on the PC) is below.

### Phase 2, Part B: the detector search on fold 0 (final)

The detector search (`calibre/experiments/detector_search.py`) ran on the PC (RTX 4060). Every model is trained on the same fold-0 training surgeons (22,012 frames, every second), picked by its score on held-out validation surgeons (3,369 frames), and scored once on the 15 fold-0 test surgeons (6,062 frames) that the current detector never saw either. YOLO runs use 30 rounds. RetinaNet is SOCAL's detector design (ResNet-50 + FPN), trained by us for the standard 12 epochs on every second training frame, with its best round picked on validation.

Scoring (`uv run python -m calibre.experiments.detector_interim`, numbers in `results/detector_search/interim_runs.csv` and `interim_gaps.csv`) uses our own mAP50 for every model, with a 95% range from resampling the 15 test surgeons. Gaps are paired: both models are scored on the same resampled surgeons each round.

| Model | mAP50 [95% range] | Suction | Grasper | Cottonoid | Muscle | String |
|---|---|---|---|---|---|---|
| SOCAL RetinaNet (published, different split) | 0.67 | 0.91 | 0.77 | 0.52 | 0.25 | 0.50 |
| **RetinaNet, trained by us on this split** | **0.766** [0.729, 0.805] | 0.942 | 0.882 | 0.706 | 0.560 | 0.742 |
| Current app detector (YOLO11n, 20 rounds, every 4th frame) | 0.792 [0.757, 0.827] | 0.946 | 0.907 | 0.686 | 0.613 | 0.808 |
| YOLO11n, 30 rounds, every frame | 0.825 [0.796, 0.856] | 0.960 | 0.932 | 0.751 | 0.642 | 0.840 |
| YOLO11s, 640 px | 0.841 [0.808, 0.875] | 0.963 | 0.940 | 0.779 | 0.671 | 0.852 |
| YOLO11s, 960 px | 0.854 [0.819, 0.888] | 0.964 | 0.944 | 0.803 | 0.705 | 0.851 |
| **YOLO11s, 960 px + rare-tool fix (final setup)** | **0.857** [0.824, 0.889] | 0.965 | 0.947 | 0.819 | 0.694 | 0.860 |
| YOLO11m, 640 px | 0.862 [0.830, 0.892] | 0.964 | 0.945 | 0.817 | 0.708 | 0.875 |

Paired gaps in mAP50 [95% range]:

| Comparison | mAP50 gap | Cottonoid | Muscle |
|---|---|---|---|
| **Final setup minus our RetinaNet** | **+0.090 [+0.074, +0.106]** | +0.113 [+0.083, +0.137] | +0.134 [+0.079, +0.192] |
| Our RetinaNet minus current app detector | −0.026 [−0.039, −0.011] | +0.019 [−0.001, +0.043] | −0.053 [−0.114, +0.004] |
| Final setup minus current app detector | +0.065 [+0.053, +0.077] | +0.132 [+0.097, +0.165] | +0.081 [+0.059, +0.101] |
| n (training recipe only) minus current | +0.033 [+0.019, +0.046] | +0.065 [+0.036, +0.095] | +0.029 [−0.006, +0.066] |
| s 960 px minus s 640 px (resolution) | +0.013 [+0.001, +0.025] | +0.024 [+0.007, +0.046] | +0.034 [−0.013, +0.088] |
| Rare-tool fix minus s 960 px | +0.003 [−0.007, +0.013] | +0.016 [0.000, +0.028] | −0.011 [−0.046, +0.016] |
| m 640 px minus final setup | +0.005 [−0.003, +0.014] | −0.002 [−0.019, +0.018] | +0.014 [−0.011, +0.042] |

- **On the same surgeons, SOCAL's detector design scores 0.766, not 0.67.** Most of the gap between our Phase 1 detector (0.775) and the published 0.67 therefore comes from the split and training, not from YOLO. The published number should not be used as the baseline for detector claims. Our own RetinaNet is the fair one.
- **Against that fair baseline, the final setup is better by 0.090, and every tool's range excludes 0.** The gains are largest on the hard tools: muscle +0.134, string +0.117, cottonoid +0.113.
- **The gain splits into training recipe (+0.033), model size (+0.016 for s over n) and resolution (+0.013).** The rare-tool fix (oversampling and copy-paste) adds nothing measurable (+0.003, range crosses 0; muscle slightly lower). It stays in the final setup only because the setup was picked on validation, as fixed before the run (0.866 vs 0.860).
- **The small phone model at 960 px matches the largest model** (m minus final setup +0.005, range crosses 0). So one model can serve both the phone and the offline labels, which removes the open Part C question and the phone-vs-offline check it needed.
- Plan bars: 2.2 (best setup ≥ 0.80) met. 2.3 (muscle ≥ 0.55, cottonoid ≥ 0.72) met by the final setup (0.694, 0.819), but by resolution and size, not by the rare-tool fix. 2.4 (same-split gap with a range) met.
- Our RetinaNet's validation score was still rising at its last epoch (0.707, 0.759, 0.775 at epochs 8, 10, 12). It used the standard 12-epoch schedule; a longer schedule might add a little. With a gap of 0.09, that would not change the result.
- Our scorer and Ultralytics agree within 0.003 on every model. The PC and Mac scores of RetinaNet and the final setup are identical (0.766, 0.857).

### Phase 3.1: tool motion features (human labels)

Nine features describe how the tools move in the first minute, not only whether they are on screen (`calibre/experiments/motion_features.py`): suction path length, how uneven its movement is, share of seconds it holds still, grasper speed and stillness, distance from the suction to the middle of the blood, share of suction seconds over the blood, share of seconds suction and grasper work close together, and how far the middle of the blood moves per second. The blood's location uses the same colour rule as the blood amount. Run with `uv run python -m calibre.experiments.phase3`; numbers in `results/human_phase3_motion.csv`.

| Model | Severe AUC, without → with | Failure AUC, without → with | RMSE mL, without → with |
|---|---|---|---|
| B. video only | 0.781 → 0.767 (−0.010 [−0.039, +0.017]) | 0.652 → 0.596 (−0.042 [−0.091, +0.010]) | 360 → 370 (+10 [−3, +22]) |
| D. B + previous attempt | 0.813 → 0.803 (−0.007 [−0.032, +0.019]) | 0.792 → 0.759 (−0.017 [−0.041, +0.010]) | 338 → 349 (+9 [−3, +22]) |
| DG. DINOv2 + D | 0.818 → 0.805 (−0.014 [−0.030, −0.001]) | 0.738 → 0.733 (−0.010 [−0.036, +0.017]) | 312 → 323 (+11 [+2, +19]) |

- **Motion features do not help. They make every model a little worse**, and for the best model (DG) the loss is clear on severe AUC and blood-loss error. This is a negative result, kept as is. No subset of the nine was tried afterwards, to avoid picking features on the test results.
- The likely reason is 143 trials: nine more inputs add noise faster than signal. The fine-tuned model in Phase 4 learns from every second instead of from 143 trial totals, which is the planned way around this.
- With human labels, these are the best-case motion features. They were not rerun on automatic labels.

## Limitations

- These are cadaver simulation trials only, with 143 trials from 73 surgeons, so the 95% ranges are wide.
- The previous-attempt feature only exists for second attempts (70 of 143 trials). It fits a training course, where a trainee's last attempt is known.
- The blood measurement is a hand-tuned color rule.
- The SOCALNet comparison uses our rebuild, not their trained model.
