# Plan: improving on the original SOCAL results

Written 2026-11-14. Current results are in `docs/experiments.md`. This is what to run next, why, and how each result will be judged.

**Phase 1 is done** (results in `docs/experiments.md`, "Phase 1"). Sections 1 and 2 below are updated with what it found. "Before" is where we stood when this plan was written.

## 1. Where we stand

| Scoreboard | Original SOCAL | Before | After phase 1 | Status |
|---|---|---|---|---|
| Tool detection, mAP50 | 0.67 (RetinaNet) | 0.75 (YOLO11n, one split) | 0.775 on all 75 surgeons, each scored by a detector that never saw them | ahead, every tool higher (plan item 2.1 done) |
| Blood loss error under their test, RMSE | 295 mL | 320 mL | 299 mL on random test sets, 328 mL on sets balanced like theirs | level on random sets, 33 mL behind on balanced sets |
| Blood loss R² under their test | 0.74 (a squared correlation) | 0.25 | squared correlation 0.43 to 0.45, R² 0.33 to 0.37 | not comparable (see 2a) |
| Success accuracy under their test | 85% | 75% | 75% random, 70% balanced | behind |
| Bleeds over 500 mL caught | 80% | 50% | 58%, or 70% with a tuned cutoff and more false alarms | behind |
| Severe-bleed AUC, surgeons held out | not reported | 0.81 | 0.82 | no published number to beat |
| Failure AUC, surgeons held out | not reported | 0.79 | 0.79 | no published number to beat |
| Blood loss RMSE, surgeons held out | not reported | 338 mL | 312 mL | clear gain (−24 mL, range −39 to −8) |
| Works with no human tool labels | no | pending | yes: severe AUC 0.80, failure AUC 0.76, RMSE 316 mL fully automatic (0.82, 0.79, 312 with human labels); 304 mL under their test on random sets | ahead; costs about 0.02 to 0.03 AUC and 4 mL |

## 2. What is holding us back

Three quick checks run today changed the picture.

**a. Their R² of 0.74 and RMSE of 295 mL do not fit a normal test set.** R² and RMSE together fix how spread out the test videos' blood loss must be. Their two numbers need a spread (standard deviation) of about 579 mL. Our 143 video trials have a spread of 388 mL, and only 1 in 1,000 random sets of 20 videos reaches 579. On a typical set of 20, an RMSE of 295 gives an R² of about 0.37. So RMSE is the number that can be compared fairly. This is not a claim that their paper is wrong. It is a question for the lab: which 20 videos, and how was R² calculated.

*Phase 1 update.* According to PubMed (Pangal et al., Sci Rep 2022, [DOI](https://doi.org/10.1038/s41598-022-11549-2)), the paper reports blood loss with RMSE and Pearson's correlation, so the 0.74 is a squared correlation, which is always at least as high as R². Their 20 test videos were also picked to balance outcomes (11 controlled, 9 failed). Balancing barely changes the spread (median 395 mL instead of 371), so it does not explain 0.74 either. It does make the test harder for us: balanced sets hold more failed trials with big bleeds, and our RMSE on them is 328 mL against 299 on random sets.

**b. Our blood-loss model predicts too low for big bleeds.** It learns log(blood loss) and converts back, which gives a typical value (about 280 mL) instead of the average (456 mL). Scores measured in mL punish that. Learning the square root instead already moved model D from R² 0.24 to 0.30 and RMSE from 338 to 322 mL with surgeons held out. That is a one-line change.

*Phase 1 update.* Confirmed. The square root is the only target whose gain is clear (−14 mL, range −25 to −1). Adding DINOv2 features on top takes held-out RMSE to 312 mL. The models still run about 90 mL low on balanced test sets, so big bleeds remain the weak spot.

**c. SOCAL has more outcome data than we have been using.** `socal_outcomes.csv` holds 365 trials from 177 surgeons. Only 143 trials have video. The link between a surgeon's first and second attempt holds across all 176 surgeons with two attempts (Spearman 0.60), and separately in the 103 surgeons with no video (0.64). That is an independent confirmation of our main finding, and 2.5 times more data for the surgeon part of the model.

*Phase 1 update.* A model fitted only on surgeons with video predicts the later attempts of 103 surgeons with no video at 0.86 severe AUC and 0.72 failure AUC. Feeding the 365-trial prior into the video model did not raise failure AUC (0.792 both ways), so the extra data confirms the finding but does not improve the model.

**d. Only 143 labeled videos.** Deep models overfit on that. The fix is to give them thousands of per-second training signals from the same videos (section 3, phase 4).

**e. Our inputs are still simple.** The blood measure is a hand-tuned color rule, tool features are mostly presence and timing, and the model looks at a fixed 60 seconds.

## 3. The experiments

Every outcome experiment uses the existing rules: surgeons held out, 5 folds × 10 reshuffles, surgeon-level 95% ranges, paired gap against a named baseline. Each is also scored under SOCALNet's 20-video protocol.

### Phase 1. Comparability and quick wins (Mac, about half a day)

| # | Experiment | Why | Success bar |
|---|---|---|---|
| 1.1 | Write up check 2a properly: implied test spread, share of random splits that reach it, R² as squared correlation vs. true R² | Explains the 0.74 gap with numbers, not guesses | Table in the paper |
| 1.2 | Blood-loss target test: log, log with correction, square root, raw mL, gamma model, Huber loss | Fixes 2b | RMSE under their protocol below 310 mL |
| 1.3 | Two-part model: chance of failure × expected loss if failed + chance of control × expected loss if controlled | Blood loss is two groups (controlled vs. ran the full 5 minutes); one line fits both badly | Beats the best of 1.2 |
| 1.4 | Predict "over 500 mL" and "success" with their own classifiers, with the cutoff picked inside the training set | Today we threshold a mL estimate that runs low, which is why we catch only 50% | Over-500 caught ≥ 70%, accuracy ≥ 78% |
| 1.5 | Surgeon-prior model trained on all 365 trials, then added to the video model | Uses 2c | Failure AUC ≥ 0.82 |
| 1.6 | Paired test of D and G directly against H (the SOCALNet rebuild) | The direct "ours vs. their design" number | Gap range excludes 0 |

### Phase 2. Better tool detector (PC GPU, about 1–2 days of training)

The detector feeds everything else, and it is the one place we already beat a published number.

| # | Experiment | Why | Success bar |
|---|---|---|---|
| 2.1 | Held-out mAP50 on all 73 surgeons from the 5 cross-fitted YOLO11n detectors | The honest baseline; every surgeon scored by a detector that never saw them | Stays ≥ 0.72 |
| 2.2 | Size and resolution test on fold 0: YOLO11 n / s / m, image size 640 vs. 960, all frames, 30 rounds (cut from 60 to fit the GPU; still about 6x the training of the current detector). 960 px and 2.3 run on the best phone-speed model (n or s) | Ours was the smallest model, on every 4th frame, and still improving at round 20 | Best setup ≥ 0.80 on fold 0 |
| 2.3 | Rare-tool fixes: oversample frames with muscle and cottonoid, copy-paste augmentation | Muscle is 0.42 and cottonoid 0.66 | Muscle ≥ 0.55, cottonoid ≥ 0.72 |
| 2.4 | RetinaNet trained by us on the same fold-0 split | Their 0.67 was on a split they did not describe. This makes the comparison same-data, same-surgeons | Paired gap with a range |
| 2.5 | Best setup trained on all 5 folds, new auto labels written | Needed for the fully automatic pipeline | mAP50 ≥ 0.80, lower end of range above 0.67 |
| 2.6 | Smoothing over time: fill 1–2 second gaps in a tool's track | Outcome features need per-second presence, not per-frame boxes | Per-second presence F1 up. **Done:** cottonoid 0.79 → 0.84, muscle 0.63 → 0.69; blood-loss RMSE −10 mL for D (range excludes 0); handling features still add nothing |
| 2.7 | iPhone export check: accuracy after Core ML conversion at 16-bit and 8-bit | Proves the deployed model matches the reported one | Drop under 0.01 mAP50 |

### Phase 3. Richer inputs (Mac, about 1 day)

| # | Experiment | Why |
|---|---|---|
| 3.1 | Tool motion features from tracked boxes: distance moved, speed, idle time, suction-to-blood distance, how long two tools overlap | Skill shows in how tools move, not only whether they are present |
| 3.2 | Surgery-pretrained image model in place of DINOv2 (candidates: EndoViT, SurgeNet, EndoFM; I will confirm safe downloadable weights first) | Already knows tissue, blood and instruments |
| 3.3 | Video model embeddings (VideoMAE or V-JEPA 2) on short clips | Captures motion that single frames miss |
| 3.4 | Running prediction: one model that updates every second from everything seen so far | Replaces the fixed 60 s window; this is what the app needs, and it gives a lead-time result |

Each is added on top of model B and tested as a paired gap. Only inputs whose gap range excludes 0 move forward.

### Phase 4. Fine-tuned vision model (PC GPU, about 1 day of training)

This is the experiment most likely to move the blood-loss number.

- **Stage 1, per-frame training.** Start from the best image model in phase 3. Train it on every labeled frame of every video, not just the first minute, to predict four things per frame: which tools are present, how much blood is visible, seconds left until the bleed is controlled, and whether muscle has been placed yet. That is tens of thousands of training signals instead of 143.
- **Stage 2, outcome model.** Freeze it, save its per-second output for the first 60 s, and train a small sequence model plus our ridge model on top.
- **Leak rule.** "Seconds left until control" comes from the outcome, so 5 encoders are trained, one per surgeon fold, and each surgeon is only ever scored with the encoder that never saw them. Training rows and test rows in a fold both use that fold's encoder.
- **Comparisons:** frozen vs. fine-tuned, with vs. without each per-frame task, ResNet-50 vs. the phase 3 winner, sequence model vs. simple averaging. Three random seeds each.
- **Success bar:** RMSE under their protocol below 295 mL, and severe AUC ≥ 0.85 with surgeons held out.

### Phase 5. Combining models (Mac, about 2 hours)

- Weighted average of the tabular model (B/D), the embedding model (G), the sequence model (H), and the phase 4 model. Weights are learned inside each training set, so there is no leak.
- Compared against a plain average and against the best single model with a paired gap.
- **Success bar:** beats the best single model on both severe AUC and RMSE. Expect a small gain in AUC (0.01–0.03) and a larger one in mL error, because the models make different mistakes.

### Phase 6. Evaluation that makes the paper hold up (Mac, about 1 day)

- **Final model picked honestly.** The whole "try everything, keep the best" procedure runs inside nested cross-validation, as `nested_best_of` does now.
- **Calibration:** when it says 60% risk, does it happen 60% of the time.
- **Decision curve:** at which alert levels the app is more useful than alerting everyone or no one.
- **Lead time:** seconds of warning before a bleed passes 500 mL.
- **Against the experts:** at the experts' false-alarm rate, how many bleeds over 500 mL does the model catch (they caught 47.5%).
- **Automatic vs. human labels** side by side for every model.
- **Subgroups:** attendings vs. trainees, each course site, first vs. second attempt.
- **Error review:** the 10 worst-missed videos, looked at by hand.
- **On the phone:** frames per second, battery and heat on a real iPhone, and a check that the app's features match the offline pipeline on the same video.

## 4. Short answers

- **Better detectors? Yes, first.** It is our clearest win, it runs unattended on the PC, and it improves the automatic pipeline.
- **Fine-tune a vision model? Yes, but only with per-frame tasks and only after phase 3 picks the starting model.** Fine-tuning straight on 143 outcomes will overfit; our rebuild of SOCALNet already showed that.
- **Combine models? Yes, last.** It only helps once the single models are as good as they can get, and it must be weighted inside the training set.

## 5. Rules that keep it honest

1. Main scores are fixed now: severe-bleed AUC and failure AUC with surgeons held out. Everything else is secondary.
2. A gain counts only if its paired range excludes 0. With this many comparisons the final table uses Holm correction.
3. Any model trained on outcomes or outcome-derived labels is cross-fitted by surgeon.
4. Deep models report the average of 3 seeds.
5. Results that did not work stay in the write-up.

## 6. Order

1. Phase 1 on the Mac now (no GPU needed).
2. Phase 2 on the PC as soon as folds 1–4 finish.
3. Phase 3 on the Mac while phase 2 trains.
4. Phase 4 on the PC after phase 2.
5. Phases 5 and 6 last.

## 7. What we can claim, depending on how it goes

- **Detector ≥ 0.80 and beats our own RetinaNet on the same split:** a clean, same-data improvement in tool detection.
- **RMSE under 295 mL on their protocol:** matches or beats SOCALNet's blood-loss error, with the R² gap explained by 2a.
- **RMSE stays near 310–320:** we report it as comparable, and the paper leans on what SOCALNet could not do: no human labels, per-surgeon prediction, tested on unseen surgeons and sites, running on a phone.

## 8. What to ask Dr. Zada's lab

1. The 20 test video IDs for SOCALNet, and the trained model if they still have it.
2. How R² was calculated.
3. Whether any SOCAL trials recorded after the public release exist. A locked model tested on new trials would be the strongest result in the paper.
4. Two or three residents for a short usability test of the app.
