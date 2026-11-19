# Paper plan: from "better than SOCAL" to a new method

Written 2026-11-18. Builds on `docs/improvement_plan.md` (Phases 1–6) and the results in `docs/experiments.md`.

## 1. Where we stand

| Area | Result (same held-out surgeons, automatic labels) | Driven by | New idea or newer tech |
|---|---|---|---|
| Tool detection (fold 0) | 0.857 vs 0.766 mAP50 (SOCAL's RetinaNet retrained), +0.090 [+0.074, +0.106] | training recipe, model size, 960 px | mostly newer tech |
| Severe bleed > 500 mL | AUC 0.80 vs 0.74 (SOCALNet-style rebuild) | previous attempt, DINOv2, square-root target | idea + newer features |
| Not controlled in 5 min | AUC 0.76 vs 0.63, +0.12 [+0.04, +0.21] | previous attempt | **new idea** |
| Blood loss error | 316 vs 350 mL, −25 [−45, −4] | DINOv2, square-root target | modelling |
| No human labels | costs 0.02–0.03 AUC | own detector + track cleanup | **new** (SOCALNet listed this as a limitation) |
| Previous attempt on 103 unseen surgeons | AUC 0.86 | surgeon history | **new** (SOCALNet listed individualized models as a limitation) |
| Tool motion features | slightly worse | | negative |
| Per-second fine-tuning, run 0 | signal learned (time-left alone: failure AUC 0.70), no gain over D | | inconclusive: stage 2 collapsed |

SOCALNet's own paper names two limitations: it is not fully automatic (it takes human tool labels) and "individualized models are required which incorporate surgeon experience". We already answer both (Pangal et al., Sci Rep 2022, https://doi.org/10.1038/s41598-022-11549-2).

### The weak spot

On video alone, failure prediction is 0.59 AUC on automatic labels, below the rebuild's 0.63. The failure gain is surgeon history. Severe-bleed and blood-loss gains do use video. **A strong paper needs the video part to carry its own weight**, and needs one method that is clearly new, not a newer backbone.

### What the data allows

- 143 trials with video and outcomes (73 surgeons), 365 trials with outcomes (177 surgeons).
- Every second of every video (31,443 frames) has human tool boxes, a blood measurement, and a known "seconds until control" (censored at 300 s for failures).
- So we have about **25,000 labelled seconds but only 143 labelled trials**. Every result so far says: be rich where there are many examples (seconds), be simple where there are few (trials).

## 2. Proposed new method: HemoSurv

**One model that watches the bleed second by second and estimates, at every second, the chance the surgeon gets control in the next second. Blood loss is how fast blood is lost, added up over the time until control. The surgeon's history sets their personal starting risk.**

1. **Per-second state from video.** For each second: detected tools (our detector), visible blood (colour rule), and the per-second encoder's outputs (Phase 4: tools, blood, time-left, all learned from every second).
2. **Discrete-time survival ("hazard") model.** Each second of each trial is one training row: "was control achieved in this second?" Failures stop at 300 s without control (censored, handled exactly). That turns 143 trials into about 25,000 rows, so the outcome model itself learns from every second, not from 25 summary numbers. This is what failed in Phase 4 run 0, fixed at the root.
3. **Surgeon frailty.** A per-surgeon offset on the hazard, estimated from the surgeon's earlier attempts and shrunk towards the average (empirical Bayes) using all 365 trials, including the 222 without video. That makes the individualized prior part of the model instead of three extra columns.
4. **Blood loss = rate × time.** A per-second bleeding-rate model (from visible blood, suction clearing, red-out) multiplied by the predicted time to control, integrated over the predicted survival curve. That is the physical relationship SOCAL's own analysis points to: time to hemostasis alone explains about half of blood loss (Kugener et al., JAMA Netw Open 2022, https://doi.org/10.1001/jamanetworkopen.2022.3177).
5. **Live by construction.** The same model updates every second, so the lead-time result (how early a failure is predicted) comes for free, and it is what the app needs.

**Why it is new:** dense per-second survival supervision from automatically detected video, with a surgeon frailty term learned partly from trials without video, and blood loss derived from rate × time. None of these is in SOCALNet; together they are one coherent method, not a bag of features.

**What we can claim if it works:** better failure and blood-loss prediction from video, explained by the method (ablations below), with surgeon history as a principled part of the model.

**Honest risk:** it may not beat D. The ablations still make a publishable analysis, and D's results stand on their own.

## 3. Fair baselines (a reviewer will ask for each)

1. **Faithful SOCALNet rebuild.** Their paper: ImageNet ResNet with the last three layers retrained on SOCAL frames, 4 numbers per frame + 8 tool flags, two-layer LSTM. Our current rebuild freezes the ResNet and uses 5 tool flags, so it may be weaker than theirs. Rebuild it as described (GPU, a few hours).
2. **LSTM given our information.** The rebuild plus the previous attempt and our automatic per-second tools and blood. Separates "better information" from "simpler model" (Mac, ~1 h).
3. **Gradient-boosted trees** as the final model, chosen inside training folds (Mac, ~1 h).
4. **Phase 4 stage 2 fixed** (a few principal components instead of 25 numbers), as the simplest dense-to-trial baseline (Mac, ~30 min).
5. **Model combination** (Phase 5), weights learned inside training folds (Mac, ~2 h).

## 4. Ablations for HemoSurv (the evidence that the method, not the tech, gives the gain)

| Remove or swap | Tests |
|---|---|
| Survival rows → one row per trial | does per-second supervision matter |
| No frailty / frailty from video trials only | does the surgeon term matter, and do the 222 no-video trials help |
| Rate × time → direct blood-loss regression | does the physical decomposition matter |
| Phase 4 encoder → frozen ImageNet ResNet / DINOv2 / tools + blood only | is the learned per-second state needed |
| Human vs automatic (Phase 1 / Part C) labels | cost of full automation |
| Same backbone as SOCALNet (ResNet-50) throughout | gain is not from a newer backbone |

All paired against D and the faithful rebuild on the same surgeon folds, 3 encoder seeds, 95% ranges from resampling surgeons, model choices made inside training folds only.

## 5. Everything left to run

| # | Work | Where | Time | Needs |
|---|---|---|---|---|
| 1 | LSTM + our information, gradient-boosted trees, Phase 4 stage-2 fix | Mac | ~3 h | nothing |
| 2 | **HemoSurv v0** on existing per-second data (tools, blood, Phase 4 seed 0 outputs) | Mac | ~1 day to build, minutes to run | nothing |
| 3 | Phase 4 seeds 1 and 2 (fresh encoders, confirm on new runs) | home PC | ~3 h each | nothing |
| 4 | Faithful SOCALNet rebuild | home PC | a few hours | small new script |
| 5 | Phase 4 frozen + task ablations | home PC | ~2 nights | nothing |
| 6 | Part C: final detector on folds 1–4, new labels | lab computer | ~1–2 days GPU | lab computer |
| 7 | Rerun everything on the new labels | Mac | ~1 h | 6 |
| 8 | Phase 5 combination; Phase 6 calibration, lead time, experts' 47.5% comparison, subgroups | Mac | ~1–2 days | 2, 7 |
| 9 | Real OR video (agreed with Dr. Zada for after the main paper) | later | | OR clips |

Dropped: Phase 3.1 motion features (negative). Deferred: 3.2 surgery-pretrained backbone (only if safe weights exist; it would also weaken the same-backbone argument), 3.3 video transformers (too data-hungry for 147 videos). Phase 3.4 (running prediction) is absorbed into HemoSurv.

## 6. Where to submit (to decide with Dr. Zada)

- **Medical image computing venues are the natural fit:** MICCAI (main conference, deadline usually late February or early March; to be checked for 2027), IPCAI, and the MICCAI and CVPR surgical-data-science workshops.
- **A general computer vision main conference (CVPR, ICCV, ECCV) is a long shot** with 147 videos and a method built for small data; their workshops are realistic.
- **A clinical journal** (Dr. Zada's field) suits the previous-attempt finding and the automatic pipeline, independent of how HemoSurv turns out.
- A reasonable route: method paper to MICCAI or a CV workshop, clinical paper to a neurosurgery journal. Dr. Zada decides.

## 7. Success bars (set now, before running)

- HemoSurv beats D on failure AUC or blood-loss RMSE with the paired range above 0, on automatic labels.
- Failure AUC ≥ 0.82 and severe AUC ≥ 0.85 with surgeons held out (Phase 4 plan bar).
- Under SOCALNet's own protocol (20-video tests), RMSE below 295 mL on the median test set.
- Every gain survives the same-backbone ablation.
