# Prior work on SOCAL, and what is new in CALIBRE

Compiled 2026-11-20 from PubMed and web searches. All SOCAL work found is from the USC / Children's National group (Zada, Donoho, Pangal, Kugener). No paper from another group using SOCAL turned up, and none after 2023.

## 1. Every SOCAL paper found

| Paper | Task | Input at test time | How it was tested | Headline result |
|---|---|---|---|---|
| Donoho et al., J Neurosurg 2021 (https://doi.org/10.3171/2020.9.JNS202672) | Training outcomes of the simulator (177 surgeons) | none (outcomes only) | before vs after coaching | success 56% → 90%; blood loss −37%, time to control −38% |
| Kugener et al., JAMA Netw Open 2022 (https://doi.org/10.1001/jamanetworkopen.2022.3177) | Dataset release; tool detection; tool-use metrics vs blood loss; OR video | frames | 124 / 9 / 14 trials (split not listed) | RetinaNet mAP50 0.67 (YOLOv3 0.53); time to control explains 49.8% of blood loss; OR video sensitivity 0.77, PPV 0.96 (any instrument) |
| Pangal et al., Sci Rep 2022, SOCALNet (https://doi.org/10.1038/s41598-022-11549-2) | Success and blood loss from the first 60 s, vs 4 experts | frames + **human** tool labels | 127 / 20 videos, split by video, 11/9 outcomes | 85% (17/20), RMSE 295 mL, r = 0.74 (Pearson; Supplementary Fig. 1), 8/10 bleeds > 500 mL; vs experts p = 0.12 |
| Kugener et al., Neurosurgery 2022 (https://doi.org/10.1227/neu.0000000000001906) | Success and blood loss from video | frames only ("automated") or frames + human tool labels ("semiautomated") | 123 / 20 trials | automated: RMSE 358 mL, R² 0.40, 85%; semiautomated: RMSE 260 mL, 90% |
| Toh et al., letter, Neurosurgery 2023 (https://doi.org/10.1227/neu.0000000000002590) and reply (https://doi.org/10.1227/neu.0000000000002591) | Commentary: time-series architectures for other neurosurgical tasks | | | (full text not read; summaries say it is not about test design) |
| Pangal et al., J Neurosurg 2022, APMs (https://doi.org/10.3171/2021.10.JNS211064) | 60 hand-made performance metrics from **human** tool boxes over the **whole trial** | full trial, human labels | 132 trials, train/test split not described | elastic net on log blood loss R² 0.87; surgeon experience and attending status not predictive |
| Kugener et al., Neurosurg Focus 2022, code-free ML (https://doi.org/10.3171/2022.1.FOCUS21652) | Tool detection with AutoML | frames | same split as the dataset paper | mAP 0.708 (RetinaNet 0.669, YOLOv3 0.527) |
| Kugener et al., Oper Neurosurg 2022 (https://doi.org/10.1227/ons.0000000000000274) | Tool detection on real endonasal YouTube video | frames | 3 YouTube videos | mAP 0.74 |
| Kugener et al., Oper Neurosurg 2023, Shannon entropy (https://doi.org/10.1227/ons.0000000000000888) | Diversity of **automatically detected** tool sequences → success | detected tools, whole trial | not given in the abstract | 78% accuracy, 97% average precision; similar to experts |

## 2. Our claims, checked

| Claim | Status | Why |
|---|---|---|
| Fully automatic, no human labels | **Not new on its own** | Kugener 2022 (Neurosurgery) predicts from frames alone; the 2023 entropy paper uses detected tools. We can say ours is automatic and interpretable, not that automation is a first. |
| The surgeon's previous attempt predicts the next trial, confirmed on 103 unseen surgeons | **Likely new** | No SOCAL paper uses a surgeon's earlier attempt as a predictor. SOCALNet lists individualized models as future work; the APM paper finds experience and attending status not predictive, which fits. Donoho 2021 reports trial 1 → trial 2 improvement; its full text should be checked for a trial 1–trial 2 correlation. |
| Surgeon-held-out testing on all trials with 95% ranges; single 20-video tests swing 50–90% | **Likely new** | Every outcome paper uses one 20-video test (or an undescribed split) without confidence intervals. The letter appears to be about architectures, not test design (to confirm). |
| HemoSurv: per-second time-to-control (survival) modelling; adds +0.043 failure AUC and −9 mL to the best model on automatic labels | **New** | No SOCAL paper predicts time to control or uses survival models. |
| Early warning (failure AUC ~0.76 after 15 s; accuracy across 15–120 s) | **Likely new** | SOCALNet and Kugener 2022 use a fixed 60 s; APMs and entropy use the whole trial. |
| Tool detection mAP50 0.857 (fold 0), 0.775 on all 75 surgeons | Better than every published SOCAL detector (0.67, 0.708) | Mostly newer models; supporting result, not the contribution. |
| Beats SOCALNet | **Only against our reimplementation, on our fair test** | No published code (the "eSupp1" code is not in the supplement); published numbers come from one 20-video test that we cannot reproduce. |

## 3. Where we stand against the published numbers (their protocol, our models before HemoSurv)

| | Best published on SOCAL | Ours, scored their way (median of 20-video tests) |
|---|---|---|
| Success accuracy | 85% (SOCALNet), 90% (Kugener 2022, with human tool labels) | 75% (55–90%) |
| Blood-loss RMSE | 295 mL (SOCALNet), 260 mL (Kugener 2022, with human tool labels), 358 mL (frames only) | 304 mL (DG, automatic labels) |
| Correlation with blood loss | r = 0.74 (SOCALNet) | r ≈ 0.64–0.67 |
| Experts (same 20 videos) | 69%, 350 mL, r = 0.70, 47.5% of bleeds > 500 mL | we beat these on accuracy and RMSE |

We do not beat the published headline numbers. They come from single 20-video tests that are not surgeon-held-out, so they cannot be compared directly; that is a finding to report, not a claim to beat.

## 4. How to improve on all of it

1. **Re-run every prior method under one fair protocol** (surgeon-held-out, all trials, 95% ranges): SOCALNet (faithful rebuild), the Kugener 2022 frames-only and semiautomated networks, the APM elastic net (on the first 60 s and on the whole trial), and the Shannon-entropy model. The entropy and APM models are cheap (Mac). This turns "we beat a rebuild" into "every published SOCAL method, compared fairly."
2. **Ask the lab for the 20 test video IDs and the trained models.** With them we report results on the exact published test too.
3. **Borrow the best published signals:** instrument disappearance counts (the strongest APM correlate, r = 0.73 over the whole trial), time to each phase, and tool-sequence entropy, computed from our automatic labels in the first 60 s, added to D and HemoSurv.
4. **Report the metrics they reported** (accuracy, sensitivity, specificity, RMSE, Pearson r, bleeds > 500 mL caught) alongside AUC, so readers can line the numbers up.
5. **Keep HemoSurv and the previous attempt as the method contributions**, and the fair re-evaluation of all prior SOCAL work as the evaluation contribution.
