# BleedWatch: demo video script (about 2:50, limit 3:00)

Each part matches a slide in the deck; the same lines are in its speaker notes.

| Time | Slide | On screen | Say |
|---|---|---|---|
| 0:00–0:15 | Cover | Black, then endoscope footage turning red | "You’re a surgeon operating through the nose, millimeters from the carotid artery. It tears. In seconds, your screen is red. Up to 1 in 7 patients die from this. And most surgeons have never practiced it." |
| 0:15–0:35 | Problem | Three big numbers | "It happens in under 1 in 200 of these surgeries, so surgeons can’t learn it on real patients. In a national study of 177 surgeons, only 56% could control this bleed before training. Even experts watching video caught fewer than half of the severe bleeds." |
| 0:35–0:55 | Training | 56% → 90% | "Simulation training works. After one coached session, success jumped from 56% to 90%, with 37% less blood lost. But that coaching needs an expert watching every single attempt. That’s the bottleneck." |
| 0:55–1:05 | Solution | App home screen | "So I built BleedWatch: an AI coach that gives every trainee expert-level feedback, using just an iPhone." |
| 1:05–1:35 | Demo | Screen recording: Severe bleed, then Controlled bleed | "I open a video or aim the camera at the surgical monitor. BleedWatch boxes every instrument it sees: suction, grasper, cotton patty. It measures how much of the view is blood. And by 15 seconds it calls the risk. Here: HIGH. Fifty-six percent of cases like this became severe bleeds. And it says why: the cotton patty was never used. In this other case, the surgeon packs the injury, and the risk drops to MODERATE." |
| 1:35–2:10 | How it works | Pipeline slide, then a few seconds of code | "Under the hood: I trained a YOLO object detector in Python on 31,000 frames from 177 surgeons in USC’s public SOCAL dataset, and converted it to run on the iPhone with Core ML. A color-analysis algorithm I wrote in Swift measures the blood. Seven features feed a ridge-regression model I trained to predict blood loss. The whole app is SwiftUI, and everything runs on the phone, with no internet needed." |
| 2:10–2:25 | Proof | Results slide | "I tested everything on surgeons the models had never seen. My tool detector scored 0.75, versus 0.67 for the original published benchmark on this dataset. And BleedWatch flagged 75% of severe bleeds; experts caught 47.5%." |
| 2:25–2:35 | Impact | Statement slide | "Any residency program with a phone can now coach every trainee, on every attempt. Better-trained surgeons mean that when this happens to a real patient, the surgeon has practiced it before." |
| 2:35–2:45 | Next | 1.0 vs 2.0 | "Next, BleedWatch 2.0 will mark the carotid artery on screen, to help prevent the injury in the first place." |
| 2:45–2:55 | Close | Closing line, name | "Surgeons get seconds. BleedWatch makes sure they’ve practiced them. I’m Aariv Bhalla, and this is BleedWatch." |

## Recording tips

- Record the demo from the app itself: the iPhone's built-in screen recorder, or the Simulator's File › Record Screen.
- The sample videos play at real speed, so speed up the demo clip about 2x when editing.
- Upload the final video to YouTube as Public (required), 1–3 minutes.

## Written answers for the submission form

- **Technical challenge:** The prediction jumped to 3,500 mL in the first seconds because a "blood trend" feature calculated from too few frames was mostly noise. I fixed it by waiting for enough data and capping extreme values. I also found my demo surgeons were in the detector's training set, so I re-picked cases neither model had seen, to keep the demo honest.
- **What I learned:** testing on unseen surgeons, nested cross-validation, and why a feature that correlates with the answer can still make a model worse.

## Sources

- 56% → 90%, 37% less blood lost: Donoho et al., Journal of Neurosurgery 2021 (check the abstract before recording)
- Up to 1 in 7 die, up to 24% severely disabled: NCBI, Complications in Neurosurgery II (2024)
- Under 0.5% of cases: SOCAL dataset documentation, USC Neurosurgery
- Experts caught 47.5% of severe bleeds: Pangal et al., Scientific Reports 2022
- Published detector benchmark 0.67: Kugener et al., JAMA Network Open 2022
- Carotid detection is feasible: Neurosurgical Focus 2025
