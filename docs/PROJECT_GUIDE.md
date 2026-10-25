# BleedWatch: project guide

Everything about the project in one place: the medical problem, the data, the pipeline, the code, how the models are trained, the problems we solved, and a bank of statistics with sources for slides.

---

## 1. The medical problem

### The surgery
**Endoscopic endonasal surgery (EES)**, also called endoscopic skull-base or transsphenoidal surgery, reaches the base of the brain *through the nose*. The surgeon passes a thin camera (an endoscope) and long instruments through the nostrils and the sphenoid sinus to reach tumors such as pituitary adenomas. There is no cut in the skin and recovery is fast, which is why it has largely replaced older open approaches.

- About **15,000+ pituitary tumors** are diagnosed in the US each year (American Cancer Society).
- About **6,000–10,000 pituitary surgeries** a year in US hospital data (Nationwide Inpatient Sample studies).

### The danger: the internal carotid artery (ICA)
The two **internal carotid arteries** carry most of the blood to the brain. In this surgery they run along both sides of the sphenoid sinus and pituitary gland, often only millimeters from where the surgeon is working, sometimes covered by bone thinner than paper.

**Laceration** means a tear or cut. A **carotid artery laceration** (or "ICA injury") happens when an instrument, often a drill or cutting tool, accidentally tears the artery wall.

### What happens when it tears
1. Blood under arterial pressure floods the narrow nasal corridor. Within seconds the camera view is completely red ("red-out"), so the surgeon is working blind.
2. The team must stop the bleeding fast. Standard steps:
   - **Suction** to keep a small view open, with a second surgeon often driving the camera.
   - **Cottonoid (cotton patty) packing:** pressing cotton patties on the tear for temporary control.
   - **Crushed muscle patch:** a piece of the patient's own muscle pressed over the tear. Muscle is considered the gold standard for lasting control (SOCAL documentation).
   - Then usually **angiography**, with the **endovascular team** sealing or stenting the artery from inside, or closing it off if the brain can tolerate that.
3. Later risks include stroke, a **pseudoaneurysm** (a weak bulge that can re-bleed days or weeks later), and nerve damage.

### Key statistics (with sources)
| Fact | Number | Source |
|---|---|---|
| How often it happens | 0.004% to 3.4% of endonasal surgeries, depending on approach; 0.46% (18 of 3,889) in one large single-center series | Laryngoscope 2016 systematic review; review in World J Otorhinolaryngol HNS; Thieme 2021 series |
| Death rate when it happens | About 10% to 14% ("up to 1 in 7") | NCBI *Complications in Neurosurgery II* (2024); Delphi protocol paper (PMC9331087) |
| Severe disability among survivors | Up to 24% | NCBI *Complications in Neurosurgery II* |
| Why it's rarely practiced | Fewer than 0.5% of cases, so surgeons "may not develop proficiency" | SOCAL dataset documentation (USC) |
| Risk higher early in a surgeon's career | Greater risk early in the learning curve | J Neurol Surg B 2018 |
| Before vs after simulation training | 56% → 90% controlled the bleed; blood loss −37%; time to control −38% (177 surgeons) | Donoho et al., *J Neurosurg* 2021 |
| Experts judging blood loss from video | Caught only 47.5% of bleeds over 500 mL; underestimated on average | Pangal et al., *Scientific Reports* 2022 |
| Humans estimate blood loss poorly in general | Large losses underestimated; accuracy doesn't improve with experience | Reviews of visual blood-loss estimation; BMC Surgery 2024 |

**Check before citing:** the 56% → 90% numbers came from a summary of Donoho 2021, so read its abstract yourself. The "50% of experienced neurosurgeons have faced an ICA injury" figure (Ciric survey, 1997) is quoted second-hand; verify it before using.

---

## 2. The data: SOCAL

**SOCAL = Simulated Outcomes following Carotid Artery Laceration**, a public dataset from USC Neurosurgery (figshare, CC BY-NC 4.0; cite the data release and Kugener et al., *JAMA Network Open* 2022).

- **What it shows:** surgeons managing a deliberate carotid tear in a **perfused cadaver** (a donated body with artificial blood pumped through it at real pressure). Recorded at nationwide training courses, 2017–2020.
- **Who:** 177 surgeons, neurosurgeons and ENT surgeons, attendings and residents.
- **Outcomes:** 365 trials, each with blood loss (mL), time to hemostasis (seconds) and success (bleed controlled within 5 minutes).
- **Video:** 147 trials have video, 31,443 frames at 1 frame per second.
- **Labels:** 65,071 tool bounding boxes: suction, grasper, cottonoid, muscle, string, drill, scalpel, tool.
- **What we used:** 143 trials have both video labels and outcomes (73 surgeons with video).

---

## 3. The pipeline

```
Video frame (1 per second)
   │
   ├─► Tool detector (YOLO11 nano, Core ML) ──► boxes: suction, grasper, cottonoid, muscle, string
   │
   └─► Blood measurement (color rule) ──────► % of the view covered in blood
                    │
                    ▼
   7 features over the first 60 seconds
   (suction_frac, cottonoid_frac, tools_per_frame,
    blood_mean, blood_max, blood_trend, blood_late)
                    │
                    ▼
   Ridge regression on log(blood loss)  ──► estimated blood loss
                    │
                    ▼
   Risk tier (LOW / MODERATE / HIGH) + 80% range + the top reason
```

Everything after the camera runs **on the iPhone**, with no internet.

---

## 4. The code, file by file (`~/dev/bleedwatch`)

### Python (data, training, analysis): `src/`
| File | What it does |
|---|---|
| `explore.py` | First look at the data: counts, blood-loss summary, video lengths |
| `features.py` | Turns each trial's first 60 s into features. **You wrote** `cottonoid_frac`, `tools_per_frame`, `suction_speed` and `blood_mean` |
| `blood_coverage.py` | Measures blood in all 31,443 frames (about 23 s on the Mac, all 14 CPU cores) |
| `train.py` | Trains and tests the risk model and saves it. **You wrote** the first model's train/predict lines |
| `model_search.py` | Compares 40 model/feature combinations with overfitting guards (repeated + nested CV) |
| `live_analysis.py` | Tests accuracy at 15, 30 … 180 s and against surgeon experience; draws the chart |
| `train_detector.py` | Builds the YOLO dataset and trains the tool detector on the Mac's GPU |
| `export_app_assets.py` | Exports the app's risk model (`risk_model.json`) and the 3 sample videos |
| `colab/` | Google Colab version of the blood and detector steps |

### iPhone app (Swift / SwiftUI): `ios/BleedWatch/`
| File | What it does |
|---|---|
| `HomeView.swift` | Home screen: Live camera, Open video, sample cases |
| `ScopeView.swift` | The monitor screen: video, tool boxes, blood map, risk readout, trend line |
| `SessionEngine.swift` | Runs a session: one frame per second → analyze → update the estimate |
| `VideoSources.swift` | Camera input (`CameraSource`) and video-file playback (`FileSource`) |
| `FrameAnalyzer.swift` | Blood color rule (`BloodDetector`) and YOLO via Vision/Core ML (`ToolDetector`) |
| `RiskModel.swift` | Computes the same 7 features, applies the ridge weights, picks the tier and reasons |
| `Models.swift` | Data types: tool box, observation, sample cases |
| `BleedWatchTools.mlpackage` | The trained detector (5 MB) |
| `Resources/risk_model.json` | The trained risk model: 7 weights, scaling numbers, tiers, range |

---

## 5. Libraries and tools
| Library / tool | Used for |
|---|---|
| **pandas** | Loading and joining the CSV tables (boxes, frames, outcomes) |
| **NumPy** | Math on arrays (slopes, log transform, image pixels) |
| **scikit-learn** | Ridge regression, scaling, GroupKFold cross-validation, AUC |
| **Pillow (PIL)** | Reading and resizing frames, converting to HSV color |
| **Ultralytics YOLO11 + PyTorch** | Training the tool detector (on the Mac's GPU via MPS) |
| **coremltools** | Converting the detector to Apple's Core ML format |
| **matplotlib** | The accuracy-over-time chart |
| **ffmpeg** | Turning frames into the sample videos |
| **uv** | Managing the Python environment and packages |
| **Swift, SwiftUI** | The iPhone app and its interface |
| **AVFoundation** | Camera capture and video playback |
| **Vision + Core ML** | Running the detector on the phone |
| **XcodeGen / Xcode** | Building the app |
| **git** | Version control (commits e34b42b, cc73a36, a6f99fe) |

---

## 6. How the models are trained

### A. Blood measurement (a rule, not trained)
For each frame, resized to 480×270 and converted to HSV color:
- **Inside the endoscope circle:** brightness V > 0.12, which ignores the black border.
- **Blood:** red hue (H < 0.04 or > 0.94) **and** very saturated (S > 0.75) **and** not too bright (V < 0.75). Pink tissue is less saturated, so it's excluded.
- **Output:** blood pixels ÷ pixels in the circle. Tuned by eye on sample frames, and checked to match between Python and the iPhone (within about 1%).

### B. Tool detector (YOLO11 nano)
- **Data:** SOCAL's boxes for 5 tools, with every 4th frame used (frames 1 second apart look nearly the same). 6,126 training and 1,735 validation images.
- **Honest split:** 20% of **surgeons** held out for validation, so it's tested on people it never saw.
- **Training:** 20 epochs, 640 px images, batch 32, on the Mac's GPU (about an hour).
- **Result:** mAP50 **0.75** (published RetinaNet benchmark: 0.67).

  | Tool | mAP50 |
  |---|---|
  | Suction | 0.94 |
  | Grasper | 0.90 |
  | String | 0.81 |
  | Cottonoid | 0.66 |
  | Muscle | 0.42 |

- **Converted to Core ML** for the iPhone (5 MB).

### C. Risk model (ridge regression)
- **Input:** 7 features from the **first 60 seconds only**, to avoid *data leakage*. Muscle appearing basically means the bleed is over, so watching longer would give away the answer.
- **Target:** **log(blood loss)**, so a few huge bleeds (up to 1,890 mL) don't dominate.
- **Model:** ridge regression, a linear model that shrinks weights to avoid overfitting, with features standardized first.
- **Testing:**
  - **GroupKFold by surgeon:** each surgeon's trials stay on one side of the split.
  - **10× repeated CV:** surgeons reshuffled into 5 groups, ten times over.
  - **Nested CV:** an honest score for the whole "pick the best of 40 models" process.
- **Outputs, all measured on held-out surgeons:**
  - **Alert line:** set so the model catches about 75% of severe bleeds.
  - **Tiers:** LOW (9% became severe), MODERATE (19%), HIGH (56%). The overall rate is 36%.
  - **80% range:** actual blood loss fell between 0.25× and 3.2× the estimate 80% of the time.
- **The app model** is retrained without the 3 sample-case surgeons, so the demos are honest.

### Results
| Model | Severe-bleed AUC | Severe caught | False alarms |
|---|---|---|---|
| Always guess the average | — | 0% | 0% |
| First model (3 tool features, linear) | 0.69 | 65% | 25% |
| **Final (7 features, ridge, log)** | **0.77** (nested CV: **0.75**) | **75%** | 33% |
| Surgeon experience only | 0.60 | — | — |

- **Warning timing:** usable at 15 s (AUC 0.68) and best at 60 s (0.75). It fades after about 90 s, because the easy cases have already ended.
- **Weak spots:** exact mL (R² 0.18, while SOCALNet reported 0.74 on 20 videos) and predicting "will it be controlled" (AUC about 0.64).

**What the model learned:**
- **Biggest protective sign:** using the cottonoid early (weight −0.25).
- **Biggest warning sign:** blood still covering the view late in the first minute (weight +0.24).

---

## 7. Problems we hit and how we solved them
| Problem | Fix |
|---|---|
| Quest 2 can't give apps its camera feed | Switched to an iPhone camera or video |
| Spine dataset had only 24 outcomes | Chose SOCAL (365 outcomes) |
| Pink tissue looked like blood | Required high saturation and low brightness, not just red hue |
| `suction_speed` correlated (+0.35) but made the model worse | Learned about redundant features; dropped it from the model |
| Risk of fooling ourselves with 40 models | Repeated and nested cross-validation |
| Detector export failed (PyTorch too new for coremltools) | Exported in a temporary PyTorch 2.7 environment |
| Prediction spiked to 3,497 mL in the first seconds | Wait 20 s before using `blood_trend`; cap features at ±3 standard deviations |
| Sample surgeons were in the detector's training data | Picked sample cases from surgeons neither model saw |
| Exact mL was too imprecise to show | Showed a risk tier + range instead |
| False HIGH at 6 s on a controlled case | No risk call before 15 s |

---

## 8. Limitations (say these honestly)
- **Cadaver data only:** not tested on real patients, so today it's a **training tool**.
- **Blood measurement is a hand-tuned color rule,** not a trained AI.
- **Detector misses some cottonoids** (0.66) and **muscle** (0.42), which can make early calls too pessimistic.
- **About 1 in 3 controlled bleeds gets a false alarm.**
- **No surgeon feedback yet.** The next step is asking surgeons, for example the USC SOCAL team.

---

## 9. Glossary
| Term | Meaning |
|---|---|
| **Endoscope** | Thin camera tube used inside the body |
| **Cottonoid** | Small cotton patty pressed on bleeding |
| **Hemostasis** | Stopping the bleeding |
| **Angiography / endovascular** | Imaging and repairing blood vessels from inside, through a catheter |
| **Pseudoaneurysm** | A weak, bulging spot after an artery injury that can re-bleed |
| **Feature** | A number summarizing the video that the model learns from |
| **Data leakage** | Accidentally letting the model see the answer |
| **Overfitting** | A model memorizing training data instead of learning patterns |
| **Cross-validation (CV)** | Repeatedly training on some data and testing on the rest |
| **AUC** | Chance the model ranks a random severe bleed above a random controlled one (0.5 = coin flip, 1 = perfect) |
| **mAP50** | Detection score: how well boxes find tools (box overlap ≥ 50% counts) |
| **R²** | Share of the variation in blood loss the model explains |
