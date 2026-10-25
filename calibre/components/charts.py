import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from calibre.models.risk_model_training import (load_training_table, unseen_surgeon_tests,
                                                 model_features, severe_bleed_amt_ml)


appfolder = "phone/CALIBRE/Resources"

# names that make sense to someone who didnt write the code
nice_names = {
    "suction_amt": "Suction in view",
    "cottonoid_amt": "Cottonoid in view",
    "tools_per_frame": "Tools per frame",
    "blood_mean": "Avg blood in view",
    "blood_max": "Most blood in view",
    "blood_trend": "Blood rising over time",
    "blood_late": "Blood in last 15 s",
}


def predicted_vs_actual(training_table):
    blood_loss, predictions = unseen_surgeon_tests(training_table, "ridge", model_features)
    alert_line_ml = json.load(open("models/calibre_risk_model.json"))["alert_threshold_mL"]
    is_severe = blood_loss > severe_bleed_amt_ml

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(predictions[~is_severe], blood_loss[~is_severe], s=22, alpha=0.6, color="#5b8def", label="Not severe")
    ax.scatter(predictions[is_severe], blood_loss[is_severe], s=22, alpha=0.8, color="#e8483b", label="Severe (>500 mL)")
    ax.axhline(severe_bleed_amt_ml, color="gray", ls="--", lw=1)
    ax.axvline(alert_line_ml, color="black", ls=":", lw=1.2, label=f"CALIBRE alert line ({alert_line_ml:.0f} mL)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CALIBRE's estimate after 60 s (mL)")
    ax.set_ylabel("Actual blood loss (mL)")
    ax.set_title("Estimate vs what really happened\n(143 trials, surgeons the model never saw)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("models/predicted_vs_actual.png", dpi=200)


def model_weights():
    saved = json.load(open("models/calibre_risk_model.json"))
    # model predicts log(blood loss) so exp(weight) is how much the estimate multiplies by
    # for a normal sized bump (1 std) in that feature
    change = [(np.exp(w) - 1) * 100 for w in saved["weights"]]
    order = np.argsort(change)
    names = [nice_names[saved["features"][i]] for i in order]
    values = [change[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(names, values, color=["#e8483b" if v > 0 else "#5b8def" for v in values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Change in estimated blood loss (%)")
    ax.set_title("What the model weighs\n(red = more bleeding, blue = less)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig("models/model_weights.png", dpi=200)


if __name__ == "__main__":
    predicted_vs_actual(load_training_table())
    model_weights()

    for chart in ["predicted_vs_actual.png", "model_weights.png", "live_accuracy.png"]:
        shutil.copy(f"models/{chart}", appfolder)
