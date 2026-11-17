# Phase 3.1: do tool-motion features add anything on top of the models we already have?
# Same rules as everything else: whole surgeons held out, 5 folds x 10 reshuffles, paired gaps from resampling surgeons.
#   uv run python -m calibre.experiments.phase3                     (human tool labels)
#   CALIBRE_LABELS=clean2 uv run python -m calibre.experiments.phase3   (Part C detector labels, cleaned)
import os
import pandas as pd
from calibre.components.vid import boxes_files
from calibre.experiments.harness import evaluate, failure_classifier, paired_differences
from calibre.experiments.image_models import trial_embedding_summary
from calibre.experiments.motion_features import add_motion_features, motion_feature_names
from calibre.experiments.new_models import with_missing_filled
from calibre.experiments.phase1 import ridge_on, with_embedding
from calibre.experiments.run_experiments import build_table, last_attempt, video_features

labels = os.environ.get("CALIBRE_LABELS", "human")


def model(features, blood_target="log", dino=None):
    def blood(more):
        return with_missing_filled(ridge_on(blood_target), more)

    def failure(more):
        return with_missing_filled(failure_classifier, more)
    if dino is None:
        return blood(features), failure(features)
    return (with_embedding(dino, blood, features), with_embedding(dino, failure, features))


if __name__ == "__main__":
    table = add_motion_features(build_table(boxes_files[labels]), boxes_files[labels])
    print(f"{labels} labels: {len(table)} trials")
    print(table[motion_feature_names].describe().round(3).T[["count", "mean", "50%"]].to_string())
    dino = trial_embedding_summary("dinov2") if os.path.exists("data/embeddings_dinov2.npz") else None
    b, d = video_features, video_features + last_attempt
    pairs = {  # name: (old features, new features, blood target, uses DINOv2)
        "B. video only": (b, b + motion_feature_names, "log", False),
        "D. B + previous attempt": (d, d + motion_feature_names, "log", False),
        "DG. DINOv2 + D (current best blood loss)": (d, d + motion_feature_names, "sqrt", True),
    }
    rows = []
    for name, (old_features, new_features, target, uses_dino) in pairs.items():
        if uses_dino and dino is None:
            continue
        old = evaluate(table, *model(old_features, target, dino if uses_dino else None), name)
        new = evaluate(table, *model(new_features, target, dino if uses_dino else None), name + " + motion")
        gap = paired_differences(table, new, old)
        for metric in ["severe_auc", "failure_auc", "rmse_ml", "false_alarms_at_75pct_caught"]:
            rows.append({"method": name, "metric": metric, "without_motion": old.loc[metric, "score"],
                         "with_motion": new.loc[metric, "score"], "gap": gap.loc[metric, "gap"],
                         "low": gap.loc[metric, "low"], "high": gap.loc[metric, "high"]})
        print(name, "done", flush=True)
    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv(f"results/{labels}_phase3_motion.csv", index=False)
    print(out.round(3).to_string(index=False))
