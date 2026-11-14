# Phase 2, Part A: do the cleaned-up tool tracks beat the raw automatic ones? Same trials, same surgeon resampling.
#   uv run python -m calibre.experiments.clean_vs_auto
import pandas as pd
from calibre.components.vid import boxes_files
from calibre.experiments.harness import evaluate, paired_differences
from calibre.experiments.run_experiments import build_table, tabular_methods, image_methods
tables = {k: build_table(boxes_files[k]) for k in ["auto", "clean"]}
assert (tables["auto"]["trial_id"].values == tables["clean"]["trial_id"].values).all()
methods = {**tabular_methods(), **image_methods(boxes_files["auto"])}
rows = []
for name in ["A. Original CALIBRE (7 features)", "B. + 10 new bleed-handling features",
             "D. B + surgeon's previous attempt", "E. B + all surgeon information", "G. DINOv2 + B"]:
    ml, fail = methods[name]
    s = {k: evaluate(tables[k], ml, fail, name) for k in tables}
    # same trials in both tables, so resampling surgeons pairs them; table only supplies outcomes and surgeons
    gap = paired_differences(tables["clean"], s["clean"], s["auto"])
    for metric in ["severe_auc", "failure_auc", "rmse_ml"]:
        rows.append({"method": name, "metric": metric, "auto": s["auto"].loc[metric, "score"],
                     "clean": s["clean"].loc[metric, "score"], "gap": gap.loc[metric, "gap"],
                     "low": gap.loc[metric, "low"], "high": gap.loc[metric, "high"]})
    print(name, flush=True)
out = pd.DataFrame(rows)
out.to_csv("results/clean_vs_auto_paired.csv", index=False)
print(out.round(3).to_string(index=False))
