# Phase 2: do one set of automatic tool labels beat another? Same trials, same surgeon resampling.
#   uv run python -m calibre.experiments.clean_vs_auto                 (Part A: cleaned vs raw Phase 1 labels)
#   uv run python -m calibre.experiments.clean_vs_auto clean clean2    (Part C: new detectors vs Phase 1, both cleaned)
import sys
import pandas as pd
from calibre.components.vid import boxes_files
from calibre.experiments.harness import evaluate, paired_differences
from calibre.experiments.run_experiments import build_table, tabular_methods, image_methods
old_set, new_set = sys.argv[1:3] if len(sys.argv) >= 3 else ("auto", "clean")
tables = {k: build_table(boxes_files[k]) for k in [old_set, new_set]}
assert (tables[old_set]["trial_id"].values == tables[new_set]["trial_id"].values).all()
methods = {**tabular_methods(), **image_methods(boxes_files[old_set])}
rows = []
for name in ["A. Original CALIBRE (7 features)", "B. + 10 new bleed-handling features",
             "D. B + surgeon's previous attempt", "E. B + all surgeon information", "G. DINOv2 + B"]:
    ml, fail = methods[name]
    s = {k: evaluate(tables[k], ml, fail, name) for k in tables}
    # same trials in both tables, so resampling surgeons pairs them; table only supplies outcomes and surgeons
    gap = paired_differences(tables[new_set], s[new_set], s[old_set])
    for metric in ["severe_auc", "failure_auc", "rmse_ml"]:
        rows.append({"method": name, "metric": metric, old_set: s[old_set].loc[metric, "score"],
                     new_set: s[new_set].loc[metric, "score"], "gap": gap.loc[metric, "gap"],
                     "low": gap.loc[metric, "low"], "high": gap.loc[metric, "high"]})
    print(name, flush=True)
out = pd.DataFrame(rows)
out_name = "clean_vs_auto" if (old_set, new_set) == ("auto", "clean") else f"{new_set}_vs_{old_set}"
out.to_csv(f"results/{out_name}_paired.csv", index=False)
print(out.round(3).to_string(index=False))
