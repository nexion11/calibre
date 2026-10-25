import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from calibre.models.risk_model_training import (blood_features, severe_bleed_amt_ml, tool_features, # type: ignore
                                                 load_training_table)
warnings.filterwarnings("ignore")

feature_sets = {
    "tools": tool_features,
    "blood": blood_features,
    "tools+blood_mean": tool_features + ["blood_mean"],
    "tools+blood": tool_features + blood_features,
    "tools+blood+motion": tool_features + blood_features + ["suction_speed"],
}
model_types = {
    "linear": lambda: LinearRegression(),
    "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
    "forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=8, n_jobs=-1, random_state=0),
    "boosting": lambda: GradientBoostingRegressor(max_depth=2, n_estimators=150,
                                                  learning_rate=0.05, subsample=0.8, random_state=0),
}
reshuffles = 10


def train_and_predict(model_type, use_log, train_inputs, train_blood_loss, test_inputs):
    model = model_types[model_type]()
    
    if use_log:
        model.fit(train_inputs, np.log(train_blood_loss))
        return np.exp(model.predict(test_inputs))
    model.fit(train_inputs, train_blood_loss)
    
    return model.predict(test_inputs)


def prediction_scores(blood_loss, predictions):
    is_severe = blood_loss > severe_bleed_amt_ml
    
    return {
        "ml_error": np.abs(predictions - blood_loss).mean(),
        "severe_auc": roc_auc_score(is_severe, predictions),
        "caught": (predictions[is_severe] > severe_bleed_amt_ml).mean(),
        "false_alarms": (predictions[~is_severe] > severe_bleed_amt_ml).mean(),
    }


def cross_validation(training_table, feature_names, model_type, use_log, reshuffles=reshuffles):
    inputs, blood_loss = training_table[feature_names].values, training_table["blood_loss"].values
    surgeon_ids = training_table["participant_id"].values
    scores = []
    
    for reshuffle in range(reshuffles):
        predictions = np.zeros(len(blood_loss))
        folds = GroupKFold(n_splits=5, shuffle=True, random_state=reshuffle)
        for train_rows, test_rows in folds.split(inputs, blood_loss, surgeon_ids):
            predictions[test_rows] = train_and_predict(model_type, use_log, inputs[train_rows], blood_loss[train_rows], inputs[test_rows])
        scores.append(prediction_scores(blood_loss, predictions))
    
    return pd.DataFrame(scores).mean()


def every_cand():
    for feature_set in feature_sets:
        for model_type in model_types:
            for use_log in [False, True]:
                yield feature_set, model_type, use_log


def nested_cross_val(training_table):
    blood_loss, surgeon_ids = training_table["blood_loss"].values, training_table["participant_id"].values
    predictions = np.zeros(len(blood_loss))
    winners = []
    
    outer_folds = GroupKFold(n_splits=5, shuffle=True, random_state=99)
    
    for train_rows, test_rows in outer_folds.split(training_table, blood_loss, surgeon_ids):
        inner_table = training_table.iloc[train_rows]

        def inner_auc(candidate):
            feature_set, model_type, use_log = candidate
            
            return cross_validation(inner_table, feature_sets[feature_set], model_type, use_log,
                                             reshuffles=3)["severe_auc"]

        best = max(every_cand(), key=inner_auc)
        winners.append(best)
        feature_names = feature_sets[best[0]]
        predictions[test_rows] = train_and_predict(best[1], best[2],
                                                   training_table.iloc[train_rows][feature_names].values,
                                                   blood_loss[train_rows],
                                                   training_table.iloc[test_rows][feature_names].values)
    
    return prediction_scores(blood_loss, predictions), winners


if __name__ == "__main__":
    training_table = load_training_table()
    leaderboard = []
   
    for feature_set, model_type, use_log in every_cand():
        result = cross_validation(training_table, feature_sets[feature_set], model_type, use_log)
        leaderboard.append({"features": feature_set, "model": model_type,
                            "target": "log" if use_log else "mL", **result})
    leaderboard = pd.DataFrame(leaderboard).sort_values("severe_auc", ascending=False)
    
    pd.set_option("display.width", 140)
    print(leaderboard.round(3).to_string(index=False))

    print("nested cv:")
    honest_score, _ = nested_cross_val(training_table)
    for metric, value in honest_score.items():
        print(f"  {metric}: {value:.3f}")
