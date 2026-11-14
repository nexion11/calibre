# New ways to turn the first-minute features into a blood-loss prediction.
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from sklearn.preprocessing import StandardScaler
from calibre.models.risk_model_training import ridge_model

time_limit = 300  # SOCAL stops the trial at 5 minutes; failures are recorded as 300 s


def add_outcome_times(table):
    outcomes = pd.read_csv("data/socal_outcomes.csv")[["trial_id", "tth"]]
    table = table.merge(outcomes, on="trial_id", how="left")
    table["tth"] = table["tth"].fillna(time_limit)
    return table


def rate_times_time(features):
    """Blood loss = bleeding rate x time to control. Learn each part separately, then multiply."""
    def fit_predict(train, test):
        log_time = np.log(train["tth"].values)
        log_rate = np.log(train["blood_loss"].values) - log_time
        time_model = ridge_model().fit(train[features].values, log_time)
        rate_model = ridge_model().fit(train[features].values, log_rate)
        predicted_time = np.minimum(np.exp(time_model.predict(test[features].values)), time_limit)
        return np.exp(rate_model.predict(test[features].values)) * predicted_time
    return fit_predict


def survival_rate_times_time(features, penalty=1.0):
    """Same idea, but time to control comes from a survival model that knows failures were cut off at 5 min."""
    def fit_predict(train, test):
        scaler = StandardScaler().fit(train[features].values)
        columns = [f"f{i}" for i in range(len(features))]
        train_frame = pd.DataFrame(scaler.transform(train[features].values), columns=columns)
        train_frame["tth"] = train["tth"].values
        train_frame["controlled"] = (train["success"].values == 1).astype(int)
        survival = WeibullAFTFitter(penalizer=penalty).fit(train_frame, "tth", "controlled")
        test_frame = pd.DataFrame(scaler.transform(test[features].values), columns=columns)
        predicted_time = np.minimum(survival.predict_median(test_frame).values, time_limit)

        log_rate = np.log(train["blood_loss"].values) - np.log(train["tth"].values)
        rate_model = ridge_model().fit(train[features].values, log_rate)
        return np.exp(rate_model.predict(test[features].values)) * predicted_time
    return fit_predict


def survival_failure_risk(features, penalty=1.0):
    """Chance the bleed is NOT controlled by 5 minutes, straight from the survival model."""
    def fit_predict(train, test):
        scaler = StandardScaler().fit(train[features].values)
        columns = [f"f{i}" for i in range(len(features))]
        train_frame = pd.DataFrame(scaler.transform(train[features].values), columns=columns)
        train_frame["tth"] = train["tth"].values
        train_frame["controlled"] = (train["success"].values == 1).astype(int)
        survival = WeibullAFTFitter(penalizer=penalty).fit(train_frame, "tth", "controlled")
        test_frame = pd.DataFrame(scaler.transform(test[features].values), columns=columns)
        return survival.predict_survival_function(test_frame, times=[time_limit - 1]).values[0]
    return fit_predict


# ---------- individualized: who the surgeon is and how their last attempt went ----------

surgeon_feature_names = [
    "attending", "neurosurgery", "years_experience", "endo_cases_last_year", "cadaver_cases_last_year",
    "prior_real_carotid_injury", "carotid_confidence_before", "resting_heart_rate",
    "second_attempt", "last_attempt_log_blood_loss", "last_attempt_controlled",
]


def add_surgeon_features(table):
    people = pd.read_csv("data/socal_participant_demographics.csv")
    outcomes = pd.read_csv("data/socal_outcomes.csv")
    people_cols = pd.DataFrame({
        "participant_id": people["participant_id"],
        "attending": (people["training_status"] == "Attending").astype(float),
        "neurosurgery": (people["specialty"] == "Neurosurgery").astype(float),
        "years_experience": pd.to_numeric(people["total_years_experience"], errors="coerce"),
        "endo_cases_last_year": pd.to_numeric(people["endo_last_12_mo"], errors="coerce"),
        "cadaver_cases_last_year": pd.to_numeric(people["cadaver_last_12_mo"], errors="coerce"),
        "prior_real_carotid_injury": pd.to_numeric(people["prior_real"], errors="coerce"),
        "carotid_confidence_before": pd.to_numeric(people["carotid_confidence_pre"], errors="coerce"),
        "resting_heart_rate": pd.to_numeric(people["hr_baseline_average"], errors="coerce"),  # measured before the trial
        "cohort": people["cohort"],
    })
    # the surgeon's previous attempt is known before this attempt starts, so it's fair to use
    outcomes = outcomes.assign(attempt=outcomes["trial_id"].str[-1].astype(int))
    previous = outcomes[["participant_id", "attempt", "blood_loss", "success"]].copy()
    previous["attempt"] += 1
    previous = previous.rename(columns={"blood_loss": "last_attempt_blood_loss", "success": "last_attempt_controlled"})
    attempts = outcomes[["trial_id", "participant_id", "attempt"]].merge(previous, on=["participant_id", "attempt"], how="left")
    attempts["second_attempt"] = (attempts["attempt"] > 1).astype(float)
    attempts["last_attempt_log_blood_loss"] = np.log(attempts["last_attempt_blood_loss"])

    table = table.merge(attempts[["trial_id", "second_attempt", "last_attempt_log_blood_loss", "last_attempt_controlled"]],
                        on="trial_id", how="left")
    return table.merge(people_cols, on="participant_id", how="left")


def with_missing_filled(fit_predict_maker, features):
    """Fill gaps with the TRAINING median (never the test data) and add 'was missing' flags."""
    def fit_predict(train, test):
        train, test = train.copy(), test.copy()
        used = list(features)
        for name in features:
            if train[name].isna().any() or test[name].isna().any():
                flag = f"{name}_missing"
                train[flag], test[flag] = train[name].isna().astype(float), test[name].isna().astype(float)
                fill = train[name].median()
                fill = 0.0 if np.isnan(fill) else fill
                train[name], test[name] = train[name].fillna(fill), test[name].fillna(fill)
                used.append(flag)
        return fit_predict_maker(used)(train, test)
    return fit_predict
