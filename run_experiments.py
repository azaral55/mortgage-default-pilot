"""Pilot experiments: LR vs RF vs XGBoost vs LightGBM on forward-in-time snapshots.

Usage: python run_experiments.py <out_dir> [fred_csv]
"""
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import optuna
import xgboost
import lightgbm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR = sys.argv[1]
FRED_CSV = sys.argv[2] if len(sys.argv) > 2 else None
N_TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
BASE_SEED = 42
SEEDS = [42, 43, 44]
N_BOOTSTRAP = 200

NUMERIC_FEATURES = [
    "credit_score", "orig_ltv", "orig_cltv", "orig_dti", "orig_rate", "orig_upb", "orig_term",
    "mi_percent", "number_of_units", "number_of_borrowers", "loan_age", "current_upb",
    "current_rate", "current_dq", "worst_dq_12m", "months_delinquent_12m",
    "current_ltv_estimated", "upb_paid_share", "is_modified", "has_assistance",
]
CATEGORICAL_FEATURES = ["first_time_buyer", "occupancy", "channel", "property_type", "loan_purpose"]
MACRO_FEATURES = ["rate_gap", "state_unemployment", "state_unemployment_change_12m"]
HISTORY_FEATURES = ["current_dq", "worst_dq_12m", "months_delinquent_12m", "is_modified", "has_assistance"]


# ---------------------------------------------------------------- data

def add_macro_features(panel, fred_csv):
    fred = pd.read_csv(fred_csv)
    date_column = fred.columns[0]
    fred[date_column] = pd.to_datetime(fred[date_column])
    fred = fred.set_index(date_column).apply(pd.to_numeric, errors="coerce")
    monthly = fred.resample("MS").mean()
    monthly["period"] = monthly.index.year * 100 + monthly.index.month
    monthly = monthly.set_index("period")
    market_rate = monthly["MORTGAGE30US"]
    panel["rate_gap"] = panel["current_rate"] - panel["snapshot"].map(market_rate)
    level_values = []
    change_values = []
    for snapshot_period, state in zip(panel["snapshot"], panel["state"]):
        column = f"{state}UR"
        year_ago = snapshot_period - 100
        if column in monthly.columns and snapshot_period in monthly.index and year_ago in monthly.index:
            level = monthly.at[snapshot_period, column]
            level_values.append(level)
            change_values.append(level - monthly.at[year_ago, column])
        else:
            level_values.append(np.nan)
            change_values.append(np.nan)
    panel["state_unemployment"] = level_values
    panel["state_unemployment_change_12m"] = change_values
    return panel


def make_design(panel, feature_list, categorical_list, reference_columns=None):
    numeric = panel[feature_list].astype(float)
    if categorical_list:
        dummies = pd.get_dummies(panel[categorical_list].fillna("missing"), prefix=categorical_list, dtype=float)
        design = pd.concat([numeric, dummies], axis=1)
    else:
        design = numeric
    if reference_columns is not None:
        design = design.reindex(columns=reference_columns, fill_value=0.0)
    return design


class LogisticPreprocessor:
    """Median imputation + missing indicators + standardisation, fitted on training data only."""

    def fit(self, frame):
        self.medians = frame.median()
        self.missing_columns = [column for column in frame.columns if frame[column].isna().any()]
        filled = self._fill(frame)
        self.means = filled.mean()
        self.stds = filled.std().replace(0, 1.0)
        return self

    def _fill(self, frame):
        filled = frame.fillna(self.medians)
        for column in self.missing_columns:
            filled[column + "_missing"] = frame[column].isna().astype(float)
        return filled

    def transform(self, frame):
        filled = self._fill(frame)
        return ((filled - self.means) / self.stds).values


# ---------------------------------------------------------------- models

def build_model(model_name, params, seed, positive_weight):
    if model_name == "LR":
        return LogisticRegression(C=params["C"], l1_ratio=0, solver="lbfgs", max_iter=2000,
                                  class_weight={0: 1.0, 1: positive_weight})
    if model_name == "RF":
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                      min_samples_leaf=params["min_samples_leaf"],
                                      max_features=params["max_features"], max_samples=params["max_samples"],
                                      class_weight={0: 1.0, 1: positive_weight}, n_jobs=2, random_state=seed)
    if model_name == "XGBoost":
        return xgboost.XGBClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
                                     max_depth=params["max_depth"], min_child_weight=params["min_child_weight"],
                                     subsample=params["subsample"], colsample_bytree=params["colsample_bytree"],
                                     reg_lambda=params["reg_lambda"], scale_pos_weight=positive_weight,
                                     tree_method="hist", n_jobs=2, random_state=seed, eval_metric="auc")
    if model_name == "LightGBM":
        return lightgbm.LGBMClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
                                       num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
                                       subsample=params["subsample"], subsample_freq=1,
                                       colsample_bytree=params["colsample_bytree"], reg_lambda=params["reg_lambda"],
                                       scale_pos_weight=positive_weight, n_jobs=2, random_state=seed, verbose=-1)
    raise ValueError(model_name)


def suggest_params(trial, model_name):
    if model_name == "LR":
        return {"C": trial.suggest_float("C", 1e-4, 10.0, log=True)}
    if model_name == "RF":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 200, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 400, log=True),
                "max_features": trial.suggest_float("max_features", 0.1, 0.6),
                "max_samples": trial.suggest_float("max_samples", 0.2, 0.6)}
    if model_name == "XGBoost":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True)}
    if model_name == "LightGBM":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True)}
    raise ValueError(model_name)


def model_inputs(model_name, design_frame, preprocessor):
    if model_name == "LR":
        return preprocessor.transform(design_frame)
    if model_name == "RF":
        # Random forest in scikit-learn accepts NaN; still use median fill for stability
        return design_frame.fillna(preprocessor.medians).values
    return design_frame.values


# ---------------------------------------------------------------- calibration and metrics

class Calibrator:
    """Choose Platt scaling or isotonic regression by cross-validated Brier score on validation."""

    def fit(self, raw_scores, labels, seed):
        logit_scores = np.log(np.clip(raw_scores, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_scores, 1e-7, 1 - 1e-7)))
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        brier_platt = []
        brier_isotonic = []
        for train_index, hold_index in folds.split(logit_scores, labels):
            platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit_scores[train_index].reshape(-1, 1), labels[train_index])
            brier_platt.append(brier_score_loss(labels[hold_index], platt.predict_proba(logit_scores[hold_index].reshape(-1, 1))[:, 1]))
            isotonic = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(raw_scores[train_index], labels[train_index])
            brier_isotonic.append(brier_score_loss(labels[hold_index], isotonic.predict(raw_scores[hold_index])))
        self.method = "platt" if np.mean(brier_platt) <= np.mean(brier_isotonic) else "isotonic"
        if self.method == "platt":
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(logit_scores.reshape(-1, 1), labels)
        else:
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(raw_scores, labels)
        return self

    def transform(self, raw_scores):
        if self.method == "platt":
            clipped = np.clip(raw_scores, 1e-7, 1 - 1e-7)
            return self.model.predict_proba(np.log(clipped / (1 - clipped)).reshape(-1, 1))[:, 1]
        return self.model.predict(raw_scores)


def best_f1_threshold(probabilities, labels):
    order = np.argsort(-probabilities)
    sorted_labels = labels[order]
    true_positive_cumulative = np.cumsum(sorted_labels)
    predicted_positive = np.arange(1, len(labels) + 1)
    precision = true_positive_cumulative / predicted_positive
    recall = true_positive_cumulative / labels.sum()
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)
    best_position = int(np.argmax(f1))
    return float(probabilities[order][best_position])


def ks_statistic(labels, probabilities):
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    return float(np.max(true_positive_rate - false_positive_rate))


def expected_calibration_error(labels, probabilities, n_bins=10):
    order = np.argsort(probabilities)
    bins = np.array_split(order, n_bins)
    total = 0.0
    for bin_indices in bins:
        total += len(bin_indices) / len(labels) * abs(labels[bin_indices].mean() - probabilities[bin_indices].mean())
    return float(total)


def all_metrics(labels, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    true_positive = int(((predicted == 1) & (labels == 1)).sum())
    false_positive = int(((predicted == 1) & (labels == 0)).sum())
    false_negative = int(((predicted == 0) & (labels == 1)).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    brier = brier_score_loss(labels, probabilities)
    base_rate = labels.mean()
    return {
        "AUC": roc_auc_score(labels, probabilities),
        "KS": ks_statistic(labels, probabilities),
        "AP": average_precision_score(labels, probabilities),
        "Precision": precision, "Recall": recall, "F1": f1,
        "Brier": brier, "BSS": 1 - brier / (base_rate * (1 - base_rate)),
        "ECE": expected_calibration_error(labels, probabilities),
        "PredictedDefaults": float(probabilities.sum()), "ActualDefaults": int(labels.sum()),
    }


def fast_auc(labels, scores):
    return roc_auc_score(labels, scores)


# ---------------------------------------------------------------- experiment

def tune(model_name, train_inputs, train_labels, validation_inputs, validation_labels, positive_weight, n_trials):
    def objective(trial):
        params = suggest_params(trial, model_name)
        model = build_model(model_name, params, BASE_SEED, positive_weight)
        model.fit(train_inputs, train_labels)
        return fast_auc(validation_labels, model.predict_proba(validation_inputs)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def fit_and_predict(model_name, params, seed, positive_weight, sets):
    model = build_model(model_name, params, seed, positive_weight)
    model.fit(sets["train"][0], sets["train"][1])
    validation_raw = model.predict_proba(sets["validation"][0])[:, 1]
    calibrator = Calibrator().fit(validation_raw, sets["validation"][1], seed)
    validation_calibrated = calibrator.transform(validation_raw)
    threshold = best_f1_threshold(validation_calibrated, sets["validation"][1])
    test_calibrated = calibrator.transform(model.predict_proba(sets["test"][0])[:, 1])
    test_raw = model.predict_proba(sets["test"][0])[:, 1]
    return model, calibrator, threshold, test_calibrated, test_raw


def prepare_sets(panel, model_name, numeric_features, categorical_features):
    train = panel[panel["split"] == "train"]
    design_train = make_design(train, numeric_features, categorical_features)
    columns = list(design_train.columns)
    preprocessor = LogisticPreprocessor().fit(design_train)
    sets = {}
    for split_name in ["train", "validation", "test"]:
        part = panel[panel["split"] == split_name]
        design = make_design(part, numeric_features, categorical_features, columns)
        sets[split_name] = (model_inputs(model_name, design, preprocessor), part["target"].values)
    return sets, columns, preprocessor


def bootstrap(labels, probability_by_model, thresholds, reference_model, n_bootstrap, seed):
    random_state = np.random.RandomState(seed)
    n = len(labels)
    samples = {name: {key: [] for key in ["AUC", "AP", "BSS"]} for name in probability_by_model}
    auc_differences = {name: [] for name in probability_by_model if name != reference_model}
    for repetition in range(n_bootstrap):
        index = random_state.randint(0, n, n)
        sample_labels = labels[index]
        aucs = {}
        for name, probabilities in probability_by_model.items():
            sample_probabilities = probabilities[index]
            brier = np.mean((sample_probabilities - sample_labels) ** 2)
            base_rate = sample_labels.mean()
            metrics = {"AUC": roc_auc_score(sample_labels, sample_probabilities),
                       "AP": average_precision_score(sample_labels, sample_probabilities),
                       "BSS": 1 - brier / (base_rate * (1 - base_rate))}
            for key in samples[name]:
                samples[name][key].append(metrics[key])
            aucs[name] = metrics["AUC"]
        for name in auc_differences:
            auc_differences[name].append(aucs[name] - aucs[reference_model])
    intervals = {name: {key: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) for key, values in metric_values.items()}
                 for name, metric_values in samples.items()}
    difference_intervals = {name: {"mean": float(np.mean(values)),
                                   "low": float(np.percentile(values, 2.5)),
                                   "high": float(np.percentile(values, 97.5)),
                                   "share_le_zero": float(np.mean(np.array(values) <= 0))}
                            for name, values in auc_differences.items()}
    return intervals, difference_intervals


def main():
    start_time = time.time()
    panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
    numeric_features = list(NUMERIC_FEATURES)
    if FRED_CSV:
        panel = add_macro_features(panel, FRED_CSV)
        numeric_features = numeric_features + MACRO_FEATURES
        print("macro coverage", panel[MACRO_FEATURES].notna().mean().to_dict())
    train_labels = panel.loc[panel["split"] == "train", "target"].values
    positive_weight = float((train_labels == 0).sum() / (train_labels == 1).sum())
    model_names = ["LR", "RF", "XGBoost", "LightGBM"]
    results = {"positive_weight": positive_weight, "features": numeric_features + CATEGORICAL_FEATURES,
               "models": {}}
    test_panel = panel[panel["split"] == "test"].reset_index(drop=True)
    test_labels = test_panel["target"].values
    predictions = {"loan_id": test_panel["loan_id"], "snapshot": test_panel["snapshot"], "target": test_labels}

    for model_name in model_names:
        model_start = time.time()
        sets, columns, preprocessor = prepare_sets(panel, model_name, numeric_features, CATEGORICAL_FEATURES)
        trials = N_TRIALS
        best_params, best_validation_auc = tune(model_name, sets["train"][0], sets["train"][1],
                                                sets["validation"][0], sets["validation"][1], positive_weight, trials)
        seed_list = [BASE_SEED] if model_name == "LR" else SEEDS
        per_seed = []
        stored_probabilities = None
        for seed in seed_list:
            model, calibrator, threshold, test_probabilities, test_raw = fit_and_predict(model_name, best_params, seed, positive_weight, sets)
            seed_metrics = {}
            for snapshot_period in sorted(test_panel["snapshot"].unique()):
                mask = (test_panel["snapshot"] == snapshot_period).values
                seed_metrics[str(snapshot_period)] = all_metrics(test_labels[mask], test_probabilities[mask], threshold)
            seed_metrics["pooled"] = all_metrics(test_labels, test_probabilities, threshold)
            seed_metrics["threshold"] = threshold
            seed_metrics["calibration_method"] = calibrator.method
            seed_metrics["raw_brier"] = float(brier_score_loss(test_labels, test_raw))
            per_seed.append(seed_metrics)
            if seed == BASE_SEED:
                stored_probabilities = test_probabilities
                stored_threshold = threshold
                if model_name in ("XGBoost", "LightGBM"):
                    model_store = (model, calibrator, columns, sets)
                    if model_name == "LightGBM":
                        lightgbm_store = model_store
                    else:
                        xgboost_store = model_store
        predictions[model_name] = stored_probabilities
        results["models"][model_name] = {
            "best_params": best_params, "validation_auc": best_validation_auc, "trials": trials,
            "per_seed": per_seed, "threshold_seed42": stored_threshold,
            "seconds": time.time() - model_start,
        }
        print(model_name, "val AUC", round(best_validation_auc, 4), "test pooled AUC",
              round(per_seed[0]["pooled"]["AUC"], 4), "time", round(time.time() - model_start), flush=True)
        with open(f"{OUT_DIR}/results_main.json", "w") as handle:
            json.dump(results, handle, indent=1, default=float)

    prediction_frame = pd.DataFrame(predictions)
    prediction_frame.to_parquet(f"{OUT_DIR}/test_predictions.parquet", index=False)

    thresholds = {name: results["models"][name]["threshold_seed42"] for name in model_names}
    bootstrap_results = {}
    for snapshot_period in sorted(test_panel["snapshot"].unique()):
        mask = (test_panel["snapshot"] == snapshot_period).values
        intervals, differences = bootstrap(test_labels[mask], {name: predictions[name][mask] for name in model_names},
                                           thresholds, "LR", N_BOOTSTRAP, BASE_SEED)
        bootstrap_results[str(snapshot_period)] = {"intervals": intervals, "auc_difference_vs_LR": differences}
    intervals, differences = bootstrap(test_labels, {name: predictions[name] for name in model_names},
                                       thresholds, "LR", N_BOOTSTRAP, BASE_SEED)
    bootstrap_results["pooled"] = {"intervals": intervals, "auc_difference_vs_LR": differences}
    results["bootstrap"] = bootstrap_results
    results["total_seconds"] = time.time() - start_time
    with open(f"{OUT_DIR}/results_main.json", "w") as handle:
        json.dump(results, handle, indent=1, default=float)
    print("done in", round(time.time() - start_time), "s")


if __name__ == "__main__":
    main()
