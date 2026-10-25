"""Additional analyses for the pilot: ablations, SMOTE vs class weights, per-group results,
SHAP explanation, SHAP-McNemar reduced feature set, error analysis.

Usage: python run_additional.py <out_dir> <fred_csv> <model_name>
"""
import sys
import json
import time
import numpy as np
import pandas as pd
import shap
from scipy.stats import chi2, binomtest
from imblearn.over_sampling import SMOTE

OUT_DIR = sys.argv[1]
FRED_CSV = sys.argv[2]
MODEL_NAME = sys.argv[3]
sys.argv = sys.argv[:3]  # run_experiments reads its own arguments at import time
import run_experiments as base
SEED = 42


def mcnemar_p_value(labels, predicted_full, predicted_reduced):
    correct_full = predicted_full == labels
    correct_reduced = predicted_reduced == labels
    only_full = int((correct_full & ~correct_reduced).sum())
    only_reduced = int((~correct_full & correct_reduced).sum())
    if only_full + only_reduced == 0:
        return 1.0, only_full, only_reduced
    if only_full + only_reduced < 25:
        return float(binomtest(only_full, only_full + only_reduced, 0.5).pvalue), only_full, only_reduced
    statistic = (abs(only_full - only_reduced) - 1) ** 2 / (only_full + only_reduced)
    return float(chi2.sf(statistic, 1)), only_full, only_reduced


def evaluate(model_name, params, positive_weight, panel, numeric_features, categorical_features, smote=False, seed=SEED):
    sets, columns, preprocessor = base.prepare_sets(panel, model_name, numeric_features, categorical_features)
    if smote:
        train_inputs, train_labels = sets["train"]
        # SMOTE needs complete numeric input: fill missing values with training medians first
        if model_name != "LR":
            medians = np.nanmedian(train_inputs, axis=0)
            train_inputs = np.where(np.isnan(train_inputs), medians, train_inputs)
            sets = {name: (np.where(np.isnan(values[0]), medians, values[0]), values[1]) for name, values in sets.items()}
        resampled_inputs, resampled_labels = SMOTE(random_state=seed, sampling_strategy=0.1).fit_resample(train_inputs, train_labels)
        sets["train"] = (resampled_inputs, resampled_labels)
        positive_weight = 1.0
    model, calibrator, threshold, test_probabilities, test_raw = base.fit_and_predict(model_name, params, seed, positive_weight, sets)
    test_labels = sets["test"][1]
    metrics = base.all_metrics(test_labels, test_probabilities, threshold)
    metrics["raw_mean_probability"] = float(test_raw.mean())
    metrics["calibration_method"] = calibrator.method
    return metrics, model, calibrator, threshold, test_probabilities, sets, columns


def main():
    start_time = time.time()
    with open(f"{OUT_DIR}/results_main.json") as handle:
        main_results = json.load(handle)
    positive_weight = main_results["positive_weight"]
    panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
    panel = base.add_macro_features(panel, FRED_CSV)
    full_numeric = base.NUMERIC_FEATURES + base.MACRO_FEATURES
    categorical = base.CATEGORICAL_FEATURES
    params = main_results["models"][MODEL_NAME]["best_params"]
    lr_params = main_results["models"]["LR"]["best_params"]
    output = {"model": MODEL_NAME}

    # 1. Ablation of feature groups (same tuned hyperparameters)
    ablation = {}
    variants = {
        "full": full_numeric,
        "without_macro": base.NUMERIC_FEATURES,
        "without_payment_history": [f for f in full_numeric if f not in base.HISTORY_FEATURES],
        "origination_only": [f for f in full_numeric if f not in base.HISTORY_FEATURES + base.MACRO_FEATURES
                             + ["loan_age", "current_upb", "current_rate", "current_ltv_estimated", "upb_paid_share"]],
    }
    for variant_name, numeric_features in variants.items():
        for model_name, model_params in [(MODEL_NAME, params)]:
            metrics = evaluate(model_name, model_params, positive_weight, panel, numeric_features, categorical)[0]
            ablation[f"{variant_name}|{model_name}"] = metrics
            print("ablation", variant_name, model_name, round(metrics["AUC"], 4), flush=True)
    output["ablation"] = ablation

    # 2. SMOTE vs class weights
    imbalance = {}
    for model_name, model_params in [(MODEL_NAME, params), ("LR", lr_params)]:
        imbalance[f"class_weights|{model_name}"] = evaluate(model_name, model_params, positive_weight, panel, full_numeric, categorical)[0]
        imbalance[f"smote|{model_name}"] = evaluate(model_name, model_params, positive_weight, panel, full_numeric, categorical, smote=True)[0]
        print("imbalance", model_name, round(imbalance[f"smote|{model_name}"]["AUC"], 4), flush=True)
    output["imbalance"] = imbalance
    with open(f"{OUT_DIR}/results_additional_partial.json", "w") as handle:
        json.dump(output, handle, indent=1, default=float)

    # 3. Full model for SHAP, groups and errors
    metrics, model, calibrator, threshold, test_probabilities, sets, columns = evaluate(
        MODEL_NAME, params, positive_weight, panel, full_numeric, categorical)
    test_panel = panel[panel["split"] == "test"].reset_index(drop=True)
    test_labels = sets["test"][1]
    predicted = (test_probabilities >= threshold).astype(int)

    # Per-group results
    groups = {}
    score_band = pd.cut(test_panel["credit_score"], [0, 680, 720, 760, 900], labels=["<680", "680-719", "720-759", ">=760"], right=False)
    test_panel["score_band"] = score_band.astype(str)
    test_panel["vintage_group"] = test_panel["vintage"].astype(str)
    test_panel["ltv_band"] = pd.cut(test_panel["orig_ltv"], [0, 80, 90, 200], labels=["<=80", "80-90", ">90"], right=True).astype(str)
    for group_column in ["score_band", "vintage_group", "ltv_band", "loan_purpose"]:
        for group_value in sorted(test_panel[group_column].dropna().unique()):
            mask = (test_panel[group_column] == group_value).values
            if test_labels[mask].sum() < 10 or (1 - test_labels[mask]).sum() < 10:
                groups[f"{group_column}={group_value}"] = {"n": int(mask.sum()), "defaults": int(test_labels[mask].sum()), "AUC": None}
                continue
            group_metrics = base.all_metrics(test_labels[mask], test_probabilities[mask], threshold)
            group_metrics["n"] = int(mask.sum())
            groups[f"{group_column}={group_value}"] = group_metrics
    output["groups"] = groups

    # Error analysis
    true_positive = (predicted == 1) & (test_labels == 1)
    false_negative = (predicted == 0) & (test_labels == 1)
    false_positive = (predicted == 1) & (test_labels == 0)
    true_negative = (predicted == 0) & (test_labels == 0)
    output["confusion"] = {"TP": int(true_positive.sum()), "FN": int(false_negative.sum()),
                           "FP": int(false_positive.sum()), "TN": int(true_negative.sum()), "threshold": threshold}
    profile_columns = ["credit_score", "orig_ltv", "orig_dti", "current_ltv_estimated", "loan_age", "current_dq",
                       "worst_dq_12m", "has_assistance", "state_unemployment_change_12m"]
    profiles = {}
    for name, mask in [("TP", true_positive), ("FN", false_negative), ("FP", false_positive), ("TN", true_negative)]:
        profiles[name] = test_panel.loc[mask, profile_columns].astype(float).mean().to_dict()
        profiles[name]["share_current_at_snapshot"] = float((test_panel.loc[mask, "current_dq"] == 0).mean())
    output["error_profiles"] = profiles
    fn_probabilities = test_probabilities[false_negative]
    output["false_negative_probability_quantiles"] = {str(q): float(np.quantile(fn_probabilities, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
    output["defaults_current_at_snapshot_share"] = float((test_panel.loc[test_labels == 1, "current_dq"] == 0).mean())
    # Calibration curve data (10 equal-frequency bins)
    order = np.argsort(test_probabilities)
    curve = []
    for bin_indices in np.array_split(order, 10):
        curve.append({"mean_predicted": float(test_probabilities[bin_indices].mean()),
                      "observed": float(test_labels[bin_indices].mean()), "n": int(len(bin_indices))})
    output["calibration_curve"] = curve

    # 4. SHAP values on a random test subsample (raw model output, log-odds)
    random_state = np.random.RandomState(SEED)
    sample_index = random_state.choice(len(test_labels), size=min(2000, len(test_labels)), replace=False)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sets["test"][0][sample_index])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=columns).sort_values(ascending=False)
    output["shap_mean_abs"] = mean_abs.to_dict()
    # Aggregate dummies back to their original variable
    aggregated = {}
    for column, value in mean_abs.items():
        original = column
        for categorical_name in categorical:
            if column.startswith(categorical_name + "_"):
                original = categorical_name
        aggregated[original] = aggregated.get(original, 0.0) + float(value)
    ranking = sorted(aggregated, key=lambda key: -aggregated[key])
    output["shap_ranking"] = [(name, aggregated[name]) for name in ranking]
    with open(f"{OUT_DIR}/results_additional_partial.json", "w") as handle:
        json.dump(output, handle, indent=1, default=float)

    # 5. SHAP-McNemar: shrink the feature list, test on validation, report on test
    full_validation_model = model
    validation_inputs, validation_labels = sets["validation"]
    full_validation_predicted = (calibrator.transform(full_validation_model.predict_proba(validation_inputs)[:, 1]) >= threshold).astype(int)
    reduction = []
    for keep_count in [5, 10, 15, len(ranking)]:
        kept = ranking[:keep_count]
        kept_numeric = [f for f in full_numeric if f in kept]
        kept_categorical = [f for f in categorical if f in kept]
        sets_reduced, _, _ = base.prepare_sets(panel, MODEL_NAME, kept_numeric, kept_categorical)
        reduced_model, reduced_calibrator, reduced_threshold, reduced_test_probabilities, _ = base.fit_and_predict(
            MODEL_NAME, params, SEED, positive_weight, sets_reduced)
        reduced_validation_predicted = (reduced_calibrator.transform(reduced_model.predict_proba(sets_reduced["validation"][0])[:, 1]) >= threshold).astype(int)
        p_value, only_full, only_reduced = mcnemar_p_value(validation_labels, full_validation_predicted, reduced_validation_predicted)
        test_metrics = base.all_metrics(test_labels, reduced_test_probabilities, reduced_threshold)
        reduction.append({"k": keep_count, "features": kept, "validation_mcnemar_p": p_value,
                          "only_full_correct": only_full, "only_reduced_correct": only_reduced,
                          "test_AUC": test_metrics["AUC"], "test_AP": test_metrics["AP"], "test_BSS": test_metrics["BSS"]})
        print("reduction", keep_count, round(p_value, 4), round(test_metrics["AUC"], 4), flush=True)
    output["reduction"] = reduction
    np.save(f"{OUT_DIR}/shap_values.npy", shap_values)
    pd.DataFrame(sets["test"][0][sample_index], columns=columns).to_parquet(f"{OUT_DIR}/shap_inputs.parquet")
    output["seconds"] = time.time() - start_time
    with open(f"{OUT_DIR}/results_additional.json", "w") as handle:
        json.dump(output, handle, indent=1, default=float)
    print("done", round(time.time() - start_time))


if __name__ == "__main__":
    main()
