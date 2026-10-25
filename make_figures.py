"""Vector figures for the Results section (PDF)."""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

OUT_DIR = sys.argv[1]
FIG_DIR = sys.argv[2]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "lines.linewidth": 1.1, "legend.fontsize": 7, "figure.dpi": 150})
COLORS = {"LR": "#1f4e79", "RF": "#7f7f7f", "XGBoost": "#c55a11", "LightGBM": "#2e7d32"}
MODELS = ["LR", "RF", "XGBoost", "LightGBM"]

predictions = pd.read_parquet(f"{OUT_DIR}/test_predictions.parquet")
labels = predictions["target"].values
with open(f"{OUT_DIR}/results_main.json") as handle:
    main_results = json.load(handle)
import os
extra = None
if os.path.exists(f"{OUT_DIR}/results_additional.json"):
    with open(f"{OUT_DIR}/results_additional.json") as handle:
        extra = json.load(handle)

# Figure 1: ROC and precision-recall curves on the pooled test set
figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
for model_name in MODELS:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, predictions[model_name])
    auc_value = main_results["models"][model_name]["per_seed"][0]["pooled"]["AUC"]
    axes[0].plot(false_positive_rate, true_positive_rate, color=COLORS[model_name], label=f"{model_name} ({auc_value:.3f})")
    precision, recall, _ = precision_recall_curve(labels, predictions[model_name])
    ap_value = main_results["models"][model_name]["per_seed"][0]["pooled"]["AP"]
    axes[1].plot(recall, precision, color=COLORS[model_name], label=f"{model_name} ({ap_value:.3f})")
axes[0].plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=0.7)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("(a) ROC curve, legend: AUC")
axes[0].legend(loc="lower right", frameon=False)
axes[1].axhline(labels.mean(), color="black", linestyle=":", linewidth=0.7)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("(b) Precision-recall curve, legend: AP")
axes[1].legend(loc="upper right", frameon=False)
for axis in axes:
    axis.grid(alpha=0.25, linewidth=0.4)
figure.tight_layout()
figure.savefig(f"{FIG_DIR}/fig_roc_pr.pdf")

# Figure 2: reliability diagram (10 equal-frequency bins)
figure, axis = plt.subplots(figsize=(3.4, 2.9))
for model_name in MODELS:
    probabilities = predictions[model_name].values
    order = np.argsort(probabilities)
    mean_predicted = []
    observed = []
    for bin_indices in np.array_split(order, 10):
        mean_predicted.append(probabilities[bin_indices].mean() * 100)
        observed.append(labels[bin_indices].mean() * 100)
    axis.plot(mean_predicted, observed, marker="o", markersize=2.5, color=COLORS[model_name], label=model_name)
upper = 13
axis.plot([0.05, upper], [0.05, upper], color="black", linestyle=":", linewidth=0.7, label="Perfect calibration")
axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xlabel("Mean predicted default probability (%)")
axis.set_ylabel("Observed default rate (%)")
axis.legend(frameon=False, loc="upper left")
axis.grid(alpha=0.25, linewidth=0.4, which="both")
figure.tight_layout()
figure.savefig(f"{FIG_DIR}/fig_calibration.pdf")

if extra is None:
    print("main figures only"); sys.exit(0)
# Figure 3: SHAP importance and the feature-reduction curve
ranking = extra["shap_ranking"][:12]
figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
names = [item[0] for item in ranking][::-1]
values = [item[1] for item in ranking][::-1]
axes[0].barh(names, values, color="#2e7d32" if extra["model"] == "LightGBM" else "#c55a11")
axes[0].set_xlabel("Mean |SHAP value| (log-odds)")
axes[0].set_title(f"(a) Top 12 features, {extra['model']}")
reduction = extra["reduction"]
counts = [row["k"] for row in reduction]
aucs = [row["test_AUC"] for row in reduction]
axes[1].plot(counts, aucs, marker="o", markersize=3, color="black")
full_auc = aucs[-1]
axes[1].axhline(full_auc - 0.01, color="#c00000", linestyle="--", linewidth=0.7, label="Full model AUC - 0.01")
for row in reduction:
    if row["validation_mcnemar_p"] < 0.05:
        axes[1].plot(row["k"], row["test_AUC"], marker="x", color="#c00000", markersize=6)
axes[1].set_xlabel("Number of features kept")
axes[1].set_ylabel("Test AUC")
axes[1].set_title("(b) Reduced feature sets (x: McNemar p < 0.05)")
axes[1].legend(frameon=False, loc="lower right")
for axis in axes:
    axis.grid(alpha=0.25, linewidth=0.4)
figure.tight_layout()
figure.savefig(f"{FIG_DIR}/fig_shap_reduction.pdf")

# Figure 4: per-group AUC and default rate
groups = extra["groups"]
order_keys = [key for key in groups if key.startswith("score_band")] + \
             [key for key in groups if key.startswith("vintage_group")] + \
             [key for key in groups if key.startswith("ltv_band")]
order_keys = [key for key in order_keys if groups[key].get("AUC") is not None]
pretty = []
for key in order_keys:
    column, value = key.split("=", 1)
    prefix = {"score_band": "Score ", "vintage_group": "Vintage ", "ltv_band": "LTV "}[column]
    pretty.append(prefix + value)
figure, axis = plt.subplots(figsize=(7.0, 2.5))
positions = np.arange(len(order_keys))
axis.bar(positions, [groups[key]["AUC"] for key in order_keys], color="#7f9fbf", width=0.6)
axis.set_ylim(0.6, 1.0)
axis.set_ylabel("Test AUC")
axis.set_xticks(positions)
axis.set_xticklabels(pretty, rotation=30, ha="right")
second = axis.twinx()
second.plot(positions, [groups[key]["ActualDefaults"] / groups[key]["n"] * 100 for key in order_keys],
            color="#c00000", marker="o", markersize=3, linewidth=0.8)
second.set_ylabel("Observed default rate (%)", color="#c00000")
axis.grid(alpha=0.25, linewidth=0.4, axis="y")
figure.tight_layout()
figure.savefig(f"{FIG_DIR}/fig_groups.pdf")
print("figures written")
