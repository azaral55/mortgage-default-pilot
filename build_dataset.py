"""Build loan-snapshot panel from the Freddie Mac SFLLD sample files (pilot study).

For each observation month s, a loan enters the panel if it is active at s
(row present, unpaid balance > 0, no zero-balance code) and is less than
90 days past due. The target is 1 if the loan becomes 90+ days past due,
reaches REO acquisition, or terminates with a credit-loss event
(zero-balance codes 02, 03, 09, 15) in months s+1 ... s+12.
"""
import sys
import json
import numpy as np
import pandas as pd

DATA_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]
YEARS = [2022, 2023, 2024, 2025, 2026]

ORIG_COLUMNS = [
    "credit_score", "first_payment_date", "first_time_buyer", "maturity_date", "msa",
    "mi_percent", "number_of_units", "occupancy", "orig_cltv", "orig_dti", "orig_upb",
    "orig_ltv", "orig_rate", "channel", "ppm_flag", "amortization_type", "state",
    "property_type", "postal_code", "loan_id", "loan_purpose", "orig_term",
    "number_of_borrowers", "seller", "servicer", "super_conforming", "pre_relief_id",
    "program_indicator", "relief_refi", "valuation_method", "interest_only", "mi_cancel",
]
# Performance file: only the columns that are needed (1-based positions in the guide)
PERF_POSITIONS = {0: "loan_id", 1: "period", 2: "current_upb", 3: "dq_status", 4: "loan_age",
                  7: "modification_flag", 8: "zero_balance_code", 10: "current_rate",
                  25: "estimated_ltv", 29: "assistance_status"}
DEFAULT_ZERO_BALANCE_CODES = {"02", "03", "09", "15"}
SNAPSHOTS = {
    "train": [202212, 202303, 202306],
    "validation": [202309, 202312],
    "test": [202412, 202503],
}


def month_index(yyyymm):
    return (yyyymm // 100) * 12 + (yyyymm % 100) - 1


def load_origination():
    frames = []
    for year in YEARS:
        frame = pd.read_csv(f"{DATA_DIR}/sample_orig_{year}.txt", sep="|", header=None,
                            names=ORIG_COLUMNS, dtype=str)
        frame["vintage"] = year
        frames.append(frame)
    orig = pd.concat(frames, ignore_index=True)
    numeric = ["credit_score", "mi_percent", "number_of_units", "orig_cltv", "orig_dti",
               "orig_upb", "orig_ltv", "orig_rate", "orig_term", "number_of_borrowers",
               "first_payment_date"]
    for column in numeric:
        orig[column] = pd.to_numeric(orig[column], errors="coerce")
    # Freddie Mac special codes for "not available"
    orig.loc[orig["credit_score"] == 9999, "credit_score"] = np.nan
    orig.loc[orig["mi_percent"] == 999, "mi_percent"] = np.nan
    orig.loc[orig["orig_cltv"] == 999, "orig_cltv"] = np.nan
    orig.loc[orig["orig_ltv"] == 999, "orig_ltv"] = np.nan
    orig.loc[orig["orig_dti"] == 999, "orig_dti"] = np.nan
    orig.loc[orig["number_of_units"] == 99, "number_of_units"] = np.nan
    orig.loc[orig["number_of_borrowers"] == 99, "number_of_borrowers"] = np.nan
    for column in ["first_time_buyer", "occupancy", "channel", "property_type", "loan_purpose"]:
        orig.loc[orig[column].isin(["9", "99"]), column] = np.nan
    return orig


def load_performance():
    frames = []
    for year in YEARS:
        frame = pd.read_csv(f"{DATA_DIR}/sample_perf_{year}.txt", sep="|", header=None,
                            usecols=list(PERF_POSITIONS.keys()), dtype=str)
        frame.columns = [PERF_POSITIONS[position] for position in sorted(PERF_POSITIONS)]
        frames.append(frame)
    perf = pd.concat(frames, ignore_index=True)
    perf["period"] = perf["period"].astype(int)
    perf["month"] = month_index(perf["period"])
    status = perf["dq_status"].replace({"RA": "99", "XX": np.nan})
    perf["dq_numeric"] = pd.to_numeric(status, errors="coerce")
    perf["current_upb"] = pd.to_numeric(perf["current_upb"], errors="coerce")
    perf["loan_age"] = pd.to_numeric(perf["loan_age"], errors="coerce")
    perf["current_rate"] = pd.to_numeric(perf["current_rate"], errors="coerce")
    perf["estimated_ltv"] = pd.to_numeric(perf["estimated_ltv"], errors="coerce")
    perf.loc[perf["estimated_ltv"] == 999, "estimated_ltv"] = np.nan
    perf["zero_balance_code"] = perf["zero_balance_code"].fillna("")
    perf["is_default_event"] = (perf["dq_numeric"] >= 3) | perf["zero_balance_code"].isin(DEFAULT_ZERO_BALANCE_CODES)
    return perf


def build_snapshot(perf, orig, snapshot_period, removal_log):
    snapshot_month = month_index(snapshot_period)
    current = perf[perf["month"] == snapshot_month]
    n_rows_at_snapshot = len(current)
    active = current[(current["zero_balance_code"] == "") & (current["current_upb"] > 0)]
    n_inactive = n_rows_at_snapshot - len(active)
    n_unknown_status = int(active["dq_numeric"].isna().sum())
    active = active[active["dq_numeric"].notna()]
    n_seriously_delinquent = int((active["dq_numeric"] >= 3).sum())
    eligible = active[active["dq_numeric"] < 3].copy()

    future = perf[(perf["month"] > snapshot_month) & (perf["month"] <= snapshot_month + 12)]
    default_ids = set(future.loc[future["is_default_event"], "loan_id"])
    eligible["target"] = eligible["loan_id"].isin(default_ids).astype(int)

    history = perf[(perf["month"] > snapshot_month - 12) & (perf["month"] <= snapshot_month)]
    worst = history.groupby("loan_id")["dq_numeric"].max().rename("worst_dq_12m")
    months_delinquent = history.assign(late=history["dq_numeric"] >= 1).groupby("loan_id")["late"].sum().rename("months_delinquent_12m")
    eligible = eligible.merge(worst, left_on="loan_id", right_index=True, how="left")
    eligible = eligible.merge(months_delinquent, left_on="loan_id", right_index=True, how="left")

    merged = eligible.merge(orig, on="loan_id", how="left")
    n_no_origination = int(merged["credit_score"].isna().sum() & 0)  # origination always present in sample
    merged["snapshot"] = snapshot_period
    merged["snapshot_month"] = snapshot_month
    removal_log.append({
        "snapshot": snapshot_period,
        "rows_at_snapshot": int(n_rows_at_snapshot),
        "removed_inactive_or_zero_balance": int(n_inactive),
        "removed_unknown_status": n_unknown_status,
        "removed_already_90dpd": n_seriously_delinquent,
        "kept": int(len(merged)),
        "defaults": int(merged["target"].sum()),
    })
    return merged


def main():
    orig = load_origination()
    perf = load_performance()
    print("performance rows", len(perf), "last period", perf["period"].max())
    removal_log = []
    parts = []
    for split_name, periods in SNAPSHOTS.items():
        for period in periods:
            snapshot = build_snapshot(perf, orig, period, removal_log)
            snapshot["split"] = split_name
            parts.append(snapshot)
    panel = pd.concat(parts, ignore_index=True)
    panel["current_ltv_estimated"] = panel["estimated_ltv"]
    panel["upb_paid_share"] = 1 - panel["current_upb"] / panel["orig_upb"]
    panel["is_modified"] = (panel["modification_flag"].isin(["Y", "P"])).astype(int)
    panel["has_assistance"] = panel["assistance_status"].isin(["F", "R", "T"]).astype(int)
    panel["current_dq"] = panel["dq_numeric"]
    keep_columns = [
        "loan_id", "snapshot", "snapshot_month", "split", "target", "vintage", "state",
        "credit_score", "orig_ltv", "orig_cltv", "orig_dti", "orig_rate", "orig_upb", "orig_term",
        "mi_percent", "number_of_units", "number_of_borrowers", "first_time_buyer", "occupancy",
        "channel", "property_type", "loan_purpose", "loan_age", "current_upb", "current_rate",
        "current_dq", "worst_dq_12m", "months_delinquent_12m", "current_ltv_estimated",
        "upb_paid_share", "is_modified", "has_assistance",
    ]
    panel[keep_columns].to_parquet(f"{OUT_DIR}/panel.parquet", index=False)
    with open(f"{OUT_DIR}/removal_log.json", "w") as handle:
        json.dump(removal_log, handle, indent=1)
    print(pd.DataFrame(removal_log).to_string())
    print(panel.groupby("split")["target"].agg(["size", "sum", "mean"]))


if __name__ == "__main__":
    main()
