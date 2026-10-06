from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# ============================================================
# 0. Path / Load
# ============================================================

ROOT = Path.cwd()

for p in [ROOT, *ROOT.parents]:
    if (p / "07_Feature_Engineering" / "Data").exists():
        ROOT = p
        break

DATA_DIR = ROOT / "07_Feature_Engineering" / "Data"
EXP_DIR = ROOT / "07_Feature_Engineering" / "Experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = DATA_DIR / "cycle_feature_table_all.csv"

df = pd.read_csv(FEATURE_PATH)

print("=" * 70)
print("07_02 Feature Validation")
print("=" * 70)
print("Feature data:", FEATURE_PATH)
print("shape:", df.shape)
print(df.groupby(["source", "split"]).size())


# ============================================================
# 1. Feature Column 정의
# ============================================================

META_COLS = [
    "source",
    "label",
    "split",
    "segment_id",
    "cycle_id",
    "cycle_start_sec",
    "cycle_end_sec",
]

feature_cols = [
    c for c in df.columns
    if c not in META_COLS
]

numeric_features = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c])
]

print("Numeric features:", len(numeric_features))


# ============================================================
# 2. 데이터 무결성 검사
# ============================================================

quality_rows = []

for col in numeric_features:
    s = df[col]

    quality_rows.append({
        "feature": col,
        "missing_count": s.isna().sum(),
        "missing_ratio": s.isna().mean(),
        "inf_count": np.isinf(pd.to_numeric(s, errors="coerce")).sum(),
        "unique_count": s.nunique(dropna=True),
        "variance": s.var(skipna=True),
    })


quality_df = pd.DataFrame(quality_rows)

quality_df["is_constant"] = quality_df["unique_count"] <= 1
quality_df["high_missing"] = quality_df["missing_ratio"] > 0.20

display(
    quality_df.sort_values(
        ["high_missing", "is_constant", "missing_ratio"],
        ascending=[False, False, False]
    ).head(30)
)

quality_df.to_csv(
    EXP_DIR / "feature_quality_check.csv",
    index=False
)


# ============================================================
# 3. 비교 대상 정의
# ============================================================

train = df[
    (df["source"] == "normal")
    & (df["split"] == "train")
].copy()

holdout = df[
    (df["source"] == "normal")
    & (df["split"] == "holdout")
].copy()

abnormal = df[
    df["source"] == "abnormal"
].copy()

print("Normal Train   :", len(train))
print("Normal Holdout :", len(holdout))
print("Abnormal       :", len(abnormal))


# ============================================================
# 4. 분포 통계 비교
# ============================================================

summary_rows = []

for col in numeric_features:
    for group_name, group in [
        ("Normal Train", train),
        ("Normal Holdout", holdout),
        ("Abnormal", abnormal),
    ]:
        s = group[col].replace([np.inf, -np.inf], np.nan).dropna()

        if len(s) == 0:
            continue

        summary_rows.append({
            "feature": col,
            "group": group_name,
            "count": len(s),
            "mean": s.mean(),
            "std": s.std(),
            "median": s.median(),
            "q25": s.quantile(.25),
            "q75": s.quantile(.75),
            "min": s.min(),
            "max": s.max(),
        })

distribution_summary = pd.DataFrame(summary_rows)

distribution_summary.to_csv(
    EXP_DIR / "feature_distribution_summary.csv",
    index=False
)

display(distribution_summary.head(20))


# ============================================================
# 5. 단변량 분리력
#    - AUC
#    - separation AUC
#    - median shift
# ============================================================

validation_rows = []

for col in numeric_features:
    n = holdout[col].replace([np.inf, -np.inf], np.nan).dropna()
    a = abnormal[col].replace([np.inf, -np.inf], np.nan).dropna()

    if len(n) == 0 or len(a) == 0:
        continue

    combined = pd.concat([n, a], ignore_index=True)
    labels = np.r_[
        np.zeros(len(n)),
        np.ones(len(a))
    ]

    if combined.nunique() <= 1:
        continue

    auc = roc_auc_score(labels, combined)
    separation_auc = max(auc, 1 - auc)

    normal_median = n.median()
    abnormal_median = a.median()

    iqr = n.quantile(.75) - n.quantile(.25)

    median_shift_iqr = (
        abs(abnormal_median - normal_median) / iqr
        if iqr > 0 else np.nan
    )

    direction = (
        "UP"
        if abnormal_median > normal_median
        else "DOWN"
    )

    validation_rows.append({
        "feature": col,
        "auc_raw": auc,
        "separation_auc": separation_auc,
        "direction": direction,
        "normal_median": normal_median,
        "abnormal_median": abnormal_median,
        "median_shift_iqr": median_shift_iqr,
        "normal_n": len(n),
        "abnormal_n": len(a),
    })

validation_df = pd.DataFrame(validation_rows)

validation_df = validation_df.sort_values(
    ["separation_auc", "median_shift_iqr"],
    ascending=False
).reset_index(drop=True)

validation_df.to_csv(
    EXP_DIR / "feature_auc_ranking.csv",
    index=False
)

display(validation_df.head(30))


# ============================================================
# 6. Train 기준 Threshold 검증
#    양방향 1% / 99%
# ============================================================

threshold_rows = []

for col in numeric_features:
    tr = train[col].replace([np.inf, -np.inf], np.nan).dropna()
    ho = holdout[col].replace([np.inf, -np.inf], np.nan)
    ab = abnormal[col].replace([np.inf, -np.inf], np.nan)

    if len(tr) < 10:
        continue

    low = tr.quantile(.01)
    high = tr.quantile(.99)

    holdout_valid = ho.notna()
    abnormal_valid = ab.notna()

    holdout_break = (
        (ho < low) | (ho > high)
    ) & holdout_valid

    abnormal_break = (
        (ab < low) | (ab > high)
    ) & abnormal_valid

    holdout_fpr = (
        holdout_break.sum() / holdout_valid.sum()
        if holdout_valid.sum() > 0 else np.nan
    )

    abnormal_hit_rate = (
        abnormal_break.sum() / abnormal_valid.sum()
        if abnormal_valid.sum() > 0 else np.nan
    )

    threshold_rows.append({
        "feature": col,
        "train_q01": low,
        "train_q99": high,
        "holdout_valid_n": holdout_valid.sum(),
        "holdout_break_n": holdout_break.sum(),
        "holdout_fpr": holdout_fpr,
        "abnormal_valid_n": abnormal_valid.sum(),
        "abnormal_break_n": abnormal_break.sum(),
        "abnormal_hit_rate": abnormal_hit_rate,
    })

threshold_df = pd.DataFrame(threshold_rows)

threshold_df = threshold_df.sort_values(
    ["abnormal_hit_rate", "holdout_fpr"],
    ascending=[False, True]
).reset_index(drop=True)

threshold_df.to_csv(
    EXP_DIR / "feature_threshold_validation.csv",
    index=False
)

display(threshold_df.head(30))


# ============================================================
# 7. Feature Correlation / Redundancy
# ============================================================

corr_features = [
    c for c in numeric_features
    if train[c].notna().sum() > 10
    and train[c].nunique(dropna=True) > 1
]

corr = train[corr_features].corr(method="spearman")

corr.to_csv(
    EXP_DIR / "feature_spearman_correlation.csv"
)

redundant_rows = []

for i, f1 in enumerate(corr.columns):
    for j in range(i + 1, len(corr.columns)):
        f2 = corr.columns[j]
        rho = corr.iloc[i, j]

        if pd.notna(rho) and abs(rho) >= 0.90:
            redundant_rows.append({
                "feature_1": f1,
                "feature_2": f2,
                "spearman_rho": rho,
                "abs_rho": abs(rho),
            })

redundant_df = pd.DataFrame(redundant_rows)

if not redundant_df.empty:
    redundant_df = redundant_df.sort_values(
        "abs_rho",
        ascending=False
    )

redundant_df.to_csv(
    EXP_DIR / "high_correlation_pairs.csv",
    index=False
)

display(redundant_df.head(30))


# ============================================================
# 8. Phase Feature Ranking
# ============================================================

phase_features = [
    c for c in numeric_features
    if "_phase_" in c
]

phase_ranking = validation_df[
    validation_df["feature"].isin(phase_features)
].copy()

phase_ranking = phase_ranking.merge(
    threshold_df[
        [
            "feature",
            "holdout_fpr",
            "abnormal_hit_rate",
            "abnormal_break_n",
        ]
    ],
    on="feature",
    how="left"
)

phase_ranking = phase_ranking.sort_values(
    ["separation_auc", "abnormal_hit_rate"],
    ascending=False
).reset_index(drop=True)

phase_ranking.to_csv(
    EXP_DIR / "phase_feature_ranking.csv",
    index=False
)

display(phase_ranking.head(20))


# ============================================================
# 9. Core Physics Feature 후보
# ============================================================

CORE_CANDIDATES = [
    "cycle_duration",
    "timing_deviation",
    "duration_z",

    "AI0_Vibration_rms",
    "AI1_Vibration_rms",
    "AI2_Current_rms",

    "AI0_current_residual",
    "AI1_current_residual",

    "AI0_response_z",
    "AI1_response_z",

    "PORD",
    "CNVG",

    "vibration_normalized_difference",
    "vibration_log_ratio",

    "Shape_RMSE",
    "Template_Correlation",

    "H1_share",
    "H2_share",
    "H3_share",
]

core_validation = validation_df[
    validation_df["feature"].isin(CORE_CANDIDATES)
].copy()

core_validation = core_validation.merge(
    threshold_df[
        [
            "feature",
            "holdout_fpr",
            "abnormal_hit_rate",
            "abnormal_break_n",
        ]
    ],
    on="feature",
    how="left"
)

core_validation = core_validation.sort_values(
    ["separation_auc", "abnormal_hit_rate"],
    ascending=False
).reset_index(drop=True)

core_validation.to_csv(
    EXP_DIR / "core_feature_validation.csv",
    index=False
)

display(core_validation)


# ============================================================
# 10. Final Shortlist 자동 후보
# ============================================================

merged = validation_df.merge(
    threshold_df[
        [
            "feature",
            "holdout_fpr",
            "abnormal_hit_rate",
            "abnormal_break_n",
        ]
    ],
    on="feature",
    how="left"
)

# 너무 공격적으로 자르지 않음
candidate = merged[
    (merged["separation_auc"] >= 0.70)
    & (merged["holdout_fpr"] <= 0.10)
].copy()

candidate["score"] = (
    candidate["separation_auc"]
    + candidate["abnormal_hit_rate"].fillna(0)
    - candidate["holdout_fpr"].fillna(0)
)

candidate = candidate.sort_values(
    "score",
    ascending=False
).reset_index(drop=True)

candidate.to_csv(
    EXP_DIR / "feature_shortlist_candidates.csv",
    index=False
)

display(candidate.head(30))


# ============================================================
# 11. Physics Family별 Best Feature
# ============================================================

FEATURE_FAMILIES = {
    "Timing": [
        "cycle_duration",
        "timing_deviation",
        "duration_z",
    ],

    "Current": [
        "AI2_Current_rms",
        "AI2_Current_std",
        "AI2_Current_ptp",
        "AI2_Current_energy",
    ],

    "Vibration": [
        "AI0_Vibration_rms",
        "AI1_Vibration_rms",
        "AI0_Vibration_std",
        "AI1_Vibration_std",
        "AI0_Vibration_ptp",
        "AI1_Vibration_ptp",
    ],

    "Coupling": [
        "AI0_current_residual",
        "AI1_current_residual",
        "AI0_response_z",
        "AI1_response_z",
        "PORD",
        "CNVG",
    ],

    "Balance": [
        "vibration_normalized_difference",
        "vibration_log_ratio",
    ],

    "Shape": [
        "Shape_RMSE",
        "Template_Correlation",
        "H1_share",
        "H2_share",
        "H3_share",
    ],
}

family_rows = []

for family, features in FEATURE_FAMILIES.items():
    sub = merged[
        merged["feature"].isin(features)
    ].copy()

    if sub.empty:
        continue

    sub = sub.sort_values(
        ["separation_auc", "abnormal_hit_rate"],
        ascending=False
    )

    best = sub.iloc[0]

    family_rows.append({
        "family": family,
        "feature": best["feature"],
        "separation_auc": best["separation_auc"],
        "direction": best["direction"],
        "holdout_fpr": best["holdout_fpr"],
        "abnormal_hit_rate": best["abnormal_hit_rate"],
        "abnormal_break_n": best["abnormal_break_n"],
    })

family_best = pd.DataFrame(family_rows)

family_best.to_csv(
    EXP_DIR / "family_best_features.csv",
    index=False
)

display(family_best)


# ============================================================
# 12. Top Feature 시각화
# ============================================================

top_features = (
    validation_df
    .head(12)["feature"]
    .tolist()
)

for col in top_features:
    n = holdout[col].replace([np.inf, -np.inf], np.nan).dropna()
    a = abnormal[col].replace([np.inf, -np.inf], np.nan).dropna()

    if len(n) == 0 or len(a) == 0:
        continue

    plt.figure(figsize=(6, 4))

    plt.boxplot(
        [n, a],
        tick_labels=[
            "Normal Holdout",
            "Abnormal"
        ]
    )

    plt.title(col)
    plt.ylabel("Feature value")
    plt.grid(axis="y", alpha=.25)
    plt.tight_layout()

    safe_name = col.replace("/", "_")

    plt.savefig(
        EXP_DIR / f"box_{safe_name}.png",
        dpi=150
    )

    plt.show()


# ============================================================
# 13. Summary
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDATION COMPLETE")
print("=" * 70)

print("\nTop 15 Features:")
print(
    validation_df[
        [
            "feature",
            "separation_auc",
            "direction",
            "median_shift_iqr",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nFamily Best:")
print(
    family_best.to_string(
        index=False
    )
)

print("\nSaved to:")
print(EXP_DIR)
