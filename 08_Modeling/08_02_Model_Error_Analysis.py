# %% [markdown]
# # 08_02 Model Error Analysis
# 
# ## 목적
# 
# `08_01_model_comparison.py`에서 이미 선정된 두 모델의 예측 결과를
# **재학습 없이** 분석한다.
# 
# - **A: RandomForest** (지도학습)
# - **B: Mahalanobis** (Normal-only 이상탐지)
# 
# 결과는 다음 구조로 저장한다.
# 
# ```text
# results/Error_analysis/
# ├─ A_랜덤포레스트/
# ├─ B_마할라노비스/
# └─ C_비교/
# ```
# 
# ### 분석 원칙
# 
# - 모델 재학습 / 재선정 없음
# - Test 결과를 threshold 선정에 사용하지 않음
# - 07 Cycle Feature는 predictor가 아니라 explanation context로 사용
# - Reason Code는 모델 내부 인과 설명이 아니라 EDA 기반 post-hoc 해석
# 

"""
08_02 Model Error Analysis
==========================

목적
----
08_01_model_comparison.py에서 이미 선정된 두 모델의 예측 결과를
재학습 없이 분석한다.

A: RandomForest  (지도학습)
B: Mahalanobis   (Normal-only 이상탐지)

출력 구조
---------
results/Error_analysis/
├─ A_랜덤포레스트/
├─ B_마할라노비스/
└─ C_비교/

주의
----
- 모델 재학습 / 재선정 없음
- Test 결과를 threshold 선정에 사용하지 않음
- 07 Cycle Feature는 predictor가 아니라 explanation context로 사용
- Reason Code는 모델 내부 인과 설명이 아니라 EDA 기반 post-hoc 해석
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 0. Config

# %%
MODEL_INFO = {
    "A": {
        "folder": "A_RandomForest",
        "expected_model": "RandomForest",
    },
    "B": {
        "folder": "B_Mahalanobis",
        "expected_model": "Mahalanobis",
    },
}

KEY = ["source", "segment_id", "source_row"]

MANIFEST_FEATURES = [
    "RMS_Detected",
    "Outlier_Category",
    "H1_Residual_AI0",
    "H1_Residual_AI1",
    "H2_ND",
    "H3_Current_Amplitude",
]

REASON_COLS = [
    "R1_AMPLITUDE",
    "R2_CURRENT_VIBRATION_RELATION",
    "R3_VIBRATION_BALANCE",
    "R4_CURRENT_OPERATING",
]

CYCLE_FEATURES = [
    "timing_deviation",
    "AI2_Current_rms",
    "AI0_abs_current_residual",
    "AI1_abs_current_residual",
    "PORD",
    "Shape_RMSE",
]

EPS = 1e-12

# %% [markdown]
# ## 1. Project Root / Path

# %%
def find_project_root():
    """
    정식 repo 구조와 단일 폴더 테스트 구조를 모두 지원한다.
    """

    starts = [Path.cwd()]

    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass

    candidates = []

    for start in starts:
        candidates.extend([start, *start.parents])

    # 중복 제거
    seen = set()
    roots = []

    for p in candidates:
        p = p.resolve()
        if p not in seen:
            seen.add(p)
            roots.append(p)

    # 1) 정식 프로젝트 구조
    for root in roots:
        if (
            (root / "08_Modeling" / "results" / "test_predictions_A.csv").exists()
            and (root / "08_Modeling" / "results" / "test_predictions_B.csv").exists()
            and (root / "data" / "10_04_manifest_all.csv").exists()
        ):
            return root, "project"

    # 2) 스크립트가 08_Modeling 안에 있는 경우
    for root in roots:
        if (
            (root / "results" / "test_predictions_A.csv").exists()
            and (root / "results" / "test_predictions_B.csv").exists()
            and (root.parent / "data" / "10_04_manifest_all.csv").exists()
        ):
            return root.parent, "project"

    # 3) 단일 폴더 테스트
    for root in roots:
        if (
            (root / "test_predictions_A.csv").exists()
            and (root / "test_predictions_B.csv").exists()
            and (root / "10_04_manifest_all.csv").exists()
        ):
            return root, "flat"

    raise FileNotFoundError(
        "프로젝트 root를 찾지 못했습니다. "
        "test_predictions_A/B.csv와 10_04_manifest_all.csv 위치를 확인하세요."
    )


ROOT, MODE = find_project_root()

if MODE == "project":
    MODEL_RESULT_DIR = ROOT / "08_Modeling" / "results"
    MANIFEST_PATH = ROOT / "data" / "10_04_manifest_all.csv"

    FINAL07_DIR = ROOT / "07_Feature_Engineering" / "Final_Selection"
    EXP07_DIR = ROOT / "07_Feature_Engineering" / "Experiments"

    ERROR_ROOT = MODEL_RESULT_DIR / "Error_analysis"

else:
    MODEL_RESULT_DIR = ROOT
    MANIFEST_PATH = ROOT / "10_04_manifest_all.csv"

    FINAL07_DIR = ROOT / "Final_Selection"
    EXP07_DIR = ROOT

    ERROR_ROOT = ROOT / "results" / "Error_analysis"


A_DIR = ERROR_ROOT / MODEL_INFO["A"]["folder"]
B_DIR = ERROR_ROOT / MODEL_INFO["B"]["folder"]
C_DIR = ERROR_ROOT / "C_Comparison"

for p in [A_DIR, B_DIR, C_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Utility

# %%
def require(path, name):
    if not path.exists():
        raise FileNotFoundError(f"{name} 없음: {path}")
    return path


def load_model_config():
    path = require(
        MODEL_RESULT_DIR / "model_config.json",
        "model_config.json",
    )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(model_code):
    test_path = require(
        MODEL_RESULT_DIR / f"test_predictions_{model_code}.csv",
        f"test_predictions_{model_code}.csv",
    )

    oof_path = require(
        MODEL_RESULT_DIR / f"oof_predictions_{model_code}.csv",
        f"oof_predictions_{model_code}.csv",
    )

    test = pd.read_csv(test_path)
    oof = pd.read_csv(oof_path)

    test["eval_set"] = "TEST"
    oof["eval_set"] = "OOF"

    return pd.concat([oof, test], ignore_index=True)


def validate_prediction_table(df, model_code):
    required_cols = [
        *KEY,
        "pos_in_seg",
        "seg_len",
        "label",
        "split",
        "anomaly_score",
        "alarm",
        "eval_set",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"{model_code} prediction 필수 컬럼 누락: {missing}"
        )

    dup = df.duplicated(["eval_set", *KEY]).sum()

    if dup:
        raise ValueError(
            f"{model_code} prediction key 중복: {dup}"
        )


def error_type(label, alarm):
    if label == 1 and alarm == 1:
        return "TP"
    if label == 0 and alarm == 1:
        return "FP"
    if label == 1 and alarm == 0:
        return "FN"
    if label == 0 and alarm == 0:
        return "TN"
    return "UNKNOWN"


def reason_string(row):
    reasons = [
        c for c in REASON_COLS
        if bool(row[c])
    ]

    return "|".join(reasons) if reasons else "R0_UNEXPLAINED"

# %% [markdown]
# ## 3. Load Common Data

# %%
config = load_model_config()

manifest = pd.read_csv(
    require(MANIFEST_PATH, "10_04_manifest_all.csv")
)

if manifest.duplicated(KEY).sum():
    raise ValueError(
        "10_04_manifest_all.csv의 공식 key가 중복되어 있습니다."
    )

decision07 = pd.read_csv(
    require(
        FINAL07_DIR / "feature_decision_table.csv",
        "feature_decision_table.csv",
    )
)

core07 = pd.read_csv(
    require(
        FINAL07_DIR / "final_core_feature_table.csv",
        "final_core_feature_table.csv",
    )
)

threshold07 = pd.read_csv(
    require(
        EXP07_DIR / "feature_threshold_validation.csv",
        "feature_threshold_validation.csv",
    )
)

# %% [markdown]
# ## 4. Selected Model Validation

# %%
for model_code, info in MODEL_INFO.items():

    selected = config["selected"][model_code]

    if selected["model"] != info["expected_model"]:
        raise RuntimeError(
            f"{model_code} selected model 불일치: "
            f"expected={info['expected_model']}, actual={selected['model']}"
        )

    if selected["features"] != "F4_+H1잔차":
        raise RuntimeError(
            f"{model_code} selected feature set이 F4가 아닙니다: {selected}"
        )

print("=" * 72)
print("08_02 MODEL ERROR ANALYSIS")
print("=" * 72)
print("ROOT :", ROOT)
print("MODE :", MODE)
print("A    :", config["selected"]["A"])
print("B    :", config["selected"]["B"])
print("OUT  :", ERROR_ROOT)


# ============================================================
# 5. Common Reason Threshold
#    Model-independent EDA explanation threshold
# ============================================================

# A/B prediction key는 동일해야 한다.
pred_A_raw = load_predictions("A")
pred_B_raw = load_predictions("B")

validate_prediction_table(pred_A_raw, "A")
validate_prediction_table(pred_B_raw, "B")

key_check_cols = [
    "eval_set",
    *KEY,
    "label",
    "split",
]

key_compare = pred_A_raw[key_check_cols].merge(
    pred_B_raw[key_check_cols],
    on=key_check_cols,
    how="outer",
    indicator=True,
)

if (key_compare["_merge"] != "both").any():
    raise RuntimeError(
        "A와 B prediction의 평가 window key가 서로 다릅니다."
    )

# Reason threshold는 모델과 독립적인 EDA 기준으로 고정한다.
# OOF Normal window를 사용하고 Test는 threshold 계산에 쓰지 않는다.
normal_oof_keys = pred_A_raw[
    (pred_A_raw["eval_set"] == "OOF")
    & (pred_A_raw["label"] == 0)
][KEY]

normal_oof_manifest = normal_oof_keys.merge(
    manifest,
    on=KEY,
    how="left",
    validate="one_to_one",
)

REASON_THRESHOLDS = {
    "H1_AI0_Q99": float(
        normal_oof_manifest["H1_Residual_AI0"].quantile(.99)
    ),
    "H1_AI1_Q99": float(
        normal_oof_manifest["H1_Residual_AI1"].quantile(.99)
    ),
    "H2_Q01": float(
        normal_oof_manifest["H2_ND"].quantile(.01)
    ),
    "H2_Q99": float(
        normal_oof_manifest["H2_ND"].quantile(.99)
    ),
    "CURRENT_Q01": float(
        normal_oof_manifest["H3_Current_Amplitude"].quantile(.01)
    ),
    "CURRENT_Q99": float(
        normal_oof_manifest["H3_Current_Amplitude"].quantile(.99)
    ),
}

reason_thresholds_df = pd.DataFrame(
    [
        {"metric": k, "value": v}
        for k, v in REASON_THRESHOLDS.items()
    ]
)

reason_thresholds_df.to_csv(
    C_DIR / "common_reason_thresholds.csv",
    index=False,
)

# %% [markdown]
# ## 6. Cycle / Physics Context

# %%
thr07 = (
    threshold07[
        threshold07["feature"].isin(CYCLE_FEATURES)
    ][
        ["feature", "train_q01", "train_q99"]
    ]
    .set_index("feature")
)

missing_thr = [
    feature
    for feature in CYCLE_FEATURES
    if feature not in thr07.index
]

if missing_thr:
    raise RuntimeError(
        f"07 Cycle threshold 누락: {missing_thr}"
    )


def outside_07_support(series, feature):
    low = thr07.loc[feature, "train_q01"]
    high = thr07.loc[feature, "train_q99"]

    return (series < low) | (series > high)


cycle = core07.copy()

cycle["C1_TIMING"] = outside_07_support(
    cycle["timing_deviation"],
    "timing_deviation",
)

cycle["C2_CURRENT_SUPPORT"] = outside_07_support(
    cycle["AI2_Current_rms"],
    "AI2_Current_rms",
)

cycle["C3_CURRENT_VIBRATION_RESPONSE"] = (
    outside_07_support(
        cycle["AI0_abs_current_residual"],
        "AI0_abs_current_residual",
    )
    | outside_07_support(
        cycle["AI1_abs_current_residual"],
        "AI1_abs_current_residual",
    )
    | outside_07_support(
        cycle["PORD"],
        "PORD",
    )
)

cycle["C4_SHAPE"] = outside_07_support(
    cycle["Shape_RMSE"],
    "Shape_RMSE",
)

cycle_context = (
    cycle.groupby(
        ["source", "segment_id"],
        as_index=False,
    )
    .agg(
        cycle_count=("cycle_id", "size"),
        cycle_timing_break=("C1_TIMING", "max"),
        cycle_current_break=("C2_CURRENT_SUPPORT", "max"),
        cycle_response_break=(
            "C3_CURRENT_VIBRATION_RESPONSE",
            "max",
        ),
        cycle_shape_break=("C4_SHAPE", "max"),
        max_PORD=("PORD", "max"),
        max_Shape_RMSE=("Shape_RMSE", "max"),
        max_timing_deviation=("timing_deviation", "max"),
    )
)

cycle_context["cycle_context_available"] = True

cycle_context.to_csv(
    C_DIR / "common_cycle_context_by_segment.csv",
    index=False,
)

# %% [markdown]
# ## 7. Model Analysis

# %%
def add_reason_codes(pred):
    out = pred.copy()

    out["R1_AMPLITUDE"] = (
        out["RMS_Detected"]
        .fillna(False)
        .astype(bool)
    )

    out["R2_CURRENT_VIBRATION_RELATION"] = (
        (
            out["H1_Residual_AI0"]
            > REASON_THRESHOLDS["H1_AI0_Q99"]
        )
        | (
            out["H1_Residual_AI1"]
            > REASON_THRESHOLDS["H1_AI1_Q99"]
        )
    )

    out["R3_VIBRATION_BALANCE"] = (
        (
            out["H2_ND"]
            < REASON_THRESHOLDS["H2_Q01"]
        )
        | (
            out["H2_ND"]
            > REASON_THRESHOLDS["H2_Q99"]
        )
    )

    out["R4_CURRENT_OPERATING"] = (
        (
            out["H3_Current_Amplitude"]
            < REASON_THRESHOLDS["CURRENT_Q01"]
        )
        | (
            out["H3_Current_Amplitude"]
            > REASON_THRESHOLDS["CURRENT_Q99"]
        )
    )

    out["reason_count"] = out[REASON_COLS].sum(axis=1)

    out["reason_codes"] = out.apply(
        reason_string,
        axis=1,
    )

    return out


def aggregate_segment(pred):
    seg = (
        pred.groupby(
            [
                "eval_set",
                "source",
                "segment_id",
                "label",
            ],
            as_index=False,
        )
        .agg(
            n_windows=("alarm", "size"),
            n_alarm=("alarm", "sum"),
            alarm_ratio=("alarm", "mean"),
            max_score=("anomaly_score", "max"),
            mean_score=("anomaly_score", "mean"),
            n_reason_amp=("R1_AMPLITUDE", "sum"),
            n_reason_relation=(
                "R2_CURRENT_VIBRATION_RELATION",
                "sum",
            ),
            n_reason_balance=(
                "R3_VIBRATION_BALANCE",
                "sum",
            ),
            n_reason_current=(
                "R4_CURRENT_OPERATING",
                "sum",
            ),
        )
    )

    seg["any_alarm"] = seg["n_alarm"] > 0

    seg["segment_result"] = np.select(
        [
            (seg["label"] == 1) & seg["any_alarm"],
            (seg["label"] == 0) & seg["any_alarm"],
            (seg["label"] == 1) & ~seg["any_alarm"],
            (seg["label"] == 0) & ~seg["any_alarm"],
        ],
        [
            "DETECTED_ABNORMAL",
            "FALSE_ALARM_NORMAL",
            "MISSED_ABNORMAL",
            "CLEAN_NORMAL",
        ],
        default="UNKNOWN",
    )

    return seg


def add_cycle_context(segment_summary):
    out = segment_summary.merge(
        cycle_context,
        on=["source", "segment_id"],
        how="left",
    )

    out["cycle_context_available"] = (
        out["cycle_context_available"].eq(True)
    )

    cycle_bool_cols = [
        "cycle_timing_break",
        "cycle_current_break",
        "cycle_response_break",
        "cycle_shape_break",
    ]

    for col in cycle_bool_cols:
        out[col] = out[col].eq(True)

    return out


def make_score_plot(pred, model_code, out_dir):
    test = pred[pred["eval_set"] == "TEST"]

    plt.figure(figsize=(8, 5))

    plt.hist(
        [
            test.loc[
                test["label"] == 0,
                "anomaly_score",
            ],
            test.loc[
                test["label"] == 1,
                "anomaly_score",
            ],
        ],
        bins=30,
        alpha=.65,
        label=["Normal", "Abnormal"],
    )

    plt.axvline(
        config["selected"][model_code]["threshold"],
        linestyle="--",
        label="Selected threshold",
    )

    plt.xlabel("Anomaly score")
    plt.ylabel("Window count")
    plt.title(
        f"{model_code} TEST Anomaly Score Distribution"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        out_dir / "test_score_distribution.png",
        dpi=150,
    )

    plt.close()


def analyze_model(model_code):
    info = MODEL_INFO[model_code]

    out_dir = ERROR_ROOT / info["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(model_code)
    validate_prediction_table(pred, model_code)

    pred = pred.merge(
        manifest,
        on=KEY,
        how="left",
        validate="many_to_one",
    )

    # 모든 window는 10-sample evaluable이므로 manifest 핵심 feature가 있어야 함
    missing_manifest = pred[MANIFEST_FEATURES].isna().all(axis=1).sum()

    if missing_manifest:
        raise RuntimeError(
            f"{model_code}: manifest 미매칭 window = {missing_manifest}"
        )

    pred["error_type"] = [
        error_type(label, alarm)
        for label, alarm
        in zip(pred["label"], pred["alarm"])
    ]

    pred = add_reason_codes(pred)

    error_summary = (
        pred.groupby(
            ["eval_set", "error_type"]
        )
        .size()
        .rename("n_windows")
        .reset_index()
    )

    reason_summary = (
        pred.groupby(
            [
                "eval_set",
                "error_type",
                "reason_codes",
            ]
        )
        .size()
        .rename("n_windows")
        .reset_index()
    )

    segment_summary = aggregate_segment(pred)
    segment_with_cycle = add_cycle_context(
        segment_summary
    )

    fp = pred[
        pred["error_type"] == "FP"
    ].copy()

    fn = pred[
        pred["error_type"] == "FN"
    ].copy()

    # FN이 속한 segment 전체 context
    fn_segments = (
        fn[
            ["eval_set", "source", "segment_id"]
        ]
        .drop_duplicates()
    )

    fn_context_rows = []

    for _, row in fn_segments.iterrows():
        g = pred[
            (pred["eval_set"] == row["eval_set"])
            & (pred["source"] == row["source"])
            & (pred["segment_id"] == row["segment_id"])
        ]

        fn_context_rows.append({
            "eval_set": row["eval_set"],
            "source": row["source"],
            "segment_id": row["segment_id"],
            "n_windows": len(g),
            "n_alarm": int(g["alarm"].sum()),
            "alarm_ratio": float(g["alarm"].mean()),
            "min_score": float(g["anomaly_score"].min()),
            "max_score": float(g["anomaly_score"].max()),
            "mean_score": float(g["anomaly_score"].mean()),
        })

    fn_segment_context = pd.DataFrame(
        fn_context_rows
    )

    # Save
    pred.to_csv(
        out_dir / "window_predictions_with_reasons.csv",
        index=False,
    )

    error_summary.to_csv(
        out_dir / "error_summary.csv",
        index=False,
    )

    reason_summary.to_csv(
        out_dir / "reason_code_summary.csv",
        index=False,
    )

    segment_summary.to_csv(
        out_dir / "segment_summary.csv",
        index=False,
    )

    segment_with_cycle.to_csv(
        out_dir / "segment_with_cycle_context.csv",
        index=False,
    )

    fp.to_csv(
        out_dir / "false_positive_windows.csv",
        index=False,
    )

    fn.to_csv(
        out_dir / "false_negative_windows.csv",
        index=False,
    )

    fn_segment_context.to_csv(
        out_dir / "fn_segment_context.csv",
        index=False,
    )

    make_score_plot(
        pred,
        model_code,
        out_dir,
    )

    return {
        "pred": pred,
        "segment": segment_with_cycle,
        "error_summary": error_summary,
        "reason_summary": reason_summary,
    }

# %% [markdown]
# ## 8. Run A / B

# %%
result_A = analyze_model("A")
result_B = analyze_model("B")

# %% [markdown]
# ## 9. A vs B Window Comparison

# %%
def comparison_category(label, alarm_A, alarm_B):
    if label == 1:
        if alarm_A == 1 and alarm_B == 1:
            return "BOTH_DETECT"
        if alarm_A == 1 and alarm_B == 0:
            return "A_ONLY_DETECT"
        if alarm_A == 0 and alarm_B == 1:
            return "B_ONLY_DETECT"
        return "BOTH_MISS"

    if alarm_A == 0 and alarm_B == 0:
        return "BOTH_CLEAN"
    if alarm_A == 1 and alarm_B == 0:
        return "A_ONLY_FP"
    if alarm_A == 0 and alarm_B == 1:
        return "B_ONLY_FP"
    return "BOTH_FP"


A_cols = [
    "eval_set",
    *KEY,
    "pos_in_seg",
    "seg_len",
    "label",
    "split",
    "anomaly_score",
    "alarm",
    "error_type",
    "reason_codes",
    *REASON_COLS,
]

B_cols = [
    "eval_set",
    *KEY,
    "anomaly_score",
    "alarm",
    "error_type",
]

window_cmp = result_A["pred"][A_cols].merge(
    result_B["pred"][B_cols],
    on=["eval_set", *KEY],
    how="inner",
    suffixes=("_A", "_B"),
    validate="one_to_one",
)

# label/split 등은 A쪽 값 기준
window_cmp["comparison_result"] = [
    comparison_category(
        label,
        alarm_A,
        alarm_B,
    )
    for label, alarm_A, alarm_B in zip(
        window_cmp["label"],
        window_cmp["alarm_A"],
        window_cmp["alarm_B"],
    )
]

window_cmp["models_agree"] = (
    window_cmp["alarm_A"]
    == window_cmp["alarm_B"]
)

window_cmp["A_correct"] = (
    window_cmp["alarm_A"]
    == window_cmp["label"]
)

window_cmp["B_correct"] = (
    window_cmp["alarm_B"]
    == window_cmp["label"]
)

window_comparison_summary = (
    window_cmp.groupby(
        ["eval_set", "label", "comparison_result"]
    )
    .size()
    .rename("n_windows")
    .reset_index()
)

window_disagreement = window_cmp[
    ~window_cmp["models_agree"]
].copy()

# %% [markdown]
# ## 10. A vs B Segment Comparison

# %%
seg_A = result_A["segment"].copy()
seg_B = result_B["segment"].copy()

segment_cmp = seg_A[
    [
        "eval_set",
        "source",
        "segment_id",
        "label",
        "n_windows",
        "n_alarm",
        "alarm_ratio",
        "max_score",
        "mean_score",
        "any_alarm",
        "segment_result",
        "cycle_context_available",
        "cycle_timing_break",
        "cycle_current_break",
        "cycle_response_break",
        "cycle_shape_break",
        "max_PORD",
        "max_Shape_RMSE",
        "max_timing_deviation",
    ]
].merge(
    seg_B[
        [
            "eval_set",
            "source",
            "segment_id",
            "label",
            "n_alarm",
            "alarm_ratio",
            "max_score",
            "mean_score",
            "any_alarm",
            "segment_result",
        ]
    ],
    on=[
        "eval_set",
        "source",
        "segment_id",
        "label",
    ],
    how="inner",
    suffixes=("_A", "_B"),
    validate="one_to_one",
)

segment_cmp["comparison_result"] = [
    comparison_category(
        label,
        int(alarm_A),
        int(alarm_B),
    )
    for label, alarm_A, alarm_B in zip(
        segment_cmp["label"],
        segment_cmp["any_alarm_A"],
        segment_cmp["any_alarm_B"],
    )
]

segment_cmp["models_agree"] = (
    segment_cmp["any_alarm_A"]
    == segment_cmp["any_alarm_B"]
)

segment_comparison_summary = (
    segment_cmp.groupby(
        ["eval_set", "label", "comparison_result"]
    )
    .size()
    .rename("n_segments")
    .reset_index()
)

segment_disagreement = segment_cmp[
    ~segment_cmp["models_agree"]
].copy()

# %% [markdown]
# ## 11. Error overlap comparison

# %%
# Normal: A/B FP overlap
normal_cmp = window_cmp[
    window_cmp["label"] == 0
].copy()

abnormal_cmp = window_cmp[
    window_cmp["label"] == 1
].copy()

error_overlap_summary = pd.DataFrame([
    {
        "eval_set": eval_set,
        "BOTH_FP": int(
            (
                (normal_cmp["eval_set"] == eval_set)
                & (normal_cmp["comparison_result"] == "BOTH_FP")
            ).sum()
        ),
        "A_ONLY_FP": int(
            (
                (normal_cmp["eval_set"] == eval_set)
                & (normal_cmp["comparison_result"] == "A_ONLY_FP")
            ).sum()
        ),
        "B_ONLY_FP": int(
            (
                (normal_cmp["eval_set"] == eval_set)
                & (normal_cmp["comparison_result"] == "B_ONLY_FP")
            ).sum()
        ),
        "BOTH_MISS": int(
            (
                (abnormal_cmp["eval_set"] == eval_set)
                & (abnormal_cmp["comparison_result"] == "BOTH_MISS")
            ).sum()
        ),
        "A_ONLY_DETECT": int(
            (
                (abnormal_cmp["eval_set"] == eval_set)
                & (abnormal_cmp["comparison_result"] == "A_ONLY_DETECT")
            ).sum()
        ),
        "B_ONLY_DETECT": int(
            (
                (abnormal_cmp["eval_set"] == eval_set)
                & (abnormal_cmp["comparison_result"] == "B_ONLY_DETECT")
            ).sum()
        ),
        "BOTH_DETECT": int(
            (
                (abnormal_cmp["eval_set"] == eval_set)
                & (abnormal_cmp["comparison_result"] == "BOTH_DETECT")
            ).sum()
        ),
    }
    for eval_set in ["OOF", "TEST"]
])

# %% [markdown]
# ## 12. Save Comparison

# %%
window_cmp.to_csv(
    C_DIR / "window_A_vs_B.csv",
    index=False,
)

window_comparison_summary.to_csv(
    C_DIR / "window_comparison_summary.csv",
    index=False,
)

window_disagreement.to_csv(
    C_DIR / "window_disagreement.csv",
    index=False,
)

segment_cmp.to_csv(
    C_DIR / "segment_A_vs_B.csv",
    index=False,
)

segment_comparison_summary.to_csv(
    C_DIR / "segment_comparison_summary.csv",
    index=False,
)

segment_disagreement.to_csv(
    C_DIR / "segment_disagreement.csv",
    index=False,
)

error_overlap_summary.to_csv(
    C_DIR / "error_overlap_summary.csv",
    index=False,
)

# %% [markdown]
# ## 13. Final Console Summary

# %%
def print_model_summary(model_code, result):
    model_name = MODEL_INFO[model_code]["expected_model"]

    print("\n" + "=" * 72)
    print(f"{model_code} | {model_name}")
    print("=" * 72)

    for eval_set in ["OOF", "TEST"]:
        sub = result["pred"][
            result["pred"]["eval_set"] == eval_set
        ]

        cnt = (
            sub["error_type"]
            .value_counts()
            .reindex(
                ["TP", "FP", "FN", "TN"],
                fill_value=0,
            )
        )

        print(f"\n[{eval_set}]")
        print(cnt.to_string())

        fp_seg = (
            sub[
                sub["error_type"] == "FP"
            ]["segment_id"]
            .value_counts()
            .to_dict()
        )

        fn_seg = (
            sub[
                sub["error_type"] == "FN"
            ]["segment_id"]
            .value_counts()
            .to_dict()
        )

        print("FP segments:", fp_seg)
        print("FN segments:", fn_seg)


print_model_summary("A", result_A)
print_model_summary("B", result_B)

print("\n" + "=" * 72)
print("C | A vs B 비교")
print("=" * 72)

print("\nWindow comparison:")
print(
    window_comparison_summary.to_string(
        index=False
    )
)

print("\nSegment comparison:")
print(
    segment_comparison_summary.to_string(
        index=False
    )
)

print("\nError overlap:")
print(
    error_overlap_summary.to_string(
        index=False
    )
)

print("\nSaved:")
print("A:", A_DIR)
print("B:", B_DIR)
print("C:", C_DIR)

print("\n08_02 MODEL ERROR ANALYSIS COMPLETE")


