# %% [markdown]
# 
# # 07_03 Final Feature Selection
# 
# ## 목적
# 
# 이 노트북은 **새 Feature를 추가로 생성하는 단계가 아니다.**
# 
# `07_01_Feature.py`에서 생성한 Cycle Feature와  
# `07_02_Feature_validation.py`에서 산출한 검증 결과를 통합하여 다음을 최종 확정한다.
# 
# 1. 어떤 Feature를 **핵심 설명 Feature(KEEP_CORE)** 로 유지할지
# 2. 어떤 Feature를 **보조 Feature(SUPPORT)** 로 유지할지
# 3. 어떤 Feature가 다른 Feature와 사실상 중복되어 **REDUNDANT** 인지
# 4. 물리적 해석 또는 강건성 문제 때문에 **REJECT** 할지
# 5. 통계적으로 후보가 되었더라도 최종 선택하지 않는 **NOT_SELECTED** 인지
# 
# ## 선정 원칙
# 
# 최종 Feature 판정은 단순 AUC 순위가 아니라 다음을 함께 고려한다.
# 
# - Normal Holdout 대비 Abnormal 분리력
# - Holdout False Positive Rate
# - 05 EDA의 물리적 해석
# - Operating-condition dependence
# - Feature 간 중복성
# - Cycle/Physics 설명 단위 적합성
# - Abnormal full-cycle 표본이 8개뿐이라는 한계
# 
# > `07_02`의 자동 shortlist는 **통계 screening 결과**이며 최종 판정 그 자체가 아니다.
# 

# %%

from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 200)

# ------------------------------------------------------------
# 0. Project root / 결과 파일 탐색
# ------------------------------------------------------------
def find_root():
    starts = [Path.cwd(), *Path.cwd().parents]

    # 일반 프로젝트 구조
    for root in starts:
        if (root / "07_Feature_Engineering" / "Data" / "cycle_feature_table_all.csv").exists():
            return root, "project"

    # 07_Feature_Engineering 폴더 안에서 직접 실행하는 경우
    for root in starts:
        if (root / "Data" / "cycle_feature_table_all.csv").exists():
            return root.parent, "project"

    # 업로드/단일 폴더 테스트용 fallback
    for root in starts:
        if (root / "cycle_feature_table_all.csv").exists():
            return root, "flat"

    raise FileNotFoundError("cycle_feature_table_all.csv를 찾지 못했습니다.")


ROOT, MODE = find_root()

if MODE == "project":
    DATA_DIR = ROOT / "07_Feature_Engineering" / "Data"
    EXP_DIR = ROOT / "07_Feature_Engineering" / "Experiments"
    FINAL_DIR = ROOT / "07_Feature_Engineering" / "Final_Selection"
else:
    DATA_DIR = ROOT
    EXP_DIR = ROOT
    FINAL_DIR = ROOT / "Final_Selection"

FINAL_DIR.mkdir(parents=True, exist_ok=True)

print("ROOT      :", ROOT)
print("MODE      :", MODE)
print("DATA_DIR  :", DATA_DIR)
print("EXP_DIR   :", EXP_DIR)
print("FINAL_DIR :", FINAL_DIR)


# %%

# ------------------------------------------------------------
# 1. 07_01 / 07_02 결과 로드
# ------------------------------------------------------------
required_files = {
    "feature_table": DATA_DIR / "cycle_feature_table_all.csv",
    "quality": EXP_DIR / "feature_quality_check.csv",
    "auc": EXP_DIR / "feature_auc_ranking.csv",
    "threshold": EXP_DIR / "feature_threshold_validation.csv",
    "shortlist": EXP_DIR / "feature_shortlist_candidates.csv",
    "correlation": EXP_DIR / "feature_spearman_correlation.csv",
    "high_corr": EXP_DIR / "high_correlation_pairs.csv",
    "phase": EXP_DIR / "phase_feature_ranking.csv",
    "core": EXP_DIR / "core_feature_validation.csv",
    "family": EXP_DIR / "family_best_features.csv",
}

missing = [name for name, path in required_files.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"필수 07 결과 파일 누락: {missing}")

feature_table = pd.read_csv(required_files["feature_table"])
quality = pd.read_csv(required_files["quality"])
auc = pd.read_csv(required_files["auc"])
threshold = pd.read_csv(required_files["threshold"])
shortlist = pd.read_csv(required_files["shortlist"])
high_corr = pd.read_csv(required_files["high_corr"])
phase_ranking = pd.read_csv(required_files["phase"])
core_validation = pd.read_csv(required_files["core"])
family_best = pd.read_csv(required_files["family"])

print("Feature table :", feature_table.shape)
print(feature_table.groupby(["source", "split"]).size())
print("Numeric features :", len(quality))
print("07_02 shortlist :", len(shortlist))


# %% [markdown]
# 
# ## 2. 데이터 무결성 재확인
# 
# `07_03`에서는 새로운 통계량을 다시 만들지 않는다.
# 
# 대신 07_02의 결과를 받아 다음만 확인한다.
# 
# - missing / inf / constant 문제가 없는가
# - Normal Train / Holdout / Abnormal 분리가 유지되는가
# - Abnormal full-cycle이 8개라는 표본 한계를 명시하는가
# 

# %%

# ------------------------------------------------------------
# 2. Integrity check
# ------------------------------------------------------------
quality_issue = quality[
    (quality["missing_count"] > 0)
    | (quality["inf_count"] > 0)
    | (quality["is_constant"])
]

print("Quality issue features:", len(quality_issue))
if len(quality_issue):
    print(quality_issue)

counts = (
    feature_table
    .groupby(["source", "split"])
    .size()
    .rename("n")
    .reset_index()
)

print(counts)

n_abnormal = int((feature_table["source"] == "abnormal").sum())
print("Abnormal full-cycle count:", n_abnormal)

if n_abnormal < 30:
    print(
        "주의: Abnormal full-cycle 표본이 매우 작음. "
        "AUC/hit-rate는 최종 일반화 성능이 아니라 Feature screening evidence로 해석."
    )


# %% [markdown]
# 
# ## 3. 이름 충돌 정리
# 
# 현재 `H1_share / H2_share / H3_share`는 프로젝트의 Hypothesis 1/2/3가 아니다.
# 
# 이 세 변수는 **phase-normalized AI2 Current cycle waveform의 저차 Fourier power composition**이다.
# 
# 원본 CSV 컬럼은 재현성을 위해 그대로 유지하고, `07_03` 최종 표에서는 아래 alias를 함께 제공한다.
# 
# - `H1_share` → `AI2_CycleHarmonic1_Share`
# - `H2_share` → `AI2_CycleHarmonic2_Share`
# - `H3_share` → `AI2_CycleHarmonic3_Share`
# 

# %%

# ------------------------------------------------------------
# 3. Reporting alias
# ------------------------------------------------------------
FEATURE_ALIAS = {
    "H1_share": "AI2_CycleHarmonic1_Share",
    "H2_share": "AI2_CycleHarmonic2_Share",
    "H3_share": "AI2_CycleHarmonic3_Share",
}

alias_table = pd.DataFrame(
    [{"feature": k, "report_name": v} for k, v in FEATURE_ALIAS.items()]
)

print(alias_table)


# %% [markdown]
# 
# ## 4. 07_02 검증결과를 하나의 Master Table로 통합
# 
# 다음 값을 Feature별로 한 행에 묶는다.
# 
# - separation AUC
# - Normal / Abnormal median
# - median shift / Normal IQR
# - Train 1% / 99% threshold
# - Holdout FPR
# - Abnormal hit rate
# - 데이터 품질
# - 자동 shortlist 포함 여부
# 
# 이 Master Table 위에 **05 EDA + 물리적 타당성 기반 최종 판정**을 덧씌운다.
# 

# %%

# ------------------------------------------------------------
# 4. Master validation table
# ------------------------------------------------------------
validation = auc.merge(
    threshold[
        [
            "feature",
            "train_q01",
            "train_q99",
            "holdout_valid_n",
            "holdout_break_n",
            "holdout_fpr",
            "abnormal_valid_n",
            "abnormal_break_n",
            "abnormal_hit_rate",
        ]
    ],
    on="feature",
    how="left",
)

validation = validation.merge(
    quality[
        [
            "feature",
            "missing_count",
            "missing_ratio",
            "inf_count",
            "unique_count",
            "variance",
            "is_constant",
            "high_missing",
        ]
    ],
    on="feature",
    how="left",
)

validation["screening_candidate"] = validation["feature"].isin(shortlist["feature"])
validation["report_name"] = validation["feature"].map(FEATURE_ALIAS).fillna(validation["feature"])

print("Master features:", len(validation))
print(
    validation[
        [
            "feature",
            "report_name",
            "separation_auc",
            "holdout_fpr",
            "abnormal_hit_rate",
            "screening_candidate",
        ]
    ].head(20)
)


# %% [markdown]
# 
# # 5. 최종 Feature 판정 규칙
# 
# 판정은 다음 5단계로 고정한다.
# 
# ### KEEP_CORE
# Cycle / Physics EDA에서 이상을 설명하는 핵심 축으로 유지한다.
# 
# ### SUPPORT
# 핵심 Feature를 보조하거나 시각화 / reason-code 해석에 유용하다.
# 
# ### REDUNDANT
# 유용하지만 대표 Feature와 거의 동일한 정보를 담으므로 핵심 Feature에서 중복 제거한다.
# 
# ### REJECT
# 통계적 분리력이 높아도 물리적 타당성 또는 강건성이 부족하여 최종 Feature로 사용하지 않는다.
# 
# ### NOT_SELECTED
# 문제가 있는 Feature는 아니지만 현재 최종 설명체계에서 굳이 선택하지 않는다.
# 
# > 자동 AUC ranking은 판정을 덮어쓰지 않는다.
# 

# %%

# ------------------------------------------------------------
# 5. Explicit final decision
# ------------------------------------------------------------

# 핵심 cycle / physics 설명 Feature
KEEP_CORE = {
    "timing_deviation": "Normal 반복 timing 대비 상대적 cycle timing 이탈",
    "AI2_Current_rms": "Cycle 단위 Current operating magnitude",
    "AI0_Vibration_rms": "Cycle 단위 AI0 vibration response magnitude",
    "AI1_Vibration_rms": "Cycle 단위 AI1 vibration response magnitude",
    "AI0_abs_current_residual": "Current 조건 대비 AI0 vibration response 이탈 크기",
    "AI1_abs_current_residual": "Current 조건 대비 AI1 vibration response 이탈 크기",
    "PORD": "두 vibration channel의 standardized current-conditioned response 이탈 요약",
    "Shape_RMSE": "Normal AI2 cycle template 대비 waveform structure 이탈",
}

# 핵심을 보조하는 Feature
SUPPORT = {
    "cycle_duration": "timing_deviation의 원시 cycle duration 확인용",
    "AI2_Current_std": "Current cycle 변동성 보조",
    "AI2_Current_ptp": "Current cycle amplitude range 보조",
    "AI2_Current_kurtosis": "Current waveform shape 통계 보조",
    "AI2_Current_crest_factor": "Current waveform peakiness 보조",
    "vibration_normalized_difference": "AI0-AI1 vibration balance를 표현하는 H2 보조 Feature",
    "AI0_current_residual": "H1 residual 방향(sign) 확인용",
    "AI1_current_residual": "H1 residual 방향(sign) 확인용",
    "AI0_response_z": "AI0 response residual 표준화값 / PORD 구성요소",
    "AI1_response_z": "AI1 response residual 표준화값 / PORD 구성요소",
    "Template_Correlation": "Shape_RMSE와 사실상 동일 정보를 주지만 직관적 visualization 보조",
    "H1_share": "AI2 cycle 저차 Fourier composition 보조",
    "H2_share": "AI2 cycle 저차 Fourier composition 보조",
    "H3_share": "AI2 cycle 저차 Fourier composition 보조",
}

# 명백한 중복 Feature
REDUNDANT = {
    "duration_z": "cycle_duration과 Spearman 1.0; timing_deviation/cycle_duration으로 충분",
    "vibration_log_ratio": "vibration_normalized_difference와 Spearman 1.0",
}

# 물리적/강건성 이유로 최종 기각
REJECT = {
    "CNVG": "높은 분리력에도 operating Current 의존성이 남아 독립 gain으로 해석하기 어려움",
    "AI0_Gain": "CNVG 계열 중간 계산값",
    "AI0_GainDev": "CNVG 계열 중간 계산값",
    "AI1_Gain": "CNVG 계열 중간 계산값",
    "AI1_GainDev": "CNVG 계열 중간 계산값",
}

# AI0/AI1 equal-phase vibration features는 05 EDA에서
# Normal phase consistency가 약해 실제 공정 phase feature로 채택하지 않음
for col in validation["feature"]:
    if col.startswith("AI0_Vibration_phase_") or col.startswith("AI1_Vibration_phase_"):
        REJECT[col] = "AI2 phase에 정렬된 vibration local phase consistency가 충분하지 않음"

# AI2 phase-domain feature는 실제 공정 단계 매핑 없이 보조 설명으로만 유지
for col in validation["feature"]:
    if col.startswith("AI2_Current_phase_"):
        SUPPORT[col] = "AI2 phase-domain current profile 보조; 실제 FA/BS/HR 단계로 해석하지 않음"


# %%

# ------------------------------------------------------------
# 6. Decision table 생성
# ------------------------------------------------------------
def final_decision(feature):
    if feature in KEEP_CORE:
        return "KEEP_CORE", KEEP_CORE[feature], ""
    if feature in REDUNDANT:
        return "REDUNDANT", REDUNDANT[feature], ""
    if feature in REJECT:
        return "REJECT", REJECT[feature], ""
    if feature in SUPPORT:
        return "SUPPORT", SUPPORT[feature], ""
    return "NOT_SELECTED", "현재 최종 설명체계의 핵심/보조 Feature로 선택하지 않음", ""


decision_rows = []

for _, row in validation.iterrows():
    feature = row["feature"]
    decision, reason, _ = final_decision(feature)

    decision_rows.append({
        "feature": feature,
        "report_name": FEATURE_ALIAS.get(feature, feature),
        "decision": decision,
        "reason": reason,
        "separation_auc": row["separation_auc"],
        "direction": row["direction"],
        "normal_median": row["normal_median"],
        "abnormal_median": row["abnormal_median"],
        "median_shift_iqr": row["median_shift_iqr"],
        "holdout_fpr": row["holdout_fpr"],
        "abnormal_hit_rate": row["abnormal_hit_rate"],
        "abnormal_break_n": row["abnormal_break_n"],
        "screening_candidate": row["screening_candidate"],
        "missing_ratio": row["missing_ratio"],
        "is_constant": row["is_constant"],
    })

decision_table = pd.DataFrame(decision_rows)

order = {
    "KEEP_CORE": 0,
    "SUPPORT": 1,
    "REDUNDANT": 2,
    "REJECT": 3,
    "NOT_SELECTED": 4,
}

decision_table["_order"] = decision_table["decision"].map(order)
decision_table = decision_table.sort_values(
    ["_order", "separation_auc"],
    ascending=[True, False]
).drop(columns="_order").reset_index(drop=True)

print(
    decision_table[
        [
            "feature",
            "report_name",
            "decision",
            "separation_auc",
            "holdout_fpr",
            "abnormal_hit_rate",
            "reason",
        ]
    ].head(60)
)


# %% [markdown]
# 
# ## 7. Redundancy 최종 확인
# 
# 최종 대표 Feature를 정할 때 특히 다음 중복을 확인한다.
# 
# - `Shape_RMSE ↔ Template_Correlation`
# - `cycle_duration ↔ duration_z`
# - `vibration_normalized_difference ↔ vibration_log_ratio`
# - `AI0_current_residual ↔ AI0_response_z`
# - `AI1_current_residual ↔ AI1_response_z`
# 
# 단, residual raw/z는 용도가 다르므로 둘 다 SUPPORT 정보로 남길 수 있다.
# 
# Shape에서는 `Shape_RMSE`를 대표 핵심값으로 사용하고  
# `Template_Correlation`은 시각화/직관적 보조값으로 유지한다.
# 

# %%

# ------------------------------------------------------------
# 7. 대표 중복 pair 확인
# ------------------------------------------------------------
corr_matrix = pd.read_csv(required_files["correlation"], index_col=0)

check_pairs = [
    ("Shape_RMSE", "Template_Correlation"),
    ("cycle_duration", "duration_z"),
    ("vibration_normalized_difference", "vibration_log_ratio"),
    ("AI0_current_residual", "AI0_response_z"),
    ("AI1_current_residual", "AI1_response_z"),
]

redundancy_check = []

for a, b in check_pairs:
    rho = np.nan
    if a in corr_matrix.index and b in corr_matrix.columns:
        rho = corr_matrix.loc[a, b]

    redundancy_check.append({
        "feature_1": a,
        "feature_2": b,
        "spearman_rho": rho,
    })

redundancy_check = pd.DataFrame(redundancy_check)
print(redundancy_check)


# %% [markdown]
# 
# # 8. 최종 핵심 Feature Set
# 
# `07_03`의 최종 핵심 Cycle / Physics Feature는 아래 구조로 사용한다.
# 
# ### Timing
# - `timing_deviation`
# 
# ### Operating magnitude
# - `AI2_Current_rms`
# 
# ### Vibration response magnitude
# - `AI0_Vibration_rms`
# - `AI1_Vibration_rms`
# 
# ### Current-conditioned response
# - `AI0_abs_current_residual`
# - `AI1_abs_current_residual`
# - `PORD`
# 
# ### Temporal waveform structure
# - `Shape_RMSE`
# 
# 이 Feature들은 **08의 1초 Window RandomForest input과 동일한 테이블을 의미하지 않는다.**
# 
# 07의 Cycle Feature는 주로:
# 
# - EDA 종결
# - 물리적 설명
# - 오류분석
# - Reason code 후보
# 
# 로 사용한다.
# 

# %%

# ------------------------------------------------------------
# 8. Final selected tables
# ------------------------------------------------------------
META_COLS = [
    "source",
    "label",
    "split",
    "segment_id",
    "cycle_id",
    "cycle_start_sec",
    "cycle_end_sec",
]

core_features = [
    f for f in decision_table.loc[
        decision_table["decision"] == "KEEP_CORE", "feature"
    ]
    if f in feature_table.columns
]

support_features = [
    f for f in decision_table.loc[
        decision_table["decision"] == "SUPPORT", "feature"
    ]
    if f in feature_table.columns
]

core_cols = [c for c in META_COLS + core_features if c in feature_table.columns]
explain_cols = [
    c for c in META_COLS + core_features + support_features
    if c in feature_table.columns
]

final_core = feature_table[core_cols].copy()
final_explain = feature_table[explain_cols].copy()

print("KEEP_CORE:", core_features)
print("n KEEP_CORE:", len(core_features))
print("n SUPPORT  :", len(support_features))
print("Core table :", final_core.shape)
print("Explain table:", final_explain.shape)

print(final_core.head())


# %% [markdown]
# 
# # 9. 저장
# 
# 최종 산출물:
# 
# - `feature_decision_table.csv`
# - `final_core_feature_table.csv`
# - `final_explain_feature_table.csv`
# - `feature_name_alias.csv`
# - `final_selection_summary.csv`
# 
# 이 결과를 이후 08 Model Interpretation / Error Analysis의 Cycle-Physics 설명 레이어에서 사용한다.
# 

# %%

# ------------------------------------------------------------
# 9. Save
# ------------------------------------------------------------
decision_path = FINAL_DIR / "feature_decision_table.csv"
core_path = FINAL_DIR / "final_core_feature_table.csv"
explain_path = FINAL_DIR / "final_explain_feature_table.csv"
alias_path = FINAL_DIR / "feature_name_alias.csv"
summary_path = FINAL_DIR / "final_selection_summary.csv"

decision_table.to_csv(decision_path, index=False)
final_core.to_csv(core_path, index=False)
final_explain.to_csv(explain_path, index=False)
alias_table.to_csv(alias_path, index=False)

summary = (
    decision_table["decision"]
    .value_counts()
    .reindex(["KEEP_CORE", "SUPPORT", "REDUNDANT", "REJECT", "NOT_SELECTED"], fill_value=0)
    .rename_axis("decision")
    .reset_index(name="n_features")
)

summary.to_csv(summary_path, index=False)

print(summary)

print("\nSaved:")
for p in [decision_path, core_path, explain_path, alias_path, summary_path]:
    print(" -", p)


# %% [markdown]
# 
# # 10. 최종 해석
# 
# `07_03`의 역할은 다음과 같이 종료한다.
# 
# ```text
# 07_01 Feature Extraction
#         ↓
# 07_02 Statistical Validation
#         ↓
# 07_03 Final Feature Selection
#         ├─ KEEP_CORE
#         ├─ SUPPORT
#         ├─ REDUNDANT
#         ├─ REJECT
#         └─ NOT_SELECTED
#         ↓
# 08 Model Interpretation / Error Analysis
# ```
# 
# 핵심 원칙:
# 
# > **높은 AUC ≠ 자동 채택**
# 
# 최종 Feature는  
# **분리력 + Holdout FPR + 물리적 타당성 + 강건성 + 중복성 + 사용 단위 적합성**으로 결정한다.
# 


