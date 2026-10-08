
# AI2로 추출한 정상 Cycle 전체에서 AI0/AI1 진동 응답을 분석한다.
# Cycle별 RMS, STD, PTP의 분포와 시간에 따른 변화를 확인한다.
# AI2 Current 수준과 AI0/AI1 진동 크기의 관계도 함께 확인한다.
# 아직 Phase를 나누지 않고 전체 Cycle 단위의 Global Response만 본다.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 04_Data_Processing/preprocess.py에서 만든 공식 전처리 데이터를 직접 불러온다. (01_Cycle.py와 같은 방식)
from pathlib import Path
import pandas as pd

def find_data_path():
    for root in [Path.cwd(), *Path.cwd().parents]:
        for p in (root/"data"/"processed"/"preprocessed_data.csv", root/"data"/"preprocessed_data.csv"):
            if p.exists():
                return p
    raise FileNotFoundError("data/(processed/)preprocessed_data.csv를 찾지 못함")

DATA_PATH=find_data_path()
data=pd.read_csv(DATA_PATH,parse_dates=["TimeStamp"])

normal=data[data["source"]=="normal"].copy()
outlier=data[data["source"]=="abnormal"].copy()

# 01_Cycle.py / 03_operating_regime.py와 동일한 기준값
MIN_SEGMENT_ROWS=30
MIN_PEAK_DISTANCE=12

# 01_Cycle.py와 동일한 AI2 Peak-to-Peak 기준으로 Normal cycle을 재생성한다.
# cycle_features는 cycle별 통계 Feature Table, cycle_bounds는 normal의 행 경계(index) Table이다.
# 중복행 제거로 normal의 index와 source_row가 후반부에서 달라지므로 실제 index를 따로 저장한다.
from scipy.signal import find_peaks

CHANNELS=["AI0_Vibration","AI1_Vibration","AI2_Current"]
CYCLE_ANCHOR="AI2_Current"
CYCLE_MIN_PEAK_DISTANCE_SAMPLES=12

cycle_records=[]
bounds_records=[]

for segment_id,group in normal.groupby("segment_id",sort=False):
    if len(group)<MIN_SEGMENT_ROWS:
        continue

    anchor=group[CYCLE_ANCHOR].to_numpy(float)
    smooth=pd.Series(anchor).rolling(3,center=True,min_periods=1).median().to_numpy()
    prominence=max(float(np.std(smooth,ddof=1))*.5,np.finfo(float).eps)
    peaks,_=find_peaks(
        smooth,
        distance=CYCLE_MIN_PEAK_DISTANCE_SAMPLES,
        prominence=prominence
    )

    for cycle_id,(start,stop) in enumerate(zip(peaks[:-1],peaks[1:]),start=1):
        if stop-start<CYCLE_MIN_PEAK_DISTANCE_SAMPLES:
            continue

        cycle=group.iloc[start:stop+1]
        duration=float(cycle["elapsed_sec"].iloc[-1]-cycle["elapsed_sec"].iloc[0])
        if duration<=0:
            continue

        record={
            "segment_id":segment_id,
            "cycle_number_in_segment":cycle_id,
            "start_source_row":int(cycle["source_row"].iloc[0]),
            "end_source_row":int(cycle["source_row"].iloc[-1]),
            "duration_sec":duration,
            "n_samples":len(cycle)
        }

        for channel in CHANNELS:
            x=cycle[channel].to_numpy(float)
            mean_square=float(np.mean(x**2))
            rms=float(np.sqrt(mean_square))

            record[f"{channel}_mean"]=float(np.mean(x))
            record[f"{channel}_std"]=float(np.std(x,ddof=1))
            record[f"{channel}_rms"]=rms
            record[f"{channel}_peak_to_peak"]=float(np.ptp(x))
            record[f"{channel}_mean_square"]=mean_square
            record[f"{channel}_energy"]=float(np.sum(x**2))
            record[f"{channel}_crest_factor"]=float(np.max(np.abs(x))/rms) if rms>0 else np.nan

        cycle_records.append(record)
        bounds_records.append({
            "segment_id":segment_id,
            "cycle_id":cycle_id,
            "start_idx":int(cycle.index[0]),
            "end_idx":int(cycle.index[-1]),
            "start_source_row":int(cycle["source_row"].iloc[0]),
            "end_source_row":int(cycle["source_row"].iloc[-1])
        })

cycle_features=pd.DataFrame(cycle_records)
cycle_bounds=pd.DataFrame(bounds_records)

if cycle_features.empty:
    raise ValueError("Normal Cycle Candidate가 생성되지 않았습니다.")
if len(cycle_features)!=len(cycle_bounds):
    raise RuntimeError("cycle_features와 cycle_bounds 행 수가 일치하지 않습니다.")

print("cycle_features:",cycle_features.shape)
print("cycle_bounds:",cycle_bounds.shape)
print("Cycle duration median:",cycle_features["duration_sec"].median())

global_features = cycle_features.copy()

# Cycle 시작 시점을 실제 Normal elapsed time으로 연결
elapsed_map = normal.set_index("source_row")["elapsed_sec"].to_dict()
global_features["start_elapsed_sec"] = (
    global_features["start_source_row"].map(elapsed_map)
)

FEATURES = [
    "AI0_Vibration_rms",
    "AI1_Vibration_rms",
    "AI2_Current_rms",
    "AI0_Vibration_std",
    "AI1_Vibration_std",
    "AI0_Vibration_peak_to_peak",
    "AI1_Vibration_peak_to_peak",
]

# 1. Global 요약 통계
summary = global_features[FEATURES].describe(
    percentiles=[0.25, 0.5, 0.75]
).T[["mean", "std", "25%", "50%", "75%", "min", "max"]]

summary["cv"] = summary["std"] / summary["mean"].abs()

print("=== Global Cycle Response Summary ===")
print(summary.round(6))

# 2. 핵심 관계 확인
corr_pairs = {
    "AI0 RMS ↔ AI0 STD":
        global_features["AI0_Vibration_rms"].corr(
            global_features["AI0_Vibration_std"]
        ),

    "AI1 RMS ↔ AI1 STD":
        global_features["AI1_Vibration_rms"].corr(
            global_features["AI1_Vibration_std"]
        ),

    "AI2 RMS ↔ AI0 RMS":
        global_features["AI2_Current_rms"].corr(
            global_features["AI0_Vibration_rms"]
        ),

    "AI2 RMS ↔ AI1 RMS":
        global_features["AI2_Current_rms"].corr(
            global_features["AI1_Vibration_rms"]
        ),

    "AI0 RMS ↔ AI1 RMS":
        global_features["AI0_Vibration_rms"].corr(
            global_features["AI1_Vibration_rms"]
        ),
}

print("\n=== Global Correlations ===")
for name, value in corr_pairs.items():
    print(f"{name:22s}: {value:.4f}")

# 3. 시간에 따른 AI0/AI1 Cycle RMS
plt.figure(figsize=(12, 5))
plt.scatter(
    global_features["start_elapsed_sec"],
    global_features["AI0_Vibration_rms"],
    s=18,
    alpha=0.65,
    label="AI0 RMS"
)
plt.scatter(
    global_features["start_elapsed_sec"],
    global_features["AI1_Vibration_rms"],
    s=18,
    alpha=0.65,
    label="AI1 RMS"
)
plt.xlabel("Elapsed time (s)")
plt.ylabel("Cycle RMS")
plt.title("Normal Cycle-level Vibration RMS over Time")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# 4. AI0 / AI1 RMS 분포
plt.figure(figsize=(9, 5))
plt.hist(
    global_features["AI0_Vibration_rms"],
    bins=30,
    alpha=0.6,
    label="AI0 RMS"
)
plt.hist(
    global_features["AI1_Vibration_rms"],
    bins=30,
    alpha=0.6,
    label="AI1 RMS"
)
plt.xlabel("Cycle RMS")
plt.ylabel("Count")
plt.title("Distribution of Normal Cycle-level Vibration RMS")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# 5. AI2 Current RMS 대비 AI0 진동 RMS
plt.figure(figsize=(7, 5))
plt.scatter(
    global_features["AI2_Current_rms"],
    global_features["AI0_Vibration_rms"],
    alpha=0.65
)
plt.xlabel("AI2 Current RMS")
plt.ylabel("AI0 Vibration RMS")
plt.title("AI2 Current RMS vs AI0 Vibration RMS")
plt.grid(alpha=0.25)
plt.show()

# 6. AI2 Current RMS 대비 AI1 진동 RMS
plt.figure(figsize=(7, 5))
plt.scatter(
    global_features["AI2_Current_rms"],
    global_features["AI1_Vibration_rms"],
    alpha=0.65
)
plt.xlabel("AI2 Current RMS")
plt.ylabel("AI1 Vibration RMS")
plt.title("AI2 Current RMS vs AI1 Vibration RMS")
plt.grid(alpha=0.25)
plt.show()

# 7. AI0 ↔ AI1 Cycle Response
plt.figure(figsize=(7, 5))
plt.scatter(
    global_features["AI0_Vibration_rms"],
    global_features["AI1_Vibration_rms"],
    alpha=0.65
)
plt.xlabel("AI0 Vibration RMS")
plt.ylabel("AI1 Vibration RMS")
plt.title("AI0 vs AI1 Cycle-level RMS")
plt.grid(alpha=0.25)
plt.show()


# Normal Cycle에서 AI2 RMS와 AI0/AI1 RMS의 Global 관계를 회귀로 정량화한다.
# AI2 수준에서 기대되는 진동 RMS를 계산하고 실제값과의 Residual을 만든다.
# Residual의 크기와 시간 변화를 통해 단순 진폭과 관계 이탈을 분리한다.
# 이 단계는 이후 Current-Vibration Coupling Feature의 기준이 된다.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

gf = global_features.copy()

x = gf[["AI2_Current_rms"]].to_numpy()

results = {}

for channel in ["AI0", "AI1"]:
    y_col = f"{channel}_Vibration_rms"
    y = gf[y_col].to_numpy()

    model = LinearRegression()
    model.fit(x, y)

    pred = model.predict(x)
    residual = y - pred

    gf[f"{channel}_pred_rms"] = pred
    gf[f"{channel}_residual"] = residual

    results[channel] = {
        "slope": model.coef_[0],
        "intercept": model.intercept_,
        "r2": r2_score(y, pred),
        "residual_mean": residual.mean(),
        "residual_std": residual.std(ddof=1)
    }

print("=== AI2 RMS → Vibration RMS Global Regression ===")
for channel, r in results.items():
    print(
        f"{channel}: "
        f"slope={r['slope']:.6f}, "
        f"intercept={r['intercept']:.6f}, "
        f"R²={r['r2']:.4f}, "
        f"residual std={r['residual_std']:.6f}"
    )

# AI2 → AI0
plt.figure(figsize=(7, 5))
plt.scatter(
    gf["AI2_Current_rms"],
    gf["AI0_Vibration_rms"],
    alpha=0.6,
    label="Observed"
)

order = np.argsort(gf["AI2_Current_rms"].to_numpy())

plt.plot(
    gf["AI2_Current_rms"].to_numpy()[order],
    gf["AI0_pred_rms"].to_numpy()[order],
    linewidth=2,
    label="Normal regression"
)

plt.xlabel("AI2 Current RMS")
plt.ylabel("AI0 Vibration RMS")
plt.title(f"AI2 RMS → AI0 RMS (R²={results['AI0']['r2']:.3f})")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# AI2 → AI1
plt.figure(figsize=(7, 5))
plt.scatter(
    gf["AI2_Current_rms"],
    gf["AI1_Vibration_rms"],
    alpha=0.6,
    label="Observed"
)

plt.plot(
    gf["AI2_Current_rms"].to_numpy()[order],
    gf["AI1_pred_rms"].to_numpy()[order],
    linewidth=2,
    label="Normal regression"
)

plt.xlabel("AI2 Current RMS")
plt.ylabel("AI1 Vibration RMS")
plt.title(f"AI2 RMS → AI1 RMS (R²={results['AI1']['r2']:.3f})")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# AI0 Residual 시간 변화
plt.figure(figsize=(12, 4))
plt.scatter(
    gf["start_elapsed_sec"],
    gf["AI0_residual"],
    s=18,
    alpha=0.65
)
plt.axhline(0, linewidth=1)
plt.xlabel("Elapsed time (s)")
plt.ylabel("AI0 RMS Residual")
plt.title("AI0 Current-conditioned Residual over Time")
plt.grid(alpha=0.25)
plt.show()

# AI1 Residual 시간 변화
plt.figure(figsize=(12, 4))
plt.scatter(
    gf["start_elapsed_sec"],
    gf["AI1_residual"],
    s=18,
    alpha=0.65
)
plt.axhline(0, linewidth=1)
plt.xlabel("Elapsed time (s)")
plt.ylabel("AI1 RMS Residual")
plt.title("AI1 Current-conditioned Residual over Time")
plt.grid(alpha=0.25)
plt.show()

# Residual 분포 비교
plt.figure(figsize=(9, 5))
plt.hist(gf["AI0_residual"], bins=30, alpha=0.6, label="AI0 residual")
plt.hist(gf["AI1_residual"], bins=30, alpha=0.6, label="AI1 residual")
plt.xlabel("Observed RMS - Expected RMS")
plt.ylabel("Count")
plt.title("Normal Current-conditioned Vibration Residual")
plt.legend()
plt.grid(alpha=0.25)
plt.show()


# Global Current-Vibration 관계가 Low-RMS Normal과 다른 Normal에서 같은지 비교한다.
# 기존 3600~4300초 구간은 원인 미확정의 Low-RMS candidate로만 사용한다.
# 각 regime별 AI2→AI0/AI1 회귀선, R², residual bias를 비교한다.
# 결과를 통해 단순 진폭 변화와 coupling 관계 변화를 구분한다.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

gf_regime = gf.copy()

LOW_START = 3600
LOW_END = 4300

gf_regime["regime"] = np.where(
    gf_regime["start_elapsed_sec"].between(LOW_START, LOW_END),
    "Low-RMS",
    "Other"
)

def fit_regime(df, y_col):
    x = df[["AI2_Current_rms"]].to_numpy()
    y = df[y_col].to_numpy()

    model = LinearRegression().fit(x, y)
    pred = model.predict(x)

    return {
        "n": len(df),
        "slope": model.coef_[0],
        "intercept": model.intercept_,
        "r2": r2_score(y, pred),
        "residual_mean": np.mean(y - pred),
        "residual_std": np.std(y - pred, ddof=1),
        "model": model
    }

results_regime = {}

for channel in ["AI0", "AI1"]:
    y_col = f"{channel}_Vibration_rms"

    results_regime[channel] = {
        regime: fit_regime(group, y_col)
        for regime, group in gf_regime.groupby("regime")
    }

print("=== Regime-specific Global Regression ===")

for channel in ["AI0", "AI1"]:
    print(f"\n[{channel}]")

    for regime in ["Other", "Low-RMS"]:
        r = results_regime[channel][regime]

        print(
            f"{regime:8s} | "
            f"n={r['n']:3d} | "
            f"slope={r['slope']:.6f} | "
            f"intercept={r['intercept']:.6f} | "
            f"R²={r['r2']:.4f} | "
            f"resid std={r['residual_std']:.6f}"
        )

# AI0 regime별 회귀
plt.figure(figsize=(8, 5))

for regime in ["Other", "Low-RMS"]:
    d = gf_regime[gf_regime["regime"] == regime]
    r = results_regime["AI0"][regime]

    plt.scatter(
        d["AI2_Current_rms"],
        d["AI0_Vibration_rms"],
        alpha=0.55,
        label=f"{regime} observed"
    )

    x_line = np.linspace(
        d["AI2_Current_rms"].min(),
        d["AI2_Current_rms"].max(),
        100
    ).reshape(-1, 1)

    plt.plot(
        x_line[:, 0],
        r["model"].predict(x_line),
        linewidth=2,
        label=f"{regime} fit"
    )

plt.xlabel("AI2 Current RMS")
plt.ylabel("AI0 Vibration RMS")
plt.title("Regime-specific AI2 RMS → AI0 RMS")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# AI1 regime별 회귀
plt.figure(figsize=(8, 5))

for regime in ["Other", "Low-RMS"]:
    d = gf_regime[gf_regime["regime"] == regime]
    r = results_regime["AI1"][regime]

    plt.scatter(
        d["AI2_Current_rms"],
        d["AI1_Vibration_rms"],
        alpha=0.55,
        label=f"{regime} observed"
    )

    x_line = np.linspace(
        d["AI2_Current_rms"].min(),
        d["AI2_Current_rms"].max(),
        100
    ).reshape(-1, 1)

    plt.plot(
        x_line[:, 0],
        r["model"].predict(x_line),
        linewidth=2,
        label=f"{regime} fit"
    )

plt.xlabel("AI2 Current RMS")
plt.ylabel("AI1 Vibration RMS")
plt.title("Regime-specific AI2 RMS → AI1 RMS")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# 기존 Global residual을 regime별로 비교
residual_summary = (
    gf_regime.groupby("regime")
    [["AI0_residual", "AI1_residual"]]
    .agg(["mean", "std", "median"])
)

print("\n=== Global-model Residual by Regime ===")
print(residual_summary.round(6))


# Low-RMS Normal이 AI2-Vibration 관계의 단순 offset인지 slope 변화인지 확인한다.
# AI2만, AI2+Regime, AI2+Regime+Interaction 세 모델을 비교한다.
# Low-RMS 내부 AI2 범위가 좁으므로 별도 회귀 slope를 직접 해석하지 않는다.
# 이 셀을 마지막 Global 진단으로 사용하고 이후 Phase 분석으로 넘어간다.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

check = gf_regime.copy()
check["low_flag"] = (check["regime"] == "Low-RMS").astype(int)

print("=== AI2 RMS Range by Regime ===")
range_summary = (
    check.groupby("regime")["AI2_Current_rms"]
    .agg(["count", "min", "max", "mean", "std"])
)
print(range_summary.round(4))

def compare_models(df, y_col):
    y = df[y_col].to_numpy()

    X1 = df[["AI2_Current_rms"]].to_numpy()

    X2 = df[
        ["AI2_Current_rms", "low_flag"]
    ].to_numpy()

    X3 = df[
        ["AI2_Current_rms", "low_flag"]
    ].copy()

    X3["interaction"] = (
        df["AI2_Current_rms"] * df["low_flag"]
    )

    X3 = X3.to_numpy()

    models = {}

    for name, X in {
        "AI2 only": X1,
        "AI2 + Regime": X2,
        "AI2 + Regime + Interaction": X3
    }.items():

        model = LinearRegression().fit(X, y)
        pred = model.predict(X)

        models[name] = {
            "R2": r2_score(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "coef": model.coef_,
            "intercept": model.intercept_
        }

    return models

all_results = {}

for channel in ["AI0", "AI1"]:
    y_col = f"{channel}_Vibration_rms"
    all_results[channel] = compare_models(check, y_col)

    print(f"\n=== {channel} Model Comparison ===")

    for name, r in all_results[channel].items():
        print(
            f"{name:28s} | "
            f"R²={r['R2']:.4f} | "
            f"RMSE={r['RMSE']:.6f}"
        )

    r2 = all_results[channel]["AI2 + Regime"]
    r3 = all_results[channel]["AI2 + Regime + Interaction"]

    print(
        f"\n{channel} Regime offset coefficient : "
        f"{r2['coef'][1]:.6f}"
    )

    print(
        f"{channel} Interaction coefficient    : "
        f"{r3['coef'][2]:.6f}"
    )

# Global model residual의 regime 차이
print("\n=== Global Residual Difference ===")

for channel in ["AI0", "AI1"]:
    col = f"{channel}_residual"

    low = check.loc[check["regime"] == "Low-RMS", col]
    other = check.loc[check["regime"] == "Other", col]

    pooled_std = np.sqrt(
        (
            (len(low)-1) * low.var(ddof=1)
            + (len(other)-1) * other.var(ddof=1)
        )
        / (len(low)+len(other)-2)
    )

    effect_size = (
        low.mean() - other.mean()
    ) / pooled_std

    print(
        f"{channel}: "
        f"Low mean={low.mean():.6f}, "
        f"Other mean={other.mean():.6f}, "
        f"Cohen d={effect_size:.3f}"
    )


# AI2 Peak-to-Peak Cycle을 상대시간 기준 A/B/C 3개 Phase로 나눈다.
# 각 Phase에서 AI0/AI1 진동의 RMS, STD, PTP, 평균절대값을 계산한다.
# 전체 Cycle RMS로 나눈 relative RMS도 만들어 amplitude와 phase 구조를 분리한다.
# Cycle 마지막 peak는 다음 Cycle과 중복되므로 Phase 계산에서는 제외한다.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE_NAMES = ["A", "B", "C"]
CHANNELS_PHASE = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]

phase_records = []

for _, bound in cycle_bounds.iterrows():
    segment_id = bound["segment_id"]
    cycle_id = bound["cycle_id"]

    cycle = normal.loc[
        int(bound["start_idx"]):int(bound["end_idx"])
    ].copy()

    # 다음 Cycle의 첫 peak와 중복되는 마지막 endpoint 제거
    if len(cycle) < 4:
        continue

    cycle_core = cycle.iloc[:-1].copy()

    # Cycle 전체 기준 통계
    cycle_rms = {}
    for channel in CHANNELS_PHASE:
        x = cycle_core[channel].to_numpy(float)
        cycle_rms[channel] = np.sqrt(np.mean(x**2))

    # 실제 샘플을 보간하지 않고 3개 상대구간으로 분리
    phase_indices = np.array_split(
        np.arange(len(cycle_core)),
        3
    )

    for phase_name, idx in zip(PHASE_NAMES, phase_indices):
        phase = cycle_core.iloc[idx]

        record = {
            "segment_id": segment_id,
            "cycle_id": cycle_id,
            "phase": phase_name,
            "n_samples": len(phase),
            "cycle_start_elapsed_sec": float(
                cycle_core["elapsed_sec"].iloc[0]
            )
        }

        for channel in CHANNELS_PHASE:
            x = phase[channel].to_numpy(float)

            rms = np.sqrt(np.mean(x**2))

            record[f"{channel}_mean"] = np.mean(x)
            record[f"{channel}_std"] = (
                np.std(x, ddof=1) if len(x) > 1 else np.nan
            )
            record[f"{channel}_rms"] = rms
            record[f"{channel}_ptp"] = np.ptp(x)
            record[f"{channel}_mean_abs"] = np.mean(np.abs(x))
            record[f"{channel}_rms_ratio"] = (
                rms / cycle_rms[channel]
                if cycle_rms[channel] > 0
                else np.nan
            )

        phase_records.append(record)

phase_features = pd.DataFrame(phase_records)

print("=== Phase Extraction ===")
print(f"Cycles          : {phase_features[['segment_id','cycle_id']].drop_duplicates().shape[0]}")
print(f"Phase rows      : {len(phase_features)}")
print("\nSamples per phase:")
print(
    phase_features.groupby("phase")["n_samples"]
    .agg(["min", "median", "max"])
)

def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

# Phase별 Raw RMS + Relative RMS 요약
for channel in ["AI0_Vibration", "AI1_Vibration"]:
    summary = (
        phase_features.groupby("phase")
        .agg(
            rms_median=(f"{channel}_rms", "median"),
            rms_q25=(f"{channel}_rms", q25),
            rms_q75=(f"{channel}_rms", q75),
            ratio_median=(f"{channel}_rms_ratio", "median"),
            ratio_q25=(f"{channel}_rms_ratio", q25),
            ratio_q75=(f"{channel}_rms_ratio", q75),
        )
        .reindex(PHASE_NAMES)
    )

    print(f"\n=== {channel} Phase Summary ===")
    print(summary.round(6))

    med = summary["ratio_median"]
    print(
        f"Relative RMS max/min median ratio: "
        f"{med.max() / med.min():.3f}"
    )

# ---------------------------
# AI0 Raw Phase RMS
# ---------------------------
plt.figure(figsize=(8, 5))
plt.boxplot(
    [
        phase_features.loc[
            phase_features["phase"] == p,
            "AI0_Vibration_rms"
        ].dropna()
        for p in PHASE_NAMES
    ],
    tick_labels=PHASE_NAMES
)
plt.xlabel("Relative Cycle Phase")
plt.ylabel("AI0 Vibration RMS")
plt.title("AI0 Raw RMS by Relative Cycle Phase")
plt.grid(alpha=0.25)
plt.show()

# ---------------------------
# AI1 Raw Phase RMS
# ---------------------------
plt.figure(figsize=(8, 5))
plt.boxplot(
    [
        phase_features.loc[
            phase_features["phase"] == p,
            "AI1_Vibration_rms"
        ].dropna()
        for p in PHASE_NAMES
    ],
    tick_labels=PHASE_NAMES
)
plt.xlabel("Relative Cycle Phase")
plt.ylabel("AI1 Vibration RMS")
plt.title("AI1 Raw RMS by Relative Cycle Phase")
plt.grid(alpha=0.25)
plt.show()

# ---------------------------
# AI0 Relative Phase RMS
# ---------------------------
plt.figure(figsize=(8, 5))
plt.boxplot(
    [
        phase_features.loc[
            phase_features["phase"] == p,
            "AI0_Vibration_rms_ratio"
        ].dropna()
        for p in PHASE_NAMES
    ],
    tick_labels=PHASE_NAMES
)
plt.axhline(1.0, linewidth=1)
plt.xlabel("Relative Cycle Phase")
plt.ylabel("Phase RMS / Whole-cycle RMS")
plt.title("AI0 Relative RMS by Cycle Phase")
plt.grid(alpha=0.25)
plt.show()

# ---------------------------
# AI1 Relative Phase RMS
# ---------------------------
plt.figure(figsize=(8, 5))
plt.boxplot(
    [
        phase_features.loc[
            phase_features["phase"] == p,
            "AI1_Vibration_rms_ratio"
        ].dropna()
        for p in PHASE_NAMES
    ],
    tick_labels=PHASE_NAMES
)
plt.axhline(1.0, linewidth=1)
plt.xlabel("Relative Cycle Phase")
plt.ylabel("Phase RMS / Whole-cycle RMS")
plt.title("AI1 Relative RMS by Cycle Phase")
plt.grid(alpha=0.25)
plt.show()


# A/B/C Phase의 RMS 차이가 실제로 반복되는 구조인지 정량적으로 검증한다.
# AI0/AI1뿐 아니라 AI2도 같이 비교해 균등 3분할 자체의 의미를 확인한다.
# p-value보다 Kendall's W와 최대 Phase 비율을 중심으로 효과 크기를 판단한다.
# 이 결과로 3분할 Phase feature 유지 여부를 결정한다.

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

PHASES = ["A", "B", "C"]

def evaluate_phase_structure(df, channel):
    value_col = f"{channel}_rms_ratio"

    pivot = (
        df.pivot_table(
            index=["segment_id", "cycle_id"],
            columns="phase",
            values=value_col
        )
        .reindex(columns=PHASES)
        .dropna()
    )

    stat, p = friedmanchisquare(
        pivot["A"],
        pivot["B"],
        pivot["C"]
    )

    n = len(pivot)
    k = len(PHASES)

    # Friedman statistic 기반 Kendall's W
    kendall_w = stat / (n * (k - 1))

    # Cycle마다 가장 RMS가 큰 Phase
    max_phase = pivot.idxmax(axis=1)

    max_share = (
        max_phase.value_counts(normalize=True)
        .reindex(PHASES, fill_value=0)
    )

    # Cycle 내부 Phase 차이
    phase_range = pivot.max(axis=1) - pivot.min(axis=1)

    return {
        "n": n,
        "friedman_stat": stat,
        "p_value": p,
        "kendall_w": kendall_w,
        "median_phase_range": phase_range.median(),
        "q75_phase_range": phase_range.quantile(0.75),
        "max_phase_share": max_share,
        "phase_median": pivot.median()
    }

results_phase = {}

for channel in [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current"
]:
    r = evaluate_phase_structure(
        phase_features,
        channel
    )

    results_phase[channel] = r

    print(f"\n=== {channel} ===")
    print(f"Cycles              : {r['n']}")
    print(f"Friedman statistic  : {r['friedman_stat']:.4f}")
    print(f"p-value             : {r['p_value']:.6g}")
    print(f"Kendall W           : {r['kendall_w']:.4f}")
    print(
        f"Median phase range  : "
        f"{r['median_phase_range']:.4f}"
    )
    print(
        f"75% phase range     : "
        f"{r['q75_phase_range']:.4f}"
    )

    print("\nPhase median ratio")
    print(r["phase_median"].round(4))

    print("\nMax RMS phase share")
    print(
        (r["max_phase_share"] * 100)
        .round(1)
        .astype(str) + "%"
    )


# ## 3페이즈 균등 분리는 폐기
# 정상 AI2 Cycle을 상대위상 기준으로 정렬해 대표 Cycle Template을 만든다.
# Cycle마다 amplitude 차이를 제거한 뒤 반복되는 AI2 shape 자체를 비교한다.
# Template에서 최대/최소 및 가장 큰 상승·하강 변화구간을 Landmark로 추출한다.
# 이후 AI0/AI1 진동이 이 Landmark 주변에서 반응하는지 검증하는 기준으로 사용한다.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_PHASE_POINTS = 17
phase_grid = np.linspace(0, 1, N_PHASE_POINTS)

ai2_cycles_norm = []
ai2_cycles_raw = []
cycle_keys = []

for _, bound in cycle_bounds.iterrows():
    cycle = normal.loc[
        int(bound["start_idx"]):int(bound["end_idx"])
    ].copy()

    # 다음 Cycle 첫 peak와 중복되는 마지막 endpoint 제거
    if len(cycle) < 4:
        continue

    cycle_core = cycle.iloc[:-1].copy()
    x = cycle_core["AI2_Current"].to_numpy(float)

    if len(x) < 3 or np.std(x, ddof=1) == 0:
        continue

    original_phase = np.linspace(0, 1, len(x))

    # 공통 상대위상 grid로만 보간
    x_raw_interp = np.interp(
        phase_grid,
        original_phase,
        x
    )

    # Cycle별 amplitude/offset 제거 후 shape 비교
    x_norm = (x - np.mean(x)) / np.std(x, ddof=1)

    x_norm_interp = np.interp(
        phase_grid,
        original_phase,
        x_norm
    )

    ai2_cycles_raw.append(x_raw_interp)
    ai2_cycles_norm.append(x_norm_interp)

    cycle_keys.append(
        (
            bound["segment_id"],
            bound["cycle_id"]
        )
    )

ai2_cycles_raw = np.asarray(ai2_cycles_raw)
ai2_cycles_norm = np.asarray(ai2_cycles_norm)

# 475개 Normal Cycle의 대표 Template
template_median = np.median(ai2_cycles_norm, axis=0)
template_q25 = np.quantile(ai2_cycles_norm, 0.25, axis=0)
template_q75 = np.quantile(ai2_cycles_norm, 0.75, axis=0)

raw_template_median = np.median(ai2_cycles_raw, axis=0)
raw_template_q25 = np.quantile(ai2_cycles_raw, 0.25, axis=0)
raw_template_q75 = np.quantile(ai2_cycles_raw, 0.75, axis=0)

# Template 주요 Landmark
max_idx = int(np.argmax(template_median))
min_idx = int(np.argmin(template_median))

diff_template = np.diff(template_median)
rise_idx = int(np.argmax(diff_template))
fall_idx = int(np.argmin(diff_template))

# Rise/Fall은 두 점 사이 변화이므로 midpoint 사용
max_phase = phase_grid[max_idx]
min_phase = phase_grid[min_idx]
rise_phase = (phase_grid[rise_idx] + phase_grid[rise_idx + 1]) / 2
fall_phase = (phase_grid[fall_idx] + phase_grid[fall_idx + 1]) / 2

landmarks = pd.DataFrame({
    "landmark": [
        "Maximum",
        "Minimum",
        "Max Rise",
        "Max Fall"
    ],
    "relative_phase": [
        max_phase,
        min_phase,
        rise_phase,
        fall_phase
    ],
    "phase_percent": [
        max_phase * 100,
        min_phase * 100,
        rise_phase * 100,
        fall_phase * 100
    ]
})

print("=== AI2 Normal Cycle Template ===")
print(f"Cycles used      : {len(ai2_cycles_norm)}")
print(f"Phase grid points: {N_PHASE_POINTS}")

print("\n=== AI2 Landmarks ===")
print(landmarks.round(4))

# -------------------------------------------------
# 1. Normalized AI2 Cycle Template
# -------------------------------------------------
plt.figure(figsize=(11, 5))

plt.fill_between(
    phase_grid * 100,
    template_q25,
    template_q75,
    alpha=0.25,
    label="Normal IQR"
)

plt.plot(
    phase_grid * 100,
    template_median,
    marker="o",
    linewidth=2,
    label="Median normalized AI2"
)

plt.axvline(
    max_phase * 100,
    linestyle="--",
    alpha=0.8,
    label=f"Maximum ({max_phase*100:.1f}%)"
)

plt.axvline(
    min_phase * 100,
    linestyle="--",
    alpha=0.8,
    label=f"Minimum ({min_phase*100:.1f}%)"
)

plt.axvline(
    rise_phase * 100,
    linestyle=":",
    alpha=0.8,
    label=f"Max rise ({rise_phase*100:.1f}%)"
)

plt.axvline(
    fall_phase * 100,
    linestyle=":",
    alpha=0.8,
    label=f"Max fall ({fall_phase*100:.1f}%)"
)

plt.xlabel("Relative Cycle Phase (%)")
plt.ylabel("Normalized AI2 Current")
plt.title("Normal AI2 Cycle Template and Signal Landmarks")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# -------------------------------------------------
# 2. Raw AI2 Median Cycle
# -------------------------------------------------
plt.figure(figsize=(11, 5))

plt.fill_between(
    phase_grid * 100,
    raw_template_q25,
    raw_template_q75,
    alpha=0.25,
    label="Normal IQR"
)

plt.plot(
    phase_grid * 100,
    raw_template_median,
    marker="o",
    linewidth=2,
    label="Median raw AI2"
)

plt.xlabel("Relative Cycle Phase (%)")
plt.ylabel("AI2 Current")
plt.title("Normal AI2 Raw Cycle Template")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# -------------------------------------------------
# 3. AI2 Template 변화량
# -------------------------------------------------
diff_phase = (
    (phase_grid[:-1] + phase_grid[1:]) / 2
)

plt.figure(figsize=(11, 4))

plt.plot(
    diff_phase * 100,
    diff_template,
    marker="o"
)

plt.axhline(0, linewidth=1)

plt.axvline(
    rise_phase * 100,
    linestyle="--",
    label="Maximum rise"
)

plt.axvline(
    fall_phase * 100,
    linestyle="--",
    label="Maximum fall"
)

plt.xlabel("Relative Cycle Phase (%)")
plt.ylabel("Δ Normalized AI2")
plt.title("AI2 Cycle-to-Cycle Template Change")
plt.legend()
plt.grid(alpha=0.25)
plt.show()


# AI2 template의 Max Fall, Minimum, Max Rise 주변에서 AI0/AI1 국소 진동을 측정한다.
# 보간된 진동값은 사용하지 않고 각 Cycle의 실제 저장 샘플에서 ±1 sample window를 사용한다.
# Local RMS를 Whole-cycle RMS로 나눠 특정 AI2 transition 주변에 진동이 집중되는지 확인한다.
# Friedman 검정과 최대-response 비율로 landmark별 반복성을 함께 평가한다.
# 10번째 셀임

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

LANDMARKS = {
    "Max Fall": fall_phase,
    "Minimum": min_phase,
    "Max Rise": rise_phase
}

LOCAL_RADIUS = 1
landmark_records = []

for _, bound in cycle_bounds.iterrows():
    cycle = normal.loc[
        int(bound["start_idx"]):int(bound["end_idx"])
    ].copy()

    if len(cycle) < 5:
        continue

    # 다음 Cycle peak와 중복되는 마지막 endpoint 제외
    cycle_core = cycle.iloc[:-1].copy()
    n = len(cycle_core)
    sample_phase = np.linspace(0, 1, n)

    whole_rms = {}
    for channel in ["AI0_Vibration", "AI1_Vibration"]:
        x = cycle_core[channel].to_numpy(float)
        whole_rms[channel] = np.sqrt(np.mean(x**2))

    for landmark_name, target_phase in LANDMARKS.items():
        center = int(np.argmin(np.abs(sample_phase - target_phase)))

        start = max(0, center - LOCAL_RADIUS)
        stop = min(n, center + LOCAL_RADIUS + 1)

        local = cycle_core.iloc[start:stop]

        record = {
            "segment_id": bound["segment_id"],
            "cycle_id": bound["cycle_id"],
            "landmark": landmark_name,
            "target_phase": target_phase,
            "actual_phase": sample_phase[center],
            "center_index": center,
            "n_local": len(local)
        }

        for channel in ["AI0_Vibration", "AI1_Vibration"]:
            x = local[channel].to_numpy(float)

            rms = np.sqrt(np.mean(x**2))

            record[f"{channel}_rms"] = rms
            record[f"{channel}_ptp"] = np.ptp(x)
            record[f"{channel}_rms_ratio"] = (
                rms / whole_rms[channel]
                if whole_rms[channel] > 0 else np.nan
            )

        landmark_records.append(record)

landmark_features = pd.DataFrame(landmark_records)

print("=== Landmark Local Response ===")
print(
    f"Cycles : "
    f"{landmark_features[['segment_id','cycle_id']].drop_duplicates().shape[0]}"
)
print(f"Rows   : {len(landmark_features)}")

# --------------------------------------------------
# Landmark별 요약 + 반복성 검정
# --------------------------------------------------
for channel in ["AI0_Vibration", "AI1_Vibration"]:
    value_col = f"{channel}_rms_ratio"

    summary = (
        landmark_features.groupby("landmark")[value_col]
        .agg(["median", "mean", "std"])
        .reindex(LANDMARKS.keys())
    )

    print(f"\n=== {channel} Landmark RMS Ratio ===")
    print(summary.round(4))

    pivot = (
        landmark_features.pivot_table(
            index=["segment_id", "cycle_id"],
            columns="landmark",
            values=value_col
        )
        .reindex(columns=LANDMARKS.keys())
        .dropna()
    )

    stat, p = friedmanchisquare(
        pivot["Max Fall"],
        pivot["Minimum"],
        pivot["Max Rise"]
    )

    n = len(pivot)
    k = 3
    kendall_w = stat / (n * (k - 1))

    max_share = (
        pivot.idxmax(axis=1)
        .value_counts(normalize=True)
        .reindex(LANDMARKS.keys(), fill_value=0)
    )

    print(f"Friedman p-value : {p:.6g}")
    print(f"Kendall W        : {kendall_w:.4f}")
    print("\nMax local RMS landmark share")
    print((max_share * 100).round(1).astype(str) + "%")

# --------------------------------------------------
# AI0 Local RMS Ratio
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.boxplot(
    [
        landmark_features.loc[
            landmark_features["landmark"] == lm,
            "AI0_Vibration_rms_ratio"
        ].dropna()
        for lm in LANDMARKS
    ],
    tick_labels=list(LANDMARKS.keys())
)
plt.axhline(1.0, linewidth=1)
plt.ylabel("Local RMS / Whole-cycle RMS")
plt.title("AI0 Local Response around AI2 Landmarks")
plt.grid(alpha=0.25)
plt.show()

# --------------------------------------------------
# AI1 Local RMS Ratio
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.boxplot(
    [
        landmark_features.loc[
            landmark_features["landmark"] == lm,
            "AI1_Vibration_rms_ratio"
        ].dropna()
        for lm in LANDMARKS
    ],
    tick_labels=list(LANDMARKS.keys())
)
plt.axhline(1.0, linewidth=1)
plt.ylabel("Local RMS / Whole-cycle RMS")
plt.title("AI1 Local Response around AI2 Landmarks")
plt.grid(alpha=0.25)
plt.show()

# --------------------------------------------------
# Local PTP도 보조 확인
# --------------------------------------------------
for channel in ["AI0_Vibration", "AI1_Vibration"]:
    ptp_summary = (
        landmark_features.groupby("landmark")[f"{channel}_ptp"]
        .median()
        .reindex(LANDMARKS.keys())
    )

    print(f"\n{channel} median Local PTP")
    print(ptp_summary.round(6))


# AI2 Current의 영향을 제거한 뒤 AI0/AI1 진동 사이에 공통 Mechanical Response가 남는지 확인한다.
# AI0/AI1 residual correlation을 계산해 Current와 독립적인 채널 coupling을 분석한다.
# 두 진동 RMS의 log-ratio를 Channel Balance feature로 만들어 시간과 regime 변화를 확인한다.
# Phase 분석 대신 Cycle-level coupling 구조를 Feature Engineering 후보로 평가한다.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

coupling = gf_regime.copy()

# --------------------------------------------------
# 1. AI2 영향을 제거한 AI0 / AI1 residual 관계
# --------------------------------------------------
residual_corr = coupling["AI0_residual"].corr(
    coupling["AI1_residual"]
)

print("=== Current-conditioned Vibration Coupling ===")
print(f"Overall residual correlation : {residual_corr:.4f}")

for regime in ["Other", "Low-RMS"]:
    d = coupling[coupling["regime"] == regime]

    corr = d["AI0_residual"].corr(
        d["AI1_residual"]
    )

    print(
        f"{regime:8s} residual correlation : "
        f"{corr:.4f}"
    )

# --------------------------------------------------
# 2. AI1이 AI0 설명력을 추가하는지 확인
# --------------------------------------------------
y0 = coupling["AI0_Vibration_rms"].to_numpy()
y1 = coupling["AI1_Vibration_rms"].to_numpy()

X_ai2 = coupling[
    ["AI2_Current_rms"]
].to_numpy()

X_ai2_ai1 = coupling[
    ["AI2_Current_rms", "AI1_Vibration_rms"]
].to_numpy()

X_ai2_ai0 = coupling[
    ["AI2_Current_rms", "AI0_Vibration_rms"]
].to_numpy()

m0_base = LinearRegression().fit(X_ai2, y0)
m0_plus = LinearRegression().fit(X_ai2_ai1, y0)

m1_base = LinearRegression().fit(X_ai2, y1)
m1_plus = LinearRegression().fit(X_ai2_ai0, y1)

r2_ai0_base = r2_score(y0, m0_base.predict(X_ai2))
r2_ai0_plus = r2_score(y0, m0_plus.predict(X_ai2_ai1))

r2_ai1_base = r2_score(y1, m1_base.predict(X_ai2))
r2_ai1_plus = r2_score(y1, m1_plus.predict(X_ai2_ai0))

print("\n=== Incremental Channel Information ===")

print(
    f"AI0 ~ AI2          : R²={r2_ai0_base:.4f}"
)
print(
    f"AI0 ~ AI2 + AI1    : R²={r2_ai0_plus:.4f} "
    f"(Δ={r2_ai0_plus-r2_ai0_base:+.4f})"
)

print(
    f"\nAI1 ~ AI2          : R²={r2_ai1_base:.4f}"
)
print(
    f"AI1 ~ AI2 + AI0    : R²={r2_ai1_plus:.4f} "
    f"(Δ={r2_ai1_plus-r2_ai1_base:+.4f})"
)

# --------------------------------------------------
# 3. Channel Balance Feature
# --------------------------------------------------
eps = 1e-12

coupling["vib_log_ratio"] = np.log(
    (coupling["AI0_Vibration_rms"] + eps)
    /
    (coupling["AI1_Vibration_rms"] + eps)
)

balance_summary = (
    coupling.groupby("regime")["vib_log_ratio"]
    .agg(["count", "mean", "std", "median"])
)

print("\n=== AI0 / AI1 Channel Balance ===")
print(balance_summary.round(4))

# --------------------------------------------------
# 4. Residual coupling scatter
# --------------------------------------------------
plt.figure(figsize=(7, 5))

plt.scatter(
    coupling["AI0_residual"],
    coupling["AI1_residual"],
    alpha=0.6
)

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel("AI0 Current-conditioned Residual")
plt.ylabel("AI1 Current-conditioned Residual")
plt.title(
    f"Residual Mechanical Coupling "
    f"(r={residual_corr:.3f})"
)

plt.grid(alpha=0.25)
plt.show()

# --------------------------------------------------
# 5. Channel Balance over Time
# --------------------------------------------------
plt.figure(figsize=(12, 4))

plt.scatter(
    coupling["start_elapsed_sec"],
    coupling["vib_log_ratio"],
    s=18,
    alpha=0.65
)

plt.axvspan(
    3600,
    4300,
    alpha=0.15,
    label="Low-RMS candidate"
)

plt.axhline(
    coupling["vib_log_ratio"].median(),
    linewidth=1,
    linestyle="--",
    label="Normal median"
)

plt.xlabel("Elapsed time (s)")
plt.ylabel("log(AI0 RMS / AI1 RMS)")
plt.title("AI0-AI1 Cycle-level Mechanical Balance over Time")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# --------------------------------------------------
# 6. Regime별 Channel Balance
# --------------------------------------------------
plt.figure(figsize=(7, 5))

plt.boxplot(
    [
        coupling.loc[
            coupling["regime"] == "Other",
            "vib_log_ratio"
        ],
        coupling.loc[
            coupling["regime"] == "Low-RMS",
            "vib_log_ratio"
        ]
    ],
    tick_labels=["Other", "Low-RMS"]
)

plt.ylabel("log(AI0 RMS / AI1 RMS)")
plt.title("Mechanical Channel Balance by Normal Regime")
plt.grid(alpha=0.25)
plt.show()


# Global AI2-Vibration 관계가 시간대별 regime 이동 때문인지 cycle-level coupling인지 분리한다.
# 300초 블록별 상관관계를 계산하고 각 블록 평균을 제거한 centered correlation도 구한다.
# AI2-AI0, AI2-AI1 관계가 Normal 내부에서도 반복적으로 유지되는지 확인한다.
# 이 검증 후 Global vibration EDA를 종료하고 abnormal 비교 단계로 넘어간다.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

TIME_BIN_SEC = 300
MIN_CYCLES_PER_BIN = 15

within = coupling.copy()

within["time_bin"] = (
    np.floor(within["start_elapsed_sec"] / TIME_BIN_SEC)
    * TIME_BIN_SEC
).astype(int)

def bin_stats(g):
    def safe_corr(a, b):
        return a.corr(b) if a.std() > 0 and b.std() > 0 else np.nan

    return pd.Series({
        "n": len(g),
        "AI2_mean": g["AI2_Current_rms"].mean(),
        "AI2_std": g["AI2_Current_rms"].std(),
        "AI0_mean": g["AI0_Vibration_rms"].mean(),
        "AI1_mean": g["AI1_Vibration_rms"].mean(),
        "corr_AI2_AI0": safe_corr(
            g["AI2_Current_rms"],
            g["AI0_Vibration_rms"]
        ),
        "corr_AI2_AI1": safe_corr(
            g["AI2_Current_rms"],
            g["AI1_Vibration_rms"]
        ),
        "corr_AI0_AI1": safe_corr(
            g["AI0_Vibration_rms"],
            g["AI1_Vibration_rms"]
        )
    })

bin_summary = (
    within.groupby("time_bin")
    .apply(bin_stats, include_groups=False)
)

bin_summary = bin_summary[
    bin_summary["n"] >= MIN_CYCLES_PER_BIN
]

print("=== 300 s Within-Regime Correlations ===")
print(bin_summary.round(4))

# 블록 평균 제거
for col in [
    "AI2_Current_rms",
    "AI0_Vibration_rms",
    "AI1_Vibration_rms"
]:
    within[f"{col}_centered"] = (
        within[col]
        - within.groupby("time_bin")[col].transform("mean")
    )

corr_ai2_ai0_centered = within[
    "AI2_Current_rms_centered"
].corr(
    within["AI0_Vibration_rms_centered"]
)

corr_ai2_ai1_centered = within[
    "AI2_Current_rms_centered"
].corr(
    within["AI1_Vibration_rms_centered"]
)

corr_ai0_ai1_centered = within[
    "AI0_Vibration_rms_centered"
].corr(
    within["AI1_Vibration_rms_centered"]
)

print("\n=== Global vs Within-Bin Centered Correlation ===")
print(
    f"AI2 ↔ AI0 | Global={within['AI2_Current_rms'].corr(within['AI0_Vibration_rms']):.4f} "
    f"| Centered={corr_ai2_ai0_centered:.4f}"
)
print(
    f"AI2 ↔ AI1 | Global={within['AI2_Current_rms'].corr(within['AI1_Vibration_rms']):.4f} "
    f"| Centered={corr_ai2_ai1_centered:.4f}"
)
print(
    f"AI0 ↔ AI1 | Global={within['AI0_Vibration_rms'].corr(within['AI1_Vibration_rms']):.4f} "
    f"| Centered={corr_ai0_ai1_centered:.4f}"
)

# AI2 ↔ AI0 centered scatter
plt.figure(figsize=(7, 5))
plt.scatter(
    within["AI2_Current_rms_centered"],
    within["AI0_Vibration_rms_centered"],
    alpha=0.6
)
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)
plt.xlabel("AI2 RMS - 300 s bin mean")
plt.ylabel("AI0 RMS - 300 s bin mean")
plt.title(
    f"Within-Regime AI2 ↔ AI0 "
    f"(r={corr_ai2_ai0_centered:.3f})"
)
plt.grid(alpha=0.25)
plt.show()

# AI2 ↔ AI1 centered scatter
plt.figure(figsize=(7, 5))
plt.scatter(
    within["AI2_Current_rms_centered"],
    within["AI1_Vibration_rms_centered"],
    alpha=0.6
)
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)
plt.xlabel("AI2 RMS - 300 s bin mean")
plt.ylabel("AI1 RMS - 300 s bin mean")
plt.title(
    f"Within-Regime AI2 ↔ AI1 "
    f"(r={corr_ai2_ai1_centered:.3f})"
)
plt.grid(alpha=0.25)
plt.show()

# 시간대별 상관계수
plt.figure(figsize=(11, 5))
plt.plot(
    bin_summary.index,
    bin_summary["corr_AI2_AI0"],
    marker="o",
    label="AI2 ↔ AI0"
)
plt.plot(
    bin_summary.index,
    bin_summary["corr_AI2_AI1"],
    marker="o",
    label="AI2 ↔ AI1"
)
plt.axhline(0, linewidth=1)
plt.ylim(-1.05, 1.05)
plt.xlabel("300 s Time Bin Start")
plt.ylabel("Pearson correlation")
plt.title("Cycle-level Coupling Stability across Normal Time Bins")
plt.legend()
plt.grid(alpha=0.25)
plt.show()


# Abnormal full-cycle을 Normal 기준으로 추출하고 RMS/residual을 계산한다.
# Normal의 AI2→AI0/AI1 관계는 고정해서 사용한다.
# 짧은 segment는 이후 Local Form 분석 대상으로 남긴다.

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

abnormal = data[data["source"] == "abnormal"].copy()
records = []

for seg_id, g in abnormal.groupby("segment_id", sort=False):
    if len(g) < MIN_SEGMENT_ROWS: continue
    y = g["AI2_Current"].to_numpy(float)
    smooth = pd.Series(y).rolling(3, center=True, min_periods=1).median().to_numpy()
    prom = max(np.std(smooth, ddof=1) * 0.5, np.finfo(float).eps)
    peaks, _ = find_peaks(smooth, distance=MIN_PEAK_DISTANCE, prominence=prom)

    for cycle_id, (s, e) in enumerate(zip(peaks[:-1], peaks[1:]), 1):
        if e - s < MIN_PEAK_DISTANCE: continue
        c = g.iloc[s:e+1]
        duration = c["elapsed_sec"].iloc[-1] - c["elapsed_sec"].iloc[0]
        if duration <= 0: continue

        r = {"segment_id": seg_id, "cycle_id": cycle_id,
             "duration_sec": duration, "n_samples": len(c)}

        for ch in ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]:
            x = c[ch].to_numpy(float)
            r[f"{ch}_rms"] = np.sqrt(np.mean(x**2))

        records.append(r)

abnormal_cycles = pd.DataFrame(records)

print("=== Abnormal Full-Cycle Extraction ===")
print(f"Rows: {len(abnormal):,} | Segments: {abnormal['segment_id'].nunique()} | Full cycles: {len(abnormal_cycles)}")

if len(abnormal_cycles):
    Xn = global_features[["AI2_Current_rms"]].to_numpy()
    models = {}

    for ch in ["AI0", "AI1"]:
        y = global_features[f"{ch}_Vibration_rms"].to_numpy()
        model = LinearRegression().fit(Xn, y)
        models[ch] = model
        abnormal_cycles[f"{ch}_expected_rms"] = model.predict(abnormal_cycles[["AI2_Current_rms"]])
        abnormal_cycles[f"{ch}_residual"] = (
            abnormal_cycles[f"{ch}_Vibration_rms"] -
            abnormal_cycles[f"{ch}_expected_rms"]
        )

    abnormal_cycles["vib_log_ratio"] = np.log(
        abnormal_cycles["AI0_Vibration_rms"] /
        abnormal_cycles["AI1_Vibration_rms"]
    )

    cols = ["segment_id", "cycle_id", "duration_sec", "AI2_Current_rms",
            "AI0_Vibration_rms", "AI1_Vibration_rms",
            "AI0_residual", "AI1_residual", "vib_log_ratio"]
    print(abnormal_cycles[cols].round(6))

    x_line = np.linspace(
        global_features["AI2_Current_rms"].min(),
        global_features["AI2_Current_rms"].max(), 200
    ).reshape(-1, 1)

    for ch in ["AI0", "AI1"]:
        plt.figure(figsize=(7, 5))
        plt.scatter(global_features["AI2_Current_rms"],
                    global_features[f"{ch}_Vibration_rms"],
                    alpha=0.3, label="Normal")
        plt.scatter(abnormal_cycles["AI2_Current_rms"],
                    abnormal_cycles[f"{ch}_Vibration_rms"],
                    s=70, label="Abnormal")
        plt.plot(x_line[:, 0], models[ch].predict(x_line),
                 linewidth=2, label="Normal relation")
        plt.xlabel("AI2 Current RMS")
        plt.ylabel(f"{ch} Vibration RMS")
        plt.title(f"Normal vs Abnormal: AI2 → {ch}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.show()
else:
    print("완전한 Abnormal Cycle 없음 → Local Form 분석으로 이동")


# Abnormal full-cycle을 Timing, AI2 range, AI0/AI1 coupling 축으로 정량화한다.
# Normal residual 분포로 z-score를 만들고 regression extrapolation 여부를 구분한다.
# Normal 범위 밖 AI2는 residual보다 operating-intensity anomaly로 우선 해석한다.
# 각 Cycle이 어떤 이상축에서 깨지는지 한 표로 정리한다.

import numpy as np
import pandas as pd

ab = abnormal_cycles.copy()

# Normal reference
dur_q1, dur_q3 = global_features["duration_sec"].quantile([.25, .75])
ai2_min = global_features["AI2_Current_rms"].min()
ai2_max = global_features["AI2_Current_rms"].max()

r0_mu = gf["AI0_residual"].mean()
r0_sd = gf["AI0_residual"].std(ddof=1)
r1_mu = gf["AI1_residual"].mean()
r1_sd = gf["AI1_residual"].std(ddof=1)

ratio_q1, ratio_q3 = coupling["vib_log_ratio"].quantile([.25, .75])
ratio_iqr = ratio_q3 - ratio_q1
ratio_lo = ratio_q1 - 1.5 * ratio_iqr
ratio_hi = ratio_q3 + 1.5 * ratio_iqr

# 이상축
ab["duration_normal"] = ab["duration_sec"].between(dur_q1, dur_q3)
ab["ai2_in_normal_range"] = ab["AI2_Current_rms"].between(ai2_min, ai2_max)

ab["AI0_resid_z"] = (ab["AI0_residual"] - r0_mu) / r0_sd
ab["AI1_resid_z"] = (ab["AI1_residual"] - r1_mu) / r1_sd

ab["AI0_coupling_break"] = ab["AI0_resid_z"].abs() > 3
ab["AI1_coupling_break"] = ab["AI1_resid_z"].abs() > 3
ab["balance_break"] = ~ab["vib_log_ratio"].between(ratio_lo, ratio_hi)

ab["broken_axes"] = (
    (~ab["duration_normal"]).astype(int)
    + (~ab["ai2_in_normal_range"]).astype(int)
    + ab["AI0_coupling_break"].astype(int)
    + ab["AI1_coupling_break"].astype(int)
    + ab["balance_break"].astype(int)
)

cols = [
    "segment_id", "cycle_id", "duration_sec",
    "AI2_Current_rms", "duration_normal", "ai2_in_normal_range",
    "AI0_resid_z", "AI1_resid_z",
    "AI0_coupling_break", "AI1_coupling_break",
    "balance_break", "broken_axes"
]

print("=== Normal Reference ===")
print(f"Duration IQR : {dur_q1:.2f} ~ {dur_q3:.2f} s")
print(f"AI2 RMS range: {ai2_min:.2f} ~ {ai2_max:.2f}")
print(f"AI0 residual SD: {r0_sd:.5f}")
print(f"AI1 residual SD: {r1_sd:.5f}")

print("\n=== Abnormal Full-Cycle Failure Axes ===")
print(ab[cols].round(3))

print("\n=== Failure Count ===")
print(f"Timing break      : {(~ab['duration_normal']).sum()} / {len(ab)}")
print(f"AI2 range break   : {(~ab['ai2_in_normal_range']).sum()} / {len(ab)}")
print(f"AI0 coupling break: {ab['AI0_coupling_break'].sum()} / {len(ab)}")
print(f"AI1 coupling break: {ab['AI1_coupling_break'].sum()} / {len(ab)}")
print(f"Balance break     : {ab['balance_break'].sum()} / {len(ab)}")


# Timing float 오차를 수정하고 Abnormal full-cycle 이상축을 다시 분류한다.
# AI2가 Normal 범위 안일 때만 Current-Vibration coupling break를 판정한다.
# AI2 범위 밖은 coupling anomaly가 아니라 Operating-state OOD로 우선 분류한다.
# 각 Cycle의 Timing / OOD / Coupling / Balance 이상을 최종 정리한다.

ab = abnormal_cycles.copy()
ab["duration_01s"] = ab["duration_sec"].round(1)

ai2_min = global_features["AI2_Current_rms"].min()
ai2_max = global_features["AI2_Current_rms"].max()

r0_mu, r0_sd = gf["AI0_residual"].mean(), gf["AI0_residual"].std(ddof=1)
r1_mu, r1_sd = gf["AI1_residual"].mean(), gf["AI1_residual"].std(ddof=1)

q1, q3 = coupling["vib_log_ratio"].quantile([.25, .75])
iqr = q3 - q1
ratio_lo, ratio_hi = q1 - 1.5*iqr, q3 + 1.5*iqr

ab["timing_break"] = ~ab["duration_01s"].between(1.6, 1.7)
ab["ai2_ood"] = ~ab["AI2_Current_rms"].between(ai2_min, ai2_max)

ab["AI0_resid_z"] = (ab["AI0_residual"] - r0_mu) / r0_sd
ab["AI1_resid_z"] = (ab["AI1_residual"] - r1_mu) / r1_sd

ab["AI0_coupling_break"] = (~ab["ai2_ood"]) & (ab["AI0_resid_z"].abs() > 3)
ab["AI1_coupling_break"] = (~ab["ai2_ood"]) & (ab["AI1_resid_z"].abs() > 3)

ab["balance_break"] = ~ab["vib_log_ratio"].between(ratio_lo, ratio_hi)

ab["broken_axes"] = (
    ab["timing_break"].astype(int)
    + ab["ai2_ood"].astype(int)
    + ab["AI0_coupling_break"].astype(int)
    + ab["AI1_coupling_break"].astype(int)
    + ab["balance_break"].astype(int)
)

cols = [
    "segment_id", "cycle_id", "duration_01s", "AI2_Current_rms",
    "timing_break", "ai2_ood", "AI0_resid_z", "AI1_resid_z",
    "AI0_coupling_break", "AI1_coupling_break",
    "balance_break", "broken_axes"
]

print(ab[cols].round(3))

valid = ab[~ab["ai2_ood"]]

print("=== Corrected Failure Count ===")
print(f"Timing break       : {ab['timing_break'].sum()} / {len(ab)}")
print(f"AI2 OOD            : {ab['ai2_ood'].sum()} / {len(ab)}")
print(f"Coupling evaluable : {len(valid)} / {len(ab)}")
print(f"AI0 coupling break : {valid['AI0_coupling_break'].sum()} / {len(valid)}")
print(f"AI1 coupling break : {valid['AI1_coupling_break'].sum()} / {len(valid)}")
print(f"Balance break      : {ab['balance_break'].sum()} / {len(ab)}")


# Normal AI2 Cycle로 5-sample Local Phase Template 13개를 만든다.
# 각 local window는 whole-cycle z-normalization 상태에서 비교한다.
# Phase별 median template과 Normal 내부 RMSE 분포를 계산한다.
# 이 Normal 기준을 고정한 뒤 다음 단계에서 짧은 Abnormal fragment에 적용한다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOCAL_LEN = 5
n_local = N_PHASE_POINTS - LOCAL_LEN + 1
local_rows, local_templates = [], []

for start in range(n_local):
    windows = ai2_cycles_norm[:, start:start + LOCAL_LEN]
    template = np.median(windows, axis=0)
    local_templates.append(template)

    for i, w in enumerate(windows):
        rmse = np.sqrt(np.mean((w - template) ** 2))
        corr = np.corrcoef(w, template)[0, 1]
        local_rows.append({
            "cycle_idx": i, "local_phase": start,
            "start_pct": phase_grid[start] * 100,
            "end_pct": phase_grid[start + LOCAL_LEN - 1] * 100,
            "rmse": rmse, "corr": corr
        })

local_templates = np.asarray(local_templates)
local_ref = pd.DataFrame(local_rows)

local_summary = local_ref.groupby("local_phase").agg(
    start_pct=("start_pct", "first"),
    end_pct=("end_pct", "first"),
    rmse_med=("rmse", "median"),
    rmse_q95=("rmse", lambda x: x.quantile(.95)),
    corr_med=("corr", "median"),
    corr_q05=("corr", lambda x: x.quantile(.05))
)

print("=== Normal Local Phase Dictionary ===")
print(f"Local length : {LOCAL_LEN} samples")
print(f"Templates    : {n_local}")
print(local_summary.round(4))

plt.figure(figsize=(10, 5))
plt.plot(local_summary.index, local_summary["rmse_med"],
         marker="o", label="Median RMSE")
plt.plot(local_summary.index, local_summary["rmse_q95"],
         marker="o", label="95% RMSE")
plt.xlabel("Local Phase Template")
plt.ylabel("RMSE")
plt.title("Normal AI2 Local-form Stability")
plt.legend()
plt.grid(alpha=.25)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(local_summary.index, local_summary["corr_med"],
         marker="o", label="Median correlation")
plt.plot(local_summary.index, local_summary["corr_q05"],
         marker="o", label="5% correlation")
plt.xlabel("Local Phase Template")
plt.ylabel("Correlation")
plt.title("Normal AI2 Local-form Correlation")
plt.legend()
plt.grid(alpha=.25)
plt.show()


# Normal 5-sample AI2 Local Shape Dictionary를 만들고 Partial Abnormal에 적용한다.
# 각 5-sample window는 local z-normalization 후 Normal 13개 phase template과 비교한다.
# Normal best-match RMSE 95%를 기준으로 Abnormal local-shape deviation을 표시한다.
# Full-cycle이 이미 잡힌 Abnormal segment는 제외하고 partial segment만 분석한다.

import numpy as np, pandas as pd, matplotlib.pyplot as plt

L = 5
z = lambda x: (x - np.mean(x)) / (np.std(x, ddof=1) + 1e-12)

# Normal local shape template
normal_local = []
for p in range(N_PHASE_POINTS - L + 1):
    w = np.array([z(x[p:p+L]) for x in ai2_cycles_raw])
    t = z(np.median(w, axis=0))
    normal_local.append(t)
normal_local = np.array(normal_local)

# Normal best-match 분포로 threshold 설정
normal_scores = []
for x in ai2_cycles_raw:
    for s in range(N_PHASE_POINTS - L + 1):
        w = z(x[s:s+L])
        rmses = np.sqrt(np.mean((normal_local - w) ** 2, axis=1))
        best = np.argmin(rmses)
        normal_scores.append({
            "rmse": rmses[best],
            "corr": np.corrcoef(w, normal_local[best])[0, 1]
        })

normal_scores = pd.DataFrame(normal_scores)
RMSE_THR = normal_scores["rmse"].quantile(.95)
CORR_THR = normal_scores["corr"].quantile(.05)

print("=== Normal Local Matching Reference ===")
print(f"RMSE 95% threshold : {RMSE_THR:.4f}")
print(f"Corr 5% threshold  : {CORR_THR:.4f}")

# Full-cycle 없는 Abnormal segment만 사용
full_segments = set(abnormal_cycles["segment_id"])
partial = abnormal[~abnormal["segment_id"].isin(full_segments)].copy()

rows = []
for seg_id, g in partial.groupby("segment_id", sort=False):
    x = g["AI2_Current"].to_numpy(float)
    if len(x) < L: continue

    for s in range(len(x) - L + 1):
        w = z(x[s:s+L])
        rmses = np.sqrt(np.mean((normal_local - w) ** 2, axis=1))
        best = np.argmin(rmses)
        corr = np.corrcoef(w, normal_local[best])[0, 1]

        rows.append({
            "segment_id": seg_id, "window": s,
            "best_phase": best, "rmse": rmses[best], "corr": corr,
            "rmse_break": rmses[best] > RMSE_THR,
            "corr_break": corr < CORR_THR
        })

local_ab = pd.DataFrame(rows)

summary = local_ab.groupby("segment_id").agg(
    n_windows=("window", "count"),
    best_rmse_med=("rmse", "median"),
    best_rmse_max=("rmse", "max"),
    bad_rmse_frac=("rmse_break", "mean"),
    corr_med=("corr", "median"),
    bad_corr_frac=("corr_break", "mean")
).sort_values("bad_rmse_frac", ascending=False)

print("\n=== Partial Abnormal Local-form Summary ===")
print(summary.round(4))

plt.figure(figsize=(10, 5))
plt.bar(summary.index, summary["bad_rmse_frac"])
plt.axhline(.5, linestyle="--", label="50% windows")
plt.xticks(rotation=45)
plt.ylabel("Fraction of Local Windows > Normal RMSE 95%")
plt.title("Partial Abnormal AI2 Local-form Deviation")
plt.legend()
plt.grid(axis="y", alpha=.25)
plt.show()


# Normal segment를 Train/Holdout으로 분리해 Local Form 기준의 과적합을 검사한다.
# Train Normal만으로 5-sample template을 만들고 Holdout Normal로 95% RMSE를 정한다.
# 그 threshold를 Partial Abnormal에 그대로 적용한다.
# Correlation은 RMSE와 중복되므로 Local Form에서는 제외한다.

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

keys = pd.DataFrame(cycle_keys, columns=["segment_id", "cycle_id"])
segments = keys["segment_id"].unique()
train_seg, val_seg = train_test_split(segments, test_size=.3, random_state=42)

train_idx = keys["segment_id"].isin(train_seg).to_numpy()
val_idx = keys["segment_id"].isin(val_seg).to_numpy()

L = 5
z = lambda x: (x-x.mean())/(x.std(ddof=1)+1e-12)

templates = []
for p in range(N_PHASE_POINTS-L+1):
    w = np.array([z(x[p:p+L]) for x in ai2_cycles_raw[train_idx]])
    templates.append(z(np.median(w, axis=0)))
templates = np.array(templates)

def best_rmse(w):
    w = z(np.asarray(w, float))
    return np.sqrt(np.mean((templates-w)**2, axis=1)).min()

val_scores = []
for x in ai2_cycles_raw[val_idx]:
    for s in range(N_PHASE_POINTS-L+1):
        val_scores.append(best_rmse(x[s:s+L]))

RMSE_THR_HOLDOUT = np.quantile(val_scores, .95)

full_segments = set(abnormal_cycles["segment_id"])
partial = abnormal[~abnormal["segment_id"].isin(full_segments)].copy()
rows = []

for seg_id, g in partial.groupby("segment_id", sort=False):
    x = g["AI2_Current"].to_numpy(float)
    if len(x) < L: continue
    scores = [best_rmse(x[s:s+L]) for s in range(len(x)-L+1)]
    rows.append({
        "segment_id": seg_id, "n_windows": len(scores),
        "rmse_med": np.median(scores), "rmse_max": np.max(scores),
        "bad_frac": np.mean(np.array(scores) > RMSE_THR_HOLDOUT)
    })

holdout_result = pd.DataFrame(rows).set_index("segment_id")

print("=== Holdout Local Form Validation ===")
print(f"Train segments     : {len(train_seg)}")
print(f"Validation segments: {len(val_seg)}")
print(f"Holdout RMSE 95%   : {RMSE_THR_HOLDOUT:.4f}")
print(f"Partial evaluable  : {len(holdout_result)}")
print(holdout_result.sort_values("bad_frac", ascending=False).round(4))

all_partial = set(partial["segment_id"].unique())
evaluated = set(holdout_result.index)
print("Too short (<5 samples):", sorted(all_partial-evaluated))


# Train Normal로 17개 cyclic Local Phase Template을 만든다.
# Holdout Normal에서 정상적인 phase progression 기준을 구한다.
# Partial Abnormal의 best phase가 +1 방향으로 진행하는지 비교한다.
# Local RMSE는 주 지표, phase progression은 보조 지표로 사용한다.

import numpy as np, pandas as pd, matplotlib.pyplot as plt

L, P = 5, N_PHASE_POINTS
z = lambda x: (x-x.mean())/(x.std(ddof=1)+1e-12)

def cycwin(x, p):
    return x[(np.arange(L)+p) % P]

# Train Normal cyclic templates
cyc_templates = []
for p in range(P):
    w = np.array([z(cycwin(x, p)) for x in ai2_cycles_raw[train_idx]])
    cyc_templates.append(z(np.median(w, axis=0)))
cyc_templates = np.array(cyc_templates)

def match_local(w):
    w = z(np.asarray(w, float))
    rmse = np.sqrt(np.mean((cyc_templates-w)**2, axis=1))
    p = int(np.argmin(rmse))
    return p, rmse[p]

def step_error(a, b):
    d = (b-a) % P
    return min((d-1) % P, (1-d) % P)

# Holdout Normal progression 기준
normal_prog, normal_rmse = [], []
for x in ai2_cycles_raw[val_idx]:
    pred, scores = [], []
    for p in range(P):
        bp, r = match_local(cycwin(x, p))
        pred.append(bp); scores.append(r)

    err = [step_error(a, b) for a, b in zip(pred[:-1], pred[1:])]
    normal_prog.append(np.mean(np.array(err) <= 1))
    normal_rmse.extend(scores)

PROG_THR = np.quantile(normal_prog, .05)
CYCLIC_RMSE_THR = np.quantile(normal_rmse, .95)

print("=== Normal Holdout Cyclic Reference ===")
print(f"RMSE 95% threshold    : {CYCLIC_RMSE_THR:.4f}")
print(f"Progression 5% cutoff : {PROG_THR:.4f}")
print(f"Median progression    : {np.median(normal_prog):.4f}")

# Partial Abnormal
rows = []
for seg_id, g in partial.groupby("segment_id", sort=False):
    x = g["AI2_Current"].to_numpy(float)
    if len(x) < L: continue

    phases, scores = [], []
    for s in range(len(x)-L+1):
        p, r = match_local(x[s:s+L])
        phases.append(p); scores.append(r)

    errors = [step_error(a, b) for a, b in zip(phases[:-1], phases[1:])]
    good = np.mean(np.array(errors) <= 1) if errors else np.nan

    rows.append({
        "segment_id": seg_id, "n_windows": len(scores),
        "rmse_med": np.median(scores),
        "bad_shape_frac": np.mean(np.array(scores) > CYCLIC_RMSE_THR),
        "progression_ok": good,
        "progression_break": good < PROG_THR if not np.isnan(good) else True
    })

progression_result = pd.DataFrame(rows).set_index("segment_id")

print("\n=== Partial Abnormal Local Phase Progression ===")
print(
    progression_result
    .sort_values("bad_shape_frac", ascending=False)
    .round(4)
)

plt.figure(figsize=(9, 5))
plt.scatter(
    progression_result["bad_shape_frac"],
    progression_result["progression_ok"],
    s=70
)
plt.axvline(.5, linestyle="--")
plt.axhline(PROG_THR, linestyle="--", label="Normal 5% progression cutoff")
plt.xlabel("Bad Local Shape Fraction")
plt.ylabel("Valid Phase Progression Fraction")
plt.title("Partial Abnormal: Local Shape vs Phase Progression")
plt.legend()
plt.grid(alpha=.25)
plt.show()


# Holdout Normal에서 Cycle별 bad Local Shape 비율의 정상 범위를 계산한다.
# Window RMSE threshold는 기존 CYCLIC_RMSE_THR을 그대로 사용한다.
# Normal Cycle별 bad fraction 95%를 Partial Abnormal 판정기준으로 고정한다.
# 이 기준으로 Local Shape / Progression 이상 여부를 최종 확정한다.

normal_cycle_ref = []

for x in ai2_cycles_raw[val_idx]:
    scores, phases = [], []

    for p in range(P):
        bp, r = match_local(cycwin(x, p))
        phases.append(bp)
        scores.append(r)

    err = [step_error(a, b) for a, b in zip(phases[:-1], phases[1:])]

    normal_cycle_ref.append({
        "bad_frac": np.mean(np.array(scores) > CYCLIC_RMSE_THR),
        "progression_ok": np.mean(np.array(err) <= 1)
    })

normal_cycle_ref = pd.DataFrame(normal_cycle_ref)

BAD_FRAC_THR = normal_cycle_ref["bad_frac"].quantile(.95)
PROG_THR_FINAL = normal_cycle_ref["progression_ok"].quantile(.05)

progression_result["shape_break"] = (
    progression_result["bad_shape_frac"] > BAD_FRAC_THR
)

progression_result["progression_break"] = (
    progression_result["progression_ok"] < PROG_THR_FINAL
)

print("=== Final Local Form Reference ===")
print(f"Normal bad-frac 95% : {BAD_FRAC_THR:.4f}")
print(f"Normal progression 5%: {PROG_THR_FINAL:.4f}")

print("\n=== Partial Abnormal Final Classification ===")
print(
    progression_result[
        ["n_windows",
        "rmse_med",
        "bad_shape_frac",
        "shape_break",
        "progression_ok",
        "progression_break"
        ]
    ].round(4)
)

print("\n=== Detection Count ===")
print(
    f"Local shape break : "
    f"{progression_result['shape_break'].sum()} / "
    f"{len(progression_result)}"
)
print(
    f"Progression break : "
    f"{progression_result['progression_break'].sum()} / "
    f"{len(progression_result)}"
)


# Abnormal 21개 segment를 Full / Partial / Too-short Branch로 통합한다.
# Break / 정상 / N-A를 서로 다른 상태로 표시해 오해를 막는다.
# Global Cycle, Local Form, Coverage 축을 시각적으로 구분한다.
# 보고서용 최종 Failure Map으로 사용한다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

segments = sorted(abnormal["segment_id"].unique())
fm = pd.DataFrame(index=segments)
fm.index.name = "segment_id"
fm["branch"] = "Too short"

cols = [
    "Timing Break", "AI2 OOD",
    "AI0 Coupling", "AI1 Coupling",
    "Local Shape", "Progression",
    "Not Evaluable"
]

for c in cols:
    fm[c] = False

# Full-cycle
for seg, g in ab.groupby("segment_id"):
    fm.loc[seg, "branch"] = "Full cycle"
    fm.loc[seg, "Timing Break"] = g["timing_break"].any()
    fm.loc[seg, "AI2 OOD"] = g["ai2_ood"].any()
    fm.loc[seg, "AI0 Coupling"] = g["AI0_coupling_break"].any()
    fm.loc[seg, "AI1 Coupling"] = g["AI1_coupling_break"].any()

# Partial
for seg, r in progression_result.iterrows():
    fm.loc[seg, "branch"] = "Partial"
    fm.loc[seg, "Local Shape"] = r["shape_break"]
    fm.loc[seg, "Progression"] = r["progression_break"]

# Too-short
fm.loc[fm["branch"] == "Too short", "Not Evaluable"] = True

order = {"Full cycle": 0, "Partial": 1, "Too short": 2}
fm_plot = (
    fm.reset_index()
    .assign(branch_order=lambda x: x["branch"].map(order))
    .sort_values(["branch_order", "segment_id"])
    .set_index("segment_id")
)

# 0 = evaluated/no break, 1 = break, 2 = N/A
state = fm_plot[cols].astype(int).to_numpy()

for i, branch in enumerate(fm_plot["branch"]):
    if branch == "Full cycle":
        state[i, 4:6] = 2
    elif branch == "Partial":
        state[i, 0:4] = 2
    else:
        state[i, 0:6] = 2

cmap = ListedColormap(["#F7F7F7", "#FBC02D", "#BDBDBD"])

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(state, aspect="auto", cmap=cmap, vmin=0, vmax=2)

ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=30, ha="right")
ax.set_yticks(range(len(fm_plot)))
ax.set_yticklabels(fm_plot.index)

# Break만 X 표시
for i in range(state.shape[0]):
    for j in range(state.shape[1]):
        if state[i, j] == 1:
            ax.text(j, i, "X", ha="center", va="center", fontweight="bold")

# Branch 경계
branches = fm_plot["branch"].to_numpy()
for i in range(1, len(branches)):
    if branches[i] != branches[i-1]:
        ax.axhline(i - .5, linewidth=2)

# 분석축 구분: Global | Local | Coverage
ax.axvline(3.5, linewidth=1.5, linestyle="--")
ax.axvline(5.5, linewidth=1.5, linestyle="--")

# Branch 이름 + 개수
for branch in ["Full cycle", "Partial", "Too short"]:
    idx = np.where(branches == branch)[0]
    if len(idx):
        mid = (idx[0] + idx[-1]) / 2
        ax.text(
            -1.15, mid,
            f"{branch}\n(n={len(idx)})",
            ha="right", va="center",
            fontweight="bold"
        )

legend = [
    Patch(facecolor="#F7F7F7", edgecolor="gray", label="Evaluated / No break"),
    Patch(facecolor="#FBC02D", edgecolor="gray", label="Break detected"),
    Patch(facecolor="#BDBDBD", edgecolor="gray", label="N/A")
]

ax.legend(
    handles=legend,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=3,
    frameon=False
)

ax.set_title("Abnormal Segment Failure Map", fontweight="bold")
ax.set_xlabel("Failure Axis")
ax.set_ylabel("Abnormal Segment")

plt.subplots_adjust(left=0.20, bottom=0.24)
plt.show()

print("=== Coverage ===")
print(f"Full cycle : {(fm['branch']=='Full cycle').sum()}")
print(f"Partial    : {(fm['branch']=='Partial').sum()}")
print(f"Too short  : {(fm['branch']=='Too short').sum()}")
print(f"Evaluable  : {(fm['branch']!='Too short').sum()} / {len(fm)}")


