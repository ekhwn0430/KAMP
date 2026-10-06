from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from scipy.stats import kurtosis

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ============================================================
# 0. Config
# ============================================================

SIGNALS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]

MIN_SEG_ROWS = 30
MIN_PEAK_DISTANCE = 12

PHASE_POINTS = 32
PHASE_NAMES = ["A", "B", "C"]

TEST_SIZE = 0.30
RANDOM_STATE = 42
EPS = 1e-12


# ============================================================
# 1. Path / Load
# ============================================================

def find_root():
    starts = [Path.cwd()]
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass

    for start in starts:
        for root in [start, *start.parents]:
            path = root / "data" / "processed" / "preprocessed_data.csv"
            if path.exists():
                return root

    raise FileNotFoundError("data/processed/preprocessed_data.csv를 찾지 못했습니다.")


ROOT = find_root()
DATA_PATH = ROOT / "data" / "processed" / "preprocessed_data.csv"
OUT_DIR = ROOT / "07_Feature_Engineering" / "Data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATA_PATH)

if "TimeStamp" in data.columns:
    data["TimeStamp"] = pd.to_datetime(data["TimeStamp"])

required = ["segment_id", "source", "elapsed_sec", "pos_in_seg", *SIGNALS]
missing = [c for c in required if c not in data.columns]

if missing:
    raise ValueError(f"필수 컬럼 누락: {missing}")

print("=" * 70)
print("07 Feature Engineering")
print("=" * 70)
print("DATA:", DATA_PATH)
print("Rows:", data["source"].value_counts().to_dict())


# ============================================================
# 2. Basic Feature Functions
# ============================================================

def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2)))


def basic_features(x):
    x = np.asarray(x, float)
    x_rms = rms(x)

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "rms": x_rms,
        "ptp": float(np.ptp(x)),
        "peak": float(np.max(np.abs(x))),
        "mean_abs": float(np.mean(np.abs(x))),
        "energy": float(np.sum(x ** 2)),
        "crest_factor": float(np.max(np.abs(x)) / x_rms) if x_rms > 0 else np.nan,
        "kurtosis": float(kurtosis(x, fisher=False, bias=False)) if len(x) >= 4 else np.nan,
    }


# ============================================================
# 3. Shape / Harmonic / Phase
# ============================================================

def normalize_cycle_shape(x, n_points=PHASE_POINTS):
    x = np.asarray(x, float)

    if len(x) < 2:
        return None

    y = np.interp(
        np.linspace(0, 1, n_points),
        np.linspace(0, 1, len(x)),
        x
    )

    sd = np.std(y, ddof=1)

    if sd <= 0:
        return None

    return (y - np.mean(y)) / sd


def harmonic_share(shape):
    if shape is None:
        return {"H1_share": np.nan, "H2_share": np.nan, "H3_share": np.nan}

    power = np.abs(np.fft.rfft(shape)) ** 2
    power = power[1:]
    share = power / (power.sum() + EPS)

    return {
        "H1_share": float(share[0]) if len(share) > 0 else np.nan,
        "H2_share": float(share[1]) if len(share) > 1 else np.nan,
        "H3_share": float(share[2]) if len(share) > 2 else np.nan,
    }


def phase_features(cycle):
    result = {}
    core = cycle.iloc[:-1].copy() if len(cycle) > 3 else cycle.copy()
    groups = np.array_split(np.arange(len(core)), 3)

    cycle_rms = {
        ch: rms(core[ch].to_numpy(float))
        for ch in SIGNALS
    }

    for phase_name, idx in zip(PHASE_NAMES, groups):
        if len(idx) == 0:
            continue

        phase = core.iloc[idx]

        for ch in SIGNALS:
            x = phase[ch].to_numpy(float)
            prefix = f"{ch}_phase_{phase_name}"
            p_rms = rms(x)

            result[f"{prefix}_rms"] = p_rms
            result[f"{prefix}_std"] = np.std(x, ddof=1) if len(x) > 1 else np.nan
            result[f"{prefix}_ptp"] = np.ptp(x)
            result[f"{prefix}_mean_abs"] = np.mean(np.abs(x))
            result[f"{prefix}_rms_ratio"] = p_rms / (cycle_rms[ch] + EPS)

    return result


# ============================================================
# 4. Cycle Extraction
# ============================================================

def extract_cycles(df, source):
    rows, shapes = [], []

    for segment_id, group in df[df["source"] == source].groupby("segment_id", sort=False):
        group = group.sort_values("pos_in_seg").reset_index(drop=True)

        if len(group) < MIN_SEG_ROWS:
            continue

        current = group["AI2_Current"].to_numpy(float)
        smooth = pd.Series(current).rolling(3, center=True, min_periods=1).median().to_numpy()

        prominence = max(np.std(smooth, ddof=1) * 0.5, np.finfo(float).eps)

        peaks, _ = find_peaks(
            smooth,
            distance=MIN_PEAK_DISTANCE,
            prominence=prominence
        )

        for cycle_id, (start, stop) in enumerate(zip(peaks[:-1], peaks[1:]), 1):
            if stop - start < MIN_PEAK_DISTANCE:
                continue

            cycle = group.iloc[start:stop + 1].copy()

            duration = float(
                cycle["elapsed_sec"].iloc[-1]
                - cycle["elapsed_sec"].iloc[0]
            )

            if duration <= 0:
                continue

            row = {
                "source": source,
                "label": int(source == "abnormal"),
                "segment_id": segment_id,
                "cycle_id": cycle_id,
                "cycle_start_sec": float(cycle["elapsed_sec"].iloc[0]),
                "cycle_end_sec": float(cycle["elapsed_sec"].iloc[-1]),
                "cycle_duration": duration,
                "n_samples_cycle": len(cycle),
            }

            for ch in SIGNALS:
                for name, value in basic_features(cycle[ch]).items():
                    row[f"{ch}_{name}"] = value

            ai0_rms = row["AI0_Vibration_rms"]
            ai1_rms = row["AI1_Vibration_rms"]

            row["vibration_normalized_difference"] = (
                (ai0_rms - ai1_rms)
                / (ai0_rms + ai1_rms + EPS)
            )

            row["vibration_log_ratio"] = np.log(
                (ai0_rms + EPS)
                / (ai1_rms + EPS)
            )

            row.update(phase_features(cycle))

            shape = normalize_cycle_shape(cycle["AI2_Current"].to_numpy(float))

            if shape is None:
                continue

            row.update(harmonic_share(shape))

            rows.append(row)
            shapes.append(shape)

    if not rows:
        return pd.DataFrame(), np.empty((0, PHASE_POINTS))

    return pd.DataFrame(rows), np.vstack(shapes)


# ============================================================
# 5. Build Normal / Abnormal Cycle
# ============================================================

normal_features, normal_shapes = extract_cycles(data, "normal")
abnormal_features, abnormal_shapes = extract_cycles(data, "abnormal")

print("\n=== Cycle Extraction ===")
print("Normal cycles   :", len(normal_features))
print("Abnormal cycles :", len(abnormal_features))

if normal_features.empty:
    raise ValueError("Normal cycle을 생성하지 못했습니다.")


# ============================================================
# 6. Train / Holdout Split
# ============================================================

segments = normal_features["segment_id"].unique()

train_seg, holdout_seg = train_test_split(
    segments,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_seg = set(train_seg)
holdout_seg = set(holdout_seg)

train_mask = normal_features["segment_id"].isin(train_seg).to_numpy()
holdout_mask = normal_features["segment_id"].isin(holdout_seg).to_numpy()

normal_features["split"] = np.where(train_mask, "train", "holdout")

normal_train = normal_features[train_mask].copy()
normal_holdout = normal_features[holdout_mask].copy()

train_shapes = normal_shapes[train_mask]
holdout_shapes = normal_shapes[holdout_mask]

if not abnormal_features.empty:
    abnormal_features["split"] = "test"

print("\nNormal Train   :", len(normal_train))
print("Normal Holdout :", len(normal_holdout))


# ============================================================
# 7. Timing Feature
# ============================================================

T_REF = normal_train["cycle_duration"].median()
T_MEAN = normal_train["cycle_duration"].mean()
T_STD = normal_train["cycle_duration"].std()


def add_timing_features(df):
    out = df.copy()

    out["timing_deviation"] = (
        (out["cycle_duration"] - T_REF).abs()
        / (T_REF + EPS)
    )

    out["duration_z"] = (
        (out["cycle_duration"] - T_MEAN)
        / (T_STD + EPS)
    )

    return out


normal_train = add_timing_features(normal_train)
normal_holdout = add_timing_features(normal_holdout)

if not abnormal_features.empty:
    abnormal_features = add_timing_features(abnormal_features)


# ============================================================
# 8. Current -> Vibration Response
# ============================================================

response_models = {}
response_refs = {}

X_train = normal_train[["AI2_Current_rms"]].to_numpy()

for ch in ["AI0", "AI1"]:
    y_col = f"{ch}_Vibration_rms"
    y_train = normal_train[y_col].to_numpy()

    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_train)
    residual = y_train - pred

    response_models[ch] = model

    response_refs[ch] = {
        "mean": residual.mean(),
        "std": residual.std(ddof=1),
        "slope": model.coef_[0],
        "intercept": model.intercept_,
        "r2": model.score(X_train, y_train),
    }

print("\n=== Normal Current -> Vibration Reference ===")

for ch, ref in response_refs.items():
    print(f"{ch} | slope={ref['slope']:.6f} | R2={ref['r2']:.4f}")


def add_response_features(df):
    out = df.copy()
    X = out[["AI2_Current_rms"]].to_numpy()

    for ch in ["AI0", "AI1"]:
        model = response_models[ch]
        ref = response_refs[ch]

        pred = model.predict(X)
        residual = out[f"{ch}_Vibration_rms"].to_numpy() - pred

        out[f"{ch}_expected_rms"] = pred
        out[f"{ch}_current_residual"] = residual
        out[f"{ch}_abs_current_residual"] = np.abs(residual)

        out[f"{ch}_response_z"] = (
            residual - ref["mean"]
        ) / (
            ref["std"] + EPS
        )

    out["PORD"] = out[
        ["AI0_response_z", "AI1_response_z"]
    ].abs().max(axis=1)

    return out


normal_train = add_response_features(normal_train)
normal_holdout = add_response_features(normal_holdout)

if not abnormal_features.empty:
    abnormal_features = add_response_features(abnormal_features)


# ============================================================
# 9. CNVG
# ============================================================

I_REF = normal_train["AI2_Current_rms"].median()

V_REF = {
    ch: normal_train[f"{ch}_Vibration_rms"].median()
    for ch in ["AI0", "AI1"]
}


def raw_gain(df, ch):
    return (
        df[f"{ch}_Vibration_rms"] / (V_REF[ch] + EPS)
    ) / (
        df["AI2_Current_rms"] / (I_REF + EPS)
    )


GAIN_REF = {
    ch: raw_gain(normal_train, ch).median()
    for ch in ["AI0", "AI1"]
}


def add_cnvg(df):
    out = df.copy()

    for ch in ["AI0", "AI1"]:
        gain = raw_gain(out, ch)

        out[f"{ch}_Gain"] = gain
        out[f"{ch}_GainDev"] = np.abs(
            np.log(
                (gain + EPS)
                / (GAIN_REF[ch] + EPS)
            )
        )

    out["CNVG"] = out[
        ["AI0_GainDev", "AI1_GainDev"]
    ].max(axis=1)

    return out


normal_train = add_cnvg(normal_train)
normal_holdout = add_cnvg(normal_holdout)

if not abnormal_features.empty:
    abnormal_features = add_cnvg(abnormal_features)


# ============================================================
# 10. Shape Feature
# ============================================================

shape_template = np.mean(train_shapes, axis=0)

shape_template = (
    shape_template - shape_template.mean()
) / (
    shape_template.std(ddof=1) + EPS
)


def add_shape_features(df, shapes):
    out = df.copy()

    if len(out) == 0:
        out["Shape_RMSE"] = []
        out["Template_Correlation"] = []
        return out

    out["Shape_RMSE"] = [
        np.sqrt(np.mean((shape - shape_template) ** 2))
        for shape in shapes
    ]

    out["Template_Correlation"] = [
        np.corrcoef(shape, shape_template)[0, 1]
        for shape in shapes
    ]

    return out


normal_train = add_shape_features(normal_train, train_shapes)
normal_holdout = add_shape_features(normal_holdout, holdout_shapes)

if not abnormal_features.empty:
    abnormal_features = add_shape_features(abnormal_features, abnormal_shapes)


# ============================================================
# 11. Final Feature Table
# ============================================================

feature_table = pd.concat(
    [normal_train, normal_holdout, abnormal_features],
    ignore_index=True
)

TIER1_FEATURES = [
    "cycle_duration",
    "timing_deviation",
    "duration_z",

    "AI0_Vibration_rms",
    "AI1_Vibration_rms",
    "AI2_Current_rms",

    "AI0_Vibration_std",
    "AI1_Vibration_std",
    "AI2_Current_std",

    "AI0_Vibration_ptp",
    "AI1_Vibration_ptp",
    "AI2_Current_ptp",

    "AI0_current_residual",
    "AI1_current_residual",

    "AI0_response_z",
    "AI1_response_z",

    "PORD",
    "vibration_normalized_difference",

    "Shape_RMSE",
    "Template_Correlation",
]

TIER2_FEATURES = [
    "CNVG",
    "vibration_log_ratio",

    "AI0_Vibration_peak",
    "AI1_Vibration_peak",

    "AI0_Vibration_crest_factor",
    "AI1_Vibration_crest_factor",

    "AI0_Vibration_kurtosis",
    "AI1_Vibration_kurtosis",

    "AI2_Current_energy",
    "AI2_Current_crest_factor",

    "H1_share",
    "H2_share",
    "H3_share",
]

PHASE_FEATURES = [
    c
    for c in feature_table.columns
    if "_phase_" in c
]


# ============================================================
# 12. Save
# ============================================================

SAVE_ALL = OUT_DIR / "cycle_feature_table_all.csv"
SAVE_TIER1 = OUT_DIR / "cycle_feature_table_tier1.csv"

feature_table.to_csv(SAVE_ALL, index=False)

tier1_cols = [
    "source",
    "label",
    "split",
    "segment_id",
    "cycle_id",
    *TIER1_FEATURES,
]

tier1_cols = [
    c
    for c in tier1_cols
    if c in feature_table.columns
]

feature_table[tier1_cols].to_csv(
    SAVE_TIER1,
    index=False
)

np.savetxt(
    OUT_DIR / "normal_ai2_cycle_template.csv",
    shape_template,
    delimiter=","
)

reference_table = pd.DataFrame({
    "metric": [
        "cycle_duration_median",
        "cycle_duration_mean",
        "cycle_duration_std",
        "AI2_Current_rms_median",
        "AI0_Vibration_rms_median",
        "AI1_Vibration_rms_median",
    ],
    "value": [
        T_REF,
        T_MEAN,
        T_STD,
        I_REF,
        V_REF["AI0"],
        V_REF["AI1"],
    ]
})

reference_table.to_csv(
    OUT_DIR / "feature_reference.csv",
    index=False
)


# ============================================================
# 13. Summary
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print("\nCycle count:")
print(
    feature_table
    .groupby(["source", "split"])
    .size()
)

print("\nTotal cycles :", len(feature_table))
print("Columns      :", len(feature_table.columns))
print("Tier 1       :", len(TIER1_FEATURES))
print("Tier 2       :", len(TIER2_FEATURES))
print("Phase        :", len(PHASE_FEATURES))

print("\nSaved:")
print(SAVE_ALL)
print(SAVE_TIER1)
print(OUT_DIR / "normal_ai2_cycle_template.csv")
print(OUT_DIR / "feature_reference.csv")
