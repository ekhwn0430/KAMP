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

# Cycle 검출 기준값 (01_Cycle.py / 05.EDA_0_Cycle.py와 동일)
CHANNELS=["AI0_Vibration","AI1_Vibration","AI2_Current"]
SAMPLE_RATE_HZ=10.0
MIN_SEGMENT_ROWS=30
CYCLE_ANCHOR="AI2_Current"
CYCLE_MIN_PEAK_DISTANCE_SAMPLES=12


required_objects = [
    "normal",
    "outlier",
    "MIN_SEGMENT_ROWS",
    "CYCLE_ANCHOR",
    "CYCLE_MIN_PEAK_DISTANCE_SAMPLES",
    "SAMPLE_RATE_HZ",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise NameError(
        f"5.EDA_0_Cycle.ipynb에서 필요한 객체를 찾을 수 없음: {missing_objects}"
    )

print("Reference notebook loaded successfully.")
print()
print(f"Cycle anchor                : {CYCLE_ANCHOR}")
print(f"Minimum segment rows        : {MIN_SEGMENT_ROWS}")
print(f"Minimum peak distance       : {CYCLE_MIN_PEAK_DISTANCE_SAMPLES} samples")
print(f"Sampling rate               : {SAMPLE_RATE_HZ} Hz")
print()
print(f"Normal segments             : {normal['segment_id'].nunique()}")
print(f"Outlier segments            : {outlier['segment_id'].nunique()}")


from scipy.signal import find_peaks
import numpy as np
import pandas as pd


def detector_diagnostics(df, dataset_name):
    rows = []

    for segment_id, group in df.groupby("segment_id", sort=False):

        n_samples = len(group)

        # 기존 detector의 최소 segment 길이 조건
        eligible = n_samples >= MIN_SEGMENT_ROWS

        values = group[CYCLE_ANCHOR].to_numpy(dtype=float)

        smooth = (
            pd.Series(values)
            .rolling(3, center=True, min_periods=1)
            .median()
            .to_numpy()
        )

        # 기존 detector와 동일한 adaptive prominence
        if len(smooth) > 1:
            prominence = max(
                float(pd.Series(smooth).std()) * 0.5,
                np.finfo(float).eps
            )
        else:
            prominence = np.nan

        # 길이가 짧으면 기존 detector에서는 아예 검사하지 않음
        if eligible and np.isfinite(values).all():
            peaks, _ = find_peaks(
                smooth,
                distance=CYCLE_MIN_PEAK_DISTANCE_SAMPLES,
                prominence=prominence,
            )
        else:
            peaks = np.array([], dtype=int)

        peak_count = len(peaks)
        cycle_count = max(peak_count - 1, 0)

        if not eligible:
            status = "short segment"
        elif peak_count == 0:
            status = "peak 0"
        elif peak_count == 1:
            status = "peak 1"
        else:
            status = "peak 2+"

        rows.append({
            "dataset": dataset_name,
            "segment_id": segment_id,
            "n_samples": n_samples,
            "eligible": eligible,
            "prominence": prominence,
            "peak_count": peak_count,
            "candidate_cycle_count": cycle_count,
            "status": status,
        })

    return pd.DataFrame(rows)


normal_diag = detector_diagnostics(normal, "Normal")
outlier_diag = detector_diagnostics(outlier, "Outlier")


def eligibility_summary(diag):
    total = len(diag)
    eligible = int(diag["eligible"].sum())
    short = int((~diag["eligible"]).sum())

    return pd.Series({
        "total_segments": total,
        "eligible_segments": eligible,
        "short_segments": short,
        "eligible_fraction": eligible / total if total > 0 else np.nan,
    })


q1_summary = pd.DataFrame([
    eligibility_summary(normal_diag).rename("Normal"),
    eligibility_summary(outlier_diag).rename("Outlier"),
])

print(q1_summary)

print(
    f"Outlier 전체 segment : {len(outlier_diag)}"
)

print(
    f"30 samples 이상      : {outlier_diag['eligible'].sum()}"
)

print(
    f"30 samples 미만      : {(~outlier_diag['eligible']).sum()}"
)


outlier_eligible = outlier_diag.loc[
    outlier_diag["eligible"]
].copy()

q2_summary = (
    outlier_eligible["status"]
    .value_counts()
    .reindex(
        ["peak 0", "peak 1", "peak 2+"],
        fill_value=0
    )
    .rename("segment_count")
    .to_frame()
)

print(q2_summary)

print(
    outlier_eligible[
        [
            "segment_id",
            "n_samples",
            "prominence",
            "peak_count",
            "candidate_cycle_count",
            "status",
        ]
    ]
    .sort_values("segment_id")
)

print(
    "Total Outlier candidate cycles:",
    int(outlier_eligible["candidate_cycle_count"].sum())
)


import matplotlib.pyplot as plt
import math

eligible_ids = (
    outlier_eligible["segment_id"]
    .tolist()
)

n = len(eligible_ids)
ncols = 2
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(14, 3.2 * nrows),
    constrained_layout=True
)

axes = np.atleast_1d(axes).ravel()

for ax, segment_id in zip(axes, eligible_ids):

    group = outlier.loc[
        outlier["segment_id"] == segment_id
    ]

    raw = group[CYCLE_ANCHOR].to_numpy(dtype=float)

    smooth = (
        pd.Series(raw)
        .rolling(3, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    row = outlier_eligible.loc[
        outlier_eligible["segment_id"] == segment_id
    ].iloc[0]

    prominence = float(row["prominence"])

    peaks, _ = find_peaks(
        smooth,
        distance=CYCLE_MIN_PEAK_DISTANCE_SAMPLES,
        prominence=prominence,
    )

    time = np.arange(len(group)) / SAMPLE_RATE_HZ

    ax.plot(time, raw, label="AI2 raw")
    ax.plot(time, smooth, label="3-sample median")

    if len(peaks):
        ax.scatter(
            time[peaks],
            smooth[peaks],
            marker="x",
            s=60,
            label="Detected peak",
        )

    ax.set_title(
        f"segment {segment_id} | "
        f"rows={len(group)} | "
        f"peaks={len(peaks)} | "
        f"cycles={max(len(peaks)-1, 0)}"
    )

    ax.set_xlabel("Time within segment (s)")
    ax.set_ylabel(CYCLE_ANCHOR)
    ax.legend()

for ax in axes[n:]:
    ax.set_visible(False)

plt.show()


normal_eligible = normal_diag.loc[
    normal_diag["eligible"]
].copy()

FIXED_NORMAL_PROMINENCE = float(
    normal_eligible["prominence"].median()
)

print(
    f"Normal-derived fixed prominence = "
    f"{FIXED_NORMAL_PROMINENCE:.6g}"
)


def detect_with_fixed_prominence(
    df,
    dataset_name,
    fixed_prominence,
):
    rows = []

    for segment_id, group in df.groupby(
        "segment_id",
        sort=False
    ):

        if len(group) < MIN_SEGMENT_ROWS:
            continue

        values = group[CYCLE_ANCHOR].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            continue

        smooth = (
            pd.Series(values)
            .rolling(3, center=True, min_periods=1)
            .median()
            .to_numpy()
        )

        peaks, _ = find_peaks(
            smooth,
            distance=CYCLE_MIN_PEAK_DISTANCE_SAMPLES,
            prominence=fixed_prominence,
        )

        rows.append({
            "dataset": dataset_name,
            "segment_id": segment_id,
            "peak_count_fixed": len(peaks),
            "candidate_cycle_count_fixed":
                max(len(peaks) - 1, 0),
        })

    return pd.DataFrame(rows)


normal_fixed = detect_with_fixed_prominence(
    normal,
    "Normal",
    FIXED_NORMAL_PROMINENCE,
)

outlier_fixed = detect_with_fixed_prominence(
    outlier,
    "Outlier",
    FIXED_NORMAL_PROMINENCE,
)


adaptive_vs_fixed = (
    outlier_eligible[
        [
            "segment_id",
            "n_samples",
            "peak_count",
            "candidate_cycle_count",
        ]
    ]
    .merge(
        outlier_fixed,
        on="segment_id",
        how="left",
    )
    .rename(columns={
        "peak_count": "peak_count_adaptive",
        "candidate_cycle_count":
            "candidate_cycle_count_adaptive",
    })
)

adaptive_vs_fixed["peak_difference"] = (
    adaptive_vs_fixed["peak_count_fixed"]
    - adaptive_vs_fixed["peak_count_adaptive"]
)

print(adaptive_vs_fixed)

print(
    "Adaptive total peaks:",
    adaptive_vs_fixed[
        "peak_count_adaptive"
    ].sum()
)

print(
    "Fixed total peaks:",
    adaptive_vs_fixed[
        "peak_count_fixed"
    ].sum()
)

print(
    "Adaptive candidate cycles:",
    adaptive_vs_fixed[
        "candidate_cycle_count_adaptive"
    ].sum()
)

print(
    "Fixed candidate cycles:",
    adaptive_vs_fixed[
        "candidate_cycle_count_fixed"
    ].sum()
)


def detector_sensitivity(
    df,
    dataset_name,
    base_prominence,
):

    rows = []

    prominence_factors = [
        0.8,
        1.0,
        1.2,
    ]

    peak_distances = [
        10,
        12,
        14,
    ]

    for prominence_factor in prominence_factors:
        for distance in peak_distances:

            total_peaks = 0
            total_cycles = 0
            eligible_segments = 0
            segments_with_cycle = 0

            threshold = (
                base_prominence
                * prominence_factor
            )

            for segment_id, group in df.groupby(
                "segment_id",
                sort=False
            ):

                if len(group) < MIN_SEGMENT_ROWS:
                    continue

                values = group[
                    CYCLE_ANCHOR
                ].to_numpy(dtype=float)

                if not np.isfinite(values).all():
                    continue

                eligible_segments += 1

                smooth = (
                    pd.Series(values)
                    .rolling(
                        3,
                        center=True,
                        min_periods=1
                    )
                    .median()
                    .to_numpy()
                )

                peaks, _ = find_peaks(
                    smooth,
                    distance=distance,
                    prominence=threshold,
                )

                cycles = max(
                    len(peaks) - 1,
                    0
                )

                total_peaks += len(peaks)
                total_cycles += cycles

                if cycles > 0:
                    segments_with_cycle += 1

            rows.append({
                "dataset": dataset_name,
                "prominence_factor":
                    prominence_factor,
                "peak_distance":
                    distance,
                "eligible_segments":
                    eligible_segments,
                "segments_with_cycle":
                    segments_with_cycle,
                "total_peaks":
                    total_peaks,
                "total_candidate_cycles":
                    total_cycles,
            })

    return pd.DataFrame(rows)


normal_sensitivity = detector_sensitivity(
    normal,
    "Normal",
    FIXED_NORMAL_PROMINENCE,
)

outlier_sensitivity = detector_sensitivity(
    outlier,
    "Outlier",
    FIXED_NORMAL_PROMINENCE,
)

sensitivity = pd.concat(
    [
        normal_sensitivity,
        outlier_sensitivity,
    ],
    ignore_index=True,
)

print(sensitivity)
