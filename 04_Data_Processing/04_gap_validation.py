import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

CHANNELS=["AI0_Vibration","AI1_Vibration","AI2_Current"]


GAP_THRESHOLD_SEC = 0.15
GAP_DATASETS = {"Normal": normal, "Outlier": outlier}

def interval_summary(values):
    values = values.dropna()
    return pd.Series({
        "count": values.count(),
        "min": values.min(),
        "Q1": values.quantile(0.25),
        "median": values.median(),
        "mean": values.mean(),
        "Q3": values.quantile(0.75),
        "95%": values.quantile(0.95),
        "max": values.max(),
    })

all_dt_summary = pd.DataFrame({name: interval_summary(df["dt_sec"]) for name, df in GAP_DATASETS.items()}).T
print("All timestamp intervals (sec):")
print(all_dt_summary)

large_gap_summary = pd.DataFrame({
    name: interval_summary(df.loc[df["dt_sec"].gt(GAP_THRESHOLD_SEC).fillna(False), "dt_sec"])
    for name, df in GAP_DATASETS.items()
}).T
print(f"Large gap intervals only (dt_sec > {GAP_THRESHOLD_SEC} sec):")
print(large_gap_summary)

for name, df in GAP_DATASETS.items():
    gaps = df.loc[df["dt_sec"].gt(GAP_THRESHOLD_SEC).fillna(False), "dt_sec"]
    print(f"{name}: rounded gap value counts (0.1 sec resolution)")
    print(gaps.round(1).value_counts().sort_index().rename_axis("rounded_dt_sec").to_string())

fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
for col, (name, df) in enumerate(GAP_DATASETS.items()):
    dt_values = df["dt_sec"].dropna()
    axes[0, col].hist(dt_values, bins=60, color="#356a8a", alpha=0.85)
    axes[0, col].axvline(GAP_THRESHOLD_SEC, color="#b34b45", linestyle="--", label="large-gap threshold")
    axes[0, col].set(title=f"{name}: all dt_sec", xlabel="dt_sec (sec)", ylabel="Count")
    axes[0, col].legend()
    axes[1, col].hist(dt_values, bins=np.arange(0, 0.56, 0.05), color="#4d9274", alpha=0.85)
    axes[1, col].axvline(GAP_THRESHOLD_SEC, color="#b34b45", linestyle="--")
    axes[1, col].set(title=f"{name}: 0-0.5 sec detail", xlabel="dt_sec (sec)", ylabel="Count")
plt.show()


gap_position_tables = {}
for name, df in GAP_DATASETS.items():
    gap_mask = df["dt_sec"].gt(GAP_THRESHOLD_SEC).fillna(False)
    gap_rows = df.loc[gap_mask, "source_row"].astype(int).to_numpy()
    row_distance = np.diff(gap_rows)
    table = pd.DataFrame({"gap_source_row": gap_rows})
    table["previous_gap_row_distance"] = np.r_[np.nan, row_distance]
    gap_position_tables[name] = table
    print(f"{name}: gap source rows and distance from previous gap")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(table)
    print("Row-distance value counts:")
    print(pd.Series(row_distance, name="row_distance").value_counts().sort_index().to_frame())
    if len(row_distance):
        print(
            f"Median row distance={np.median(row_distance):.1f}; "
            f"exactly 50 rows={np.mean(row_distance == 50):.1%}; "
            f"within 45-55 rows={np.mean((row_distance >= 45) & (row_distance <= 55)):.1%}"
        )
        print("Gap source row modulo 50:")
        print(pd.Series(gap_rows % 50, name="source_row_mod_50").value_counts().sort_index().to_frame())


normal_gap_mask = normal["dt_sec"].gt(GAP_THRESHOLD_SEC).fillna(False)
normal_gap_rows = np.flatnonzero(normal_gap_mask.to_numpy(dtype=bool))
if len(normal_gap_rows):
    normal_gap_durations = normal.loc[normal_gap_mask, "dt_sec"]
    target_values = [normal_gap_durations.min(), normal_gap_durations.median(), normal_gap_durations.max()]
    selected_gap_positions = []
    for target in target_values:
        candidate = int(np.argmin(np.abs(normal_gap_durations.to_numpy() - target)))
        if candidate not in selected_gap_positions:
            selected_gap_positions.append(candidate)

    fig, axes = plt.subplots(len(selected_gap_positions), len(CHANNELS), figsize=(15, 3.5 * len(selected_gap_positions)), squeeze=False, constrained_layout=True)
    normal_gap_index = np.flatnonzero(normal_gap_mask.to_numpy(dtype=bool))
    for row_plot, gap_number in enumerate(selected_gap_positions):
        idx = int(normal_gap_index[gap_number])
        gap = normal.iloc[idx]
        left = normal.iloc[max(0, idx - 6):idx]
        right = normal.iloc[idx:min(len(normal), idx + 6)]
        gap_start_time = float(normal.iloc[idx - 1]["elapsed_sec"])
        gap_end_time = float(gap["elapsed_sec"])
        for col_plot, channel in enumerate(CHANNELS):
            ax = axes[row_plot, col_plot]
            ax.plot(left["elapsed_sec"] - gap_start_time, left[channel], "o-", label="before gap")
            ax.plot(right["elapsed_sec"] - gap_start_time, right[channel], "o-", label="after gap")
            ax.axvspan(0, gap_end_time - gap_start_time, color="#d88b57", alpha=0.18)
            ax.axvline(0, color="gray", linestyle=":")
            ax.axvline(gap_end_time - gap_start_time, color="gray", linestyle=":")
            ax.set_title(f"source_row={int(gap['source_row'])}, dt={gap['dt_sec']:.2f}s | {channel}")
            ax.set_xlabel("Elapsed time from last pre-gap sample (sec)")
            ax.set_ylabel("Raw signal")
            if row_plot == 0 and col_plot == 0:
                ax.legend()
    plt.show()
    print("Gap windows show raw samples on both sides without drawing a line through the missing interval.")
