from pathlib import Path
import numpy as np
import pandas as pd

def find_data_dir():
    for root in [Path.cwd(), *Path.cwd().parents]:
        d = root / "data"
        if (d / "press_data_normal.csv").exists() and (d / "outlier_data.csv").exists():
            return d
    raise FileNotFoundError("data 폴더에서 press_data_normal.csv / outlier_data.csv를 찾지 못함")

DATA_DIR = find_data_dir()

FILES = {
    "normal": DATA_DIR / "press_data_normal.csv",
    "abnormal": DATA_DIR / "outlier_data.csv",
}

SIGNALS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
GAP_SEC = 0.15

print("DATA_DIR:", DATA_DIR)

SIGNALS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
GAP_SEC = 0.15

def load_and_clean(path, source):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    df["source_row"] = np.arange(len(df))
    n_raw = len(df)

    required = ["TimeStamp", *SIGNALS, "Equipment_state"]
    n_missing = int(df[required].isna().sum().sum())
    df = df.dropna(subset=required).copy()

    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    n_bad_time = int(df["TimeStamp"].isna().sum())
    if n_bad_time:
        raise ValueError(f"{source}: TimeStamp 파싱 실패 {n_bad_time}건")

    dup_cols = ["TimeStamp", *SIGNALS, "Equipment_state"]
    dup = df.duplicated(subset=dup_cols)
    n_dup = int(dup.sum())
    df = df.loc[~dup].reset_index(drop=True)

    df["elapsed_sec"] = (df["TimeStamp"] - df["TimeStamp"].iloc[0]).dt.total_seconds()
    df["dt_sec"] = df["TimeStamp"].diff().dt.total_seconds()

    gap = df["dt_sec"] > GAP_SEC
    non_inc = df["dt_sec"] <= 0
    seg_no = (gap | non_inc).cumsum()

    prefix = "N" if source == "normal" else "A"
    df["segment_id"] = prefix + seg_no.astype(str).str.zfill(4)
    df["pos_in_seg"] = df.groupby("segment_id", sort=False).cumcount()
    df["seg_len"] = df.groupby("segment_id", sort=False)["segment_id"].transform("size")
    df["source"] = source

    diag = {
        "rows_raw": n_raw,
        "missing": n_missing,
        "duplicates_removed": n_dup,
        "rows_clean": len(df),
        "time_start": df["TimeStamp"].iloc[0],
        "time_end": df["TimeStamp"].iloc[-1],
        "median_dt_sec": df["dt_sec"].dropna().median(),
        "large_gaps": int(gap.sum()),
        "non_increasing": int(non_inc.sum()),
        "segments": df["segment_id"].nunique(),
    }
    return df, diag

frames, diagnostics = [], {}

for source, path in FILES.items():
    df, diag = load_and_clean(path, source)
    frames.append(df)
    diagnostics[source] = diag

data = pd.concat(frames, ignore_index=True)

for source, d in diagnostics.items():
    print(f"\n[{source.upper()}]")
    for k, v in d.items():
        print(f"{k}: {v}")

display(data.head())


# 전처리 결과 저장
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

SAVE_PATH = PROCESSED_DIR / "preprocessed_data.csv"
data.to_csv(SAVE_PATH, index=False)

print("저장 완료:", SAVE_PATH)
print("rows:", len(data))
