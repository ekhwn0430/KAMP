from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

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

# 원본 컬럼명 -> Sensor3 모델링용 짧은 이름. *SIGNALS로 펼치면 원본 컬럼명 리스트가 된다.
SIGNALS = {
    "AI0_Vibration": "v0",
    "AI1_Vibration": "v1",
    "AI2_Current": "cur",
}
GAP_SEC = 0.15

def load_and_clean_file(path, source):
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


# Sensor3에서 사용하던 원본 window feature 생성 함수 묶음이다.
# continuity segment 내부에서만 sliding window를 만들고 통계/관계 feature를 계산한다.
# make_features와 feature_cols가 의존하는 channel_features, corr_rows, 상수까지 함께 포함한다.
# 아래 코드는 예전 preprocess.py의 원본 구조를 그대로 복원한 것이다.

META_COLS = [
    "window_id",
    "segment_id",
    "source",
    "label",
    "split",
    "t_start",
    "t_end",
    "pos_in_seg",
    "seg_len",
]

FS = 10.0
EPS = 1e-12
TEST_EVERY = 5


def load_and_clean():
    """Sensor3 코드용: Normal/Abnormal을 정제·병합하고 v0/v1/cur, label, split 컬럼을 붙인다."""
    frames, diag = [], {}

    for source, path in FILES.items():
        df, diag[source] = load_and_clean_file(path, source)

        # segment 번호 기준 고정 홀드아웃: 5개 segment마다 1개를 test로 사용
        seg_no = df["segment_id"].str[1:].astype(int)
        df["split"] = np.where(seg_no % TEST_EVERY == TEST_EVERY // 2, "test", "train")

        frames.append(df.rename(columns={"Equipment_state": "label", **SIGNALS}))

    return pd.concat(frames, ignore_index=True), diag


def channel_features(x, p, spectral=False):
    """x: (n_window, W)"""
    f = {}

    mu = x.mean(1)
    xc = x - mu[:, None]

    m2 = (xc ** 2).mean(1)
    m3 = (xc ** 3).mean(1)
    m4 = (xc ** 4).mean(1)

    rms = np.sqrt((x ** 2).mean(1))
    absmean = np.abs(x).mean(1)
    absmax = np.abs(x).max(1)

    f[f"{p}_mean"] = mu
    f[f"{p}_std"] = np.sqrt(m2)
    f[f"{p}_rms"] = rms

    f[f"{p}_absmean"] = absmean
    f[f"{p}_absmax"] = absmax

    f[f"{p}_max"] = x.max(1)
    f[f"{p}_min"] = x.min(1)
    f[f"{p}_ptp"] = x.max(1) - x.min(1)

    f[f"{p}_skew"] = m3 / (m2 ** 1.5 + EPS)
    f[f"{p}_kurt"] = m4 / (m2 ** 2 + EPS) - 3

    f[f"{p}_crest"] = absmax / (rms + EPS)
    f[f"{p}_impulse"] = absmax / (absmean + EPS)

    f[f"{p}_diffstd"] = np.diff(x, axis=1).std(1)

    # 평균 교차율
    f[f"{p}_mcr"] = (
        np.diff(np.sign(xc), axis=1) != 0
    ).mean(1)

    # Lag 1~3 autocorrelation
    denom = (xc ** 2).sum(1) + EPS

    for k in (1, 2, 3):
        f[f"{p}_ac{k}"] = (
            xc[:, :-k] * xc[:, k:]
        ).sum(1) / denom

    # FFT feature는 기본적으로 사용하지 않고 spectral=True에서만 생성
    if spectral:
        W = x.shape[1]

        P = np.abs(
            np.fft.rfft(xc, axis=1)
        )[:, 1:] ** 2

        fr = np.fft.rfftfreq(
            W,
            d=1 / FS,
        )[1:]

        tot = P.sum(1) + EPS
        pn = P / tot[:, None]

        f[f"{p}_domfreq"] = fr[P.argmax(1)]

        f[f"{p}_centroid"] = (
            pn * fr
        ).sum(1)

        f[f"{p}_spec_entropy"] = -(
            pn * np.log(pn + EPS)
        ).sum(1) / np.log(len(fr))

        f[f"{p}_hf_ratio"] = (
            P[:, fr >= FS / 4].sum(1)
            / tot
        )

    return f


def corr_rows(a, b):
    ac = a - a.mean(1, keepdims=True)
    bc = b - b.mean(1, keepdims=True)

    return (
        (ac * bc).sum(1)
        / (
            np.sqrt(
                (ac ** 2).sum(1)
                * (bc ** 2).sum(1)
            )
            + EPS
        )
    )


def make_features(data, W, stride=1, spectral=False):
    sig = list(SIGNALS.values())
    out = []

    for seg, g in data.groupby(
        "segment_id",
        sort=False,
    ):
        if len(g) < W:
            continue

        # shape: (n_windows, 3, W)
        win = sliding_window_view(
            g[sig].to_numpy(),
            W,
            axis=0,
        )[::stride]

        starts = np.arange(
            0,
            len(g) - W + 1,
            stride,
        )

        v0 = win[:, 0]
        v1 = win[:, 1]
        cur = win[:, 2]

        feat = {}

        for arr, p in (
            (v0, "v0"),
            (v1, "v1"),
            (cur, "cur"),
        ):
            feat.update(
                channel_features(
                    arr,
                    p,
                    spectral,
                )
            )

        # 채널 간 관계 feature
        feat["corr_v0_v1"] = corr_rows(v0, v1)
        feat["corr_v0_cur"] = corr_rows(v0, cur)
        feat["corr_v1_cur"] = corr_rows(v1, cur)

        feat["rms_diff_v0_v1"] = (
            feat["v0_rms"]
            - feat["v1_rms"]
        )

        feat["rms_ratio_v0_v1"] = (
            feat["v0_rms"]
            / (
                feat["v1_rms"]
                + EPS
            )
        )

        ts = g["TimeStamp"].to_numpy()

        meta = pd.DataFrame({
            "segment_id": seg,
            "source": g["source"].iloc[0],
            "label": g["label"].iloc[0],
            "split": g["split"].iloc[0],

            "t_start": ts[starts],
            "t_end": ts[starts + W - 1],

            # window의 판정 시점은 마지막 sample
            "pos_in_seg": starts + W - 1,

            "seg_len": len(g),
        })

        out.append(
            pd.concat(
                [
                    meta,
                    pd.DataFrame(feat),
                ],
                axis=1,
            )
        )

    if not out:
        raise ValueError(f"W={W}: 생성 가능한 window 없음")

    feats = pd.concat(
        out,
        ignore_index=True,
    )

    feats.insert(
        0,
        "window_id",
        np.arange(len(feats)),
    )

    return feats


def feature_cols(df):
    return [
        c
        for c in df.columns
        if c not in META_COLS
    ]


if __name__ == "__main__":
    print("DATA_DIR:", DATA_DIR)

    frames, diagnostics = [], {}

    for source, path in FILES.items():
        df, diag = load_and_clean_file(path, source)
        frames.append(df)
        diagnostics[source] = diag

    data = pd.concat(frames, ignore_index=True)

    for source, d in diagnostics.items():
        print(f"\n[{source.upper()}]")
        for k, v in d.items():
            print(f"{k}: {v}")

    print(data.head())


    # 전처리 결과 저장
    PROCESSED_DIR = DATA_DIR / "processed"
    PROCESSED_DIR.mkdir(exist_ok=True)

    SAVE_PATH = PROCESSED_DIR / "preprocessed_data.csv"
    data.to_csv(SAVE_PATH, index=False)

    print("저장 완료:", SAVE_PATH)
    print("rows:", len(data))
