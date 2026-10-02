"""
전처리: 원시 CSV -> 정제 -> segment 분할 -> 슬라이딩 윈도우 특징량 -> train/test 분할

실행: python preprocess.py --window 10 --stride 1 [--spectral]
출력: data/processed/clean_timeseries.csv
      data/processed/features_w{W}.csv
      data/processed/excluded_segments_w{W}.csv
      data/processed/diagnosis.json

규칙
- 원본 행 순서 유지, TimeStamp 기준 정렬 금지
- 완전중복 제거
- gap(>0.15초) 또는 비증가 시간에서 segment 분리
- segment를 넘어 특징 계산하지 않음
- 전류 음수값 유지
- FFT 기본 보류
- 스케일링은 모델 단계 fold별 train에만 fit
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent
RAW, OUT = ROOT / "data", ROOT / "data" / "processed"

FILES = {"normal": "press_data_normal.csv", "abnormal": "outlier_data.csv"}
SIGNALS = {"AI0_Vibration": "v0", "AI1_Vibration": "v1", "AI2_Current": "cur"}
META_COLS = ["window_id", "segment_id", "source", "label", "split",
             "t_start", "t_end", "pos_in_seg", "seg_len"]
FS, GAP_SEC, TEST_EVERY, EPS = 10.0, 0.15, 5, 1e-12


# ---------------------------------------------------------------- 1. 로드·정제
def load_and_clean():
    frames, diag = [], {}

    for src, fname in FILES.items():
        df = pd.read_csv(RAW / fname)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
        df.insert(0, "source_row", np.arange(len(df)))
        n0 = len(df)

        required = ["TimeStamp", *SIGNALS, "Equipment_state"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{fname}: missing columns {missing_cols}")

        n_missing = int(df[required].isna().sum().sum())
        df = df.dropna(subset=required).copy()
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="raise")

        dup_cols = ["TimeStamp", *SIGNALS, "Equipment_state"]
        dup = df.duplicated(subset=dup_cols)
        n_dup = int(dup.sum())
        df = df.loc[~dup].reset_index(drop=True)

        df["elapsed_sec"] = (df.TimeStamp - df.TimeStamp.iloc[0]).dt.total_seconds()
        df["dt_sec"] = df.TimeStamp.diff().dt.total_seconds()

        gap = df.dt_sec.gt(GAP_SEC)
        non_inc = df.dt_sec.le(0)
        seg_no = (gap | non_inc).cumsum()

        prefix = "N" if src == "normal" else "A"
        df["segment_id"] = prefix + seg_no.astype(str).str.zfill(4)
        df["pos_in_seg"] = df.groupby("segment_id").cumcount()
        df["seg_len"] = df.groupby("segment_id").segment_id.transform("size")
        df["split"] = np.where(seg_no % TEST_EVERY == TEST_EVERY // 2, "test", "train")
        df["source"] = src
        df = df.rename(columns={"Equipment_state": "label", **SIGNALS})

        seg_len = df.groupby("segment_id").size()
        valid_dt = df.dt_sec.dropna()
        diag[src] = {
            "rows_raw": n0, "missing": n_missing, "duplicates_removed": n_dup,
            "rows_clean": len(df),
            "time_start": str(df.TimeStamp.iloc[0]), "time_end": str(df.TimeStamp.iloc[-1]),
            "median_dt_sec": float(valid_dt.median()),
            "large_gaps": int(gap.sum()), "non_increasing": int(non_inc.sum()),
            "n_segments": int(seg_len.size),
            "segment_len": seg_len.describe().round(2).to_dict(),
            "n_test_segments": int(df.loc[df.split == "test", "segment_id"].nunique()),
        }
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    sig = list(SIGNALS.values())

    # 정상 기준 |z|>4 진단만 수행, 제거하지 않음
    ref = data.loc[data.source == "normal", sig]
    z = (data[sig] - ref.mean()) / ref.std()
    for src in FILES:
        m = data.source == src
        diag[src]["signal_stats"] = data.loc[m, sig].describe().round(4).to_dict()
        diag[src]["n_abs_z_gt4_vs_normal"] = (z.loc[m].abs() > 4).sum().astype(int).to_dict()
        diag[src]["n_negative"] = (data.loc[m, sig] < 0).sum().astype(int).to_dict()

    return data, diag


# ---------------------------------------------------------------- 2. 특징량
def channel_features(x, p, spectral=False):
    f = {}
    mu = x.mean(1)
    xc = x - mu[:, None]
    m2, m3, m4 = (xc**2).mean(1), (xc**3).mean(1), (xc**4).mean(1)
    rms, absmean, absmax = np.sqrt((x**2).mean(1)), np.abs(x).mean(1), np.abs(x).max(1)

    f[f"{p}_mean"] = mu
    f[f"{p}_std"] = np.sqrt(m2)
    f[f"{p}_rms"] = rms
    f[f"{p}_absmean"] = absmean
    f[f"{p}_absmax"] = absmax
    f[f"{p}_max"] = x.max(1)
    f[f"{p}_min"] = x.min(1)
    f[f"{p}_ptp"] = x.max(1) - x.min(1)
    f[f"{p}_skew"] = m3 / (m2**1.5 + EPS)
    f[f"{p}_kurt"] = m4 / (m2**2 + EPS) - 3
    f[f"{p}_crest"] = absmax / (rms + EPS)
    f[f"{p}_impulse"] = absmax / (absmean + EPS)
    f[f"{p}_diffstd"] = np.diff(x, axis=1).std(1)
    f[f"{p}_mcr"] = (np.diff(np.sign(xc), axis=1) != 0).mean(1)

    denom = (xc**2).sum(1) + EPS
    for k in (1, 2, 3):
        f[f"{p}_ac{k}"] = (xc[:, :-k] * xc[:, k:]).sum(1) / denom

    if spectral:
        W = x.shape[1]
        P = np.abs(np.fft.rfft(xc, axis=1))[:, 1:] ** 2
        fr = np.fft.rfftfreq(W, d=1 / FS)[1:]
        tot = P.sum(1) + EPS
        pn = P / tot[:, None]
        f[f"{p}_domfreq"] = fr[P.argmax(1)]
        f[f"{p}_centroid"] = (pn * fr).sum(1)
        f[f"{p}_spec_entropy"] = -(pn * np.log(pn + EPS)).sum(1) / np.log(len(fr))
        f[f"{p}_hf_ratio"] = P[:, fr >= FS / 4].sum(1) / tot

    return f


def corr_rows(a, b):
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (ac * bc).sum(1) / (np.sqrt((ac**2).sum(1) * (bc**2).sum(1)) + EPS)


def make_features(data, W, stride=1, spectral=False):
    sig, out = list(SIGNALS.values()), []

    for seg, g in data.groupby("segment_id", sort=False):
        if len(g) < W:
            continue

        win = sliding_window_view(g[sig].to_numpy(), W, axis=0)[::stride]
        starts = np.arange(0, len(g) - W + 1, stride)
        v0, v1, cur = win[:, 0], win[:, 1], win[:, 2]

        feat = {}
        for arr, p in ((v0, "v0"), (v1, "v1"), (cur, "cur")):
            feat.update(channel_features(arr, p, spectral))

        feat["corr_v0_v1"] = corr_rows(v0, v1)
        feat["corr_v0_cur"] = corr_rows(v0, cur)
        feat["corr_v1_cur"] = corr_rows(v1, cur)
        feat["rms_diff_v0_v1"] = feat["v0_rms"] - feat["v1_rms"]
        feat["rms_ratio_v0_v1"] = feat["v0_rms"] / (feat["v1_rms"] + EPS)

        ts = g.TimeStamp.to_numpy()
        meta = pd.DataFrame({
            "segment_id": seg, "source": g.source.iloc[0], "label": g.label.iloc[0],
            "split": g.split.iloc[0], "t_start": ts[starts], "t_end": ts[starts + W - 1],
            "pos_in_seg": starts + W - 1, "seg_len": len(g),
        })
        out.append(pd.concat([meta, pd.DataFrame(feat)], axis=1))

    if not out:
        raise ValueError(f"W={W}: 생성 가능한 window 없음")

    feats = pd.concat(out, ignore_index=True)
    feats.insert(0, "window_id", np.arange(len(feats)))
    return feats


def feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


# ---------------------------------------------------------------- 3. 실행
def main(window=10, stride=1, spectral=False, verbose=True):
    OUT.mkdir(parents=True, exist_ok=True)
    data, diag = load_and_clean()
    feats = make_features(data, window, stride, spectral)

    seg = data.groupby("segment_id").agg(
        source=("source", "first"), label=("label", "first"),
        split=("split", "first"), seg_len=("seg_len", "first"))
    excluded = seg[seg.seg_len < window].reset_index()

    diag["windowing"] = {
        "window": window, "stride": stride, "spectral": spectral,
        "excluded_segments": excluded.groupby("source").size().to_dict(),
        "excluded_rows": excluded.groupby("source").seg_len.sum().to_dict(),
        "n_windows": feats.groupby(["split", "source"]).size().unstack().to_dict(),
        "n_features": len(feature_cols(feats)),
    }

    data.to_csv(OUT / "clean_timeseries.csv", index=False)
    feats.to_csv(OUT / f"features_w{window}.csv", index=False)
    excluded.to_csv(OUT / f"excluded_segments_w{window}.csv", index=False)
    with open(OUT / "diagnosis.json", "w", encoding="utf-8") as fp:
        json.dump(diag, fp, ensure_ascii=False, indent=2, default=str)

    if verbose:
        for src in FILES:
            d = diag[src]
            print(f"[{src}] raw={d['rows_raw']}, clean={d['rows_clean']}, "
                  f"dup={d['duplicates_removed']}, dt={d['median_dt_sec']:.4f}s, "
                  f"gaps={d['large_gaps']}, non_inc={d['non_increasing']}, "
                  f"segments={d['n_segments']}")
        print(json.dumps(diag["windowing"], indent=2, ensure_ascii=False, default=str))

    return data, feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--spectral", action="store_true")
    a = ap.parse_args()
    main(a.window, a.stride, a.spectral)