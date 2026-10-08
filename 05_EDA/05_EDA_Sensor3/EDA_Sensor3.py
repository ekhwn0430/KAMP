"""RMS 미검출 이상 분석"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from preprocess import load_and_clean, make_features, feature_cols

try: ROOT = Path(__file__).resolve().parent
except NameError: ROOT = Path.cwd()

OUT = ROOT / "results" / "undetected"
RMS_COLS = ["v0_rms", "v1_rms", "cur_rms"]
Q_LO, Q_HI = .005, .995

HYPOTHESES = [
    ("H1_전류-진동_관계이탈", ["v0_resid", "v1_resid"]),
    ("H2_진동채널_관계", ["corr_v0_v1"]),
    ("H3a_전류_크기", ["cur_mean", "cur_std", "cur_min"]),
    ("H3b_전류_시간구조", ["cur_ac1", "cur_mcr"]),
]

def rms_iqr_flag(feats):
    normal = feats[feats.label == 0]
    q1, q3 = normal[RMS_COLS].quantile(.25), normal[RMS_COLS].quantile(.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((feats[RMS_COLS] < lo) | (feats[RMS_COLS] > hi)).any(axis=1)

def add_residuals(feats):
    feats = feats.copy()
    tr = feats[(feats.label == 0) & (feats.split == "train")]
    info = {}
    for v in ["v0", "v1"]:
        x, y = tr["cur_rms"], tr[f"{v}_rms"]
        b = np.polyfit(x, y, 1)
        pred_tr = np.polyval(b, x)
        resid_tr = y - pred_tr
        feats[f"{v}_resid"] = (feats[f"{v}_rms"] - np.polyval(b, feats["cur_rms"])) / resid_tr.std()
        info[v] = {"slope": b[0], "intercept": b[1], "R2": 1 - resid_tr.var() / y.var()}
    return feats, info

def rule_flag(feats, cols):
    tr = feats[(feats.label == 0) & (feats.split == "train")]
    flag = pd.Series(False, index=feats.index)
    for c in cols:
        lo, hi = tr[c].quantile(Q_LO), tr[c].quantile(Q_HI)
        flag |= (feats[c] < lo) | (feats[c] > hi)
    return flag

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, _ = load_and_clean()
    print("=" * 70, "\n데이터 로드 완료\n", "=" * 70, sep="")
    print(data.groupby("source").size().to_string(), "\n")

    # 1. Window별 RMS-IQR
    rows = []
    n_seg = data.groupby("label")["segment_id"].nunique()

    for W in (5, 10, 20, 50):
        g = make_features(data, W)
        g["flag"] = rms_iqr_flag(g)
        A, N = g.label == 1, g.label == 0
        seg = g.groupby("segment_id").agg(label=("label", "first"), alarm=("flag", "max"))

        rows.append({
            "window": W, "seconds": W / 10,
            "abnormal_rows_evaluable": int(A.sum()),
            "abnormal_rows_not_evaluable": int((data.label == 1).sum() - A.sum()),
            "abnormal_segments_evaluable": f"{(seg.label == 1).sum()}/{n_seg[1]}",
            "normal_segments_evaluable": f"{(seg.label == 0).sum()}/{n_seg[0]}",
            "row_detect_rate": g.loc[A, "flag"].mean(),
            "abnormal_segment_detect_rate": seg.loc[seg.label == 1, "alarm"].mean(),
            "normal_row_false_alarm_rate": g.loc[N, "flag"].mean(),
            "normal_segment_false_alarm_rate": seg.loc[seg.label == 0, "alarm"].mean(),
        })

    by_window = pd.DataFrame(rows)
    by_window.to_csv(OUT / "baseline_by_window.csv", index=False)

    # 2. W=10 + H1 residual
    feats = make_features(data, 10)
    feats["rms_flag"] = rms_iqr_flag(feats)
    feats, resid_info = add_residuals(feats)

    A, N = feats.label == 1, feats.label == 0
    Nt = N & (feats.split == "test")
    U = A & ~feats.rms_flag

    # 3. RMS 미검출 vs 정상 Feature AUC
    y = np.r_[np.zeros(N.sum()), np.ones(U.sum())]
    auc_rows = []

    for c in [c for c in feature_cols(feats) if c != "rms_flag"]:
        values = np.r_[feats.loc[N, c], feats.loc[U, c]]
        valid = np.isfinite(values)
        if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
            continue

        auc_raw = roc_auc_score(y[valid], values[valid])
        auc_rows.append({
            "feature": c,
            "auc": max(auc_raw, 1 - auc_raw),
            "direction": "higher_in_undetected" if auc_raw >= .5 else "lower_in_undetected",
            "normal_median": feats.loc[N, c].median(),
            "undetected_median": feats.loc[U, c].median(),
            "detected_median": feats.loc[A & feats.rms_flag, c].median(),
        })

    auc_df = pd.DataFrame(auc_rows).sort_values("auc", ascending=False)
    auc_df.to_csv(OUT / "feature_auc_undetected_vs_normal.csv", index=False)

    # 4. H1~H3 누적 설명력
    cum = feats["rms_flag"].copy()
    feats["explained_by"] = np.where(feats.rms_flag, "RMS_baseline", "unexplained")
    cov = [{
        "step": "RMS_baseline",
        "abnormal_detected": int(cum[A].sum()),
        "added": int(cum[A].sum()),
        "normal_test_false_alarm": cum[Nt].mean(),
    }]

    for name, cols in HYPOTHESES:
        rule = rule_flag(feats, cols)
        new = rule & ~cum
        feats.loc[new & A, "explained_by"] = name
        cum |= new
        cov.append({
            "step": f"+{name}",
            "abnormal_detected": int(cum[A].sum()),
            "added": int((new & A).sum()),
            "normal_test_false_alarm": cum[Nt].mean(),
        })

    cov_df = pd.DataFrame(cov)
    cov_df.to_csv(OUT / "hypothesis_coverage.csv", index=False)

    # 5. 이상 행별 상태
    ab = data.loc[data.label == 1, ["TimeStamp", "segment_id", "pos_in_seg", "v0", "v1", "cur"]].copy()
    st = feats.loc[A, ["segment_id", "pos_in_seg", "explained_by"]]
    ab = ab.merge(st, on=["segment_id", "pos_in_seg"], how="left")
    ab["explained_by"] = ab["explained_by"].fillna("not_evaluable")
    ab.to_csv(OUT / "abnormal_rows_status.csv", index=False)

    # 6. 이상 Segment별 요약
    seg = ab.groupby(["segment_id", "explained_by"]).size().unstack(fill_value=0)
    med_cols = ["v0_rms", "v1_rms", "cur_rms", "v1_resid", "corr_v0_v1", "cur_mean", "cur_ac1"]
    med = feats[A].groupby("segment_id")[med_cols].median()
    seg = seg.join(med).round(3)
    seg.to_csv(OUT / "segment_summary.csv")

    # 7. 출력
    pd.set_option("display.width", 200)
    print("\n[윈도우 크기별 베이스라인]\n", by_window.round(4).to_string(index=False))
    print("\n[H1 잔차 회귀 (정상 train)]")
    print({k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in resid_info.items()})
    print("\n[미검출 vs 정상 분리도 상위 10]\n", auc_df.head(10).round(3).to_string(index=False))
    print(f"\n[가설 누적 설명력] (판정 가능 이상 {A.sum()}행)\n", cov_df.round(4).to_string(index=False))
    print("\n[이상 600행 판정 상태]\n", ab["explained_by"].value_counts().to_string())
    print("\n[이상 구간별 요약]\n", seg.to_string())

    return {
        "data": data, "features": feats, "window_result": by_window,
        "auc_result": auc_df, "coverage": cov_df,
        "abnormal_rows": ab, "segment_summary": seg,
    }

if __name__ == "__main__":
    results = main()


"""
전처리: 원시 CSV -> 정제 -> 측정구간(segment) 분할 -> 슬라이딩 윈도우 특징량 -> train/test 분할

실행:  python preprocess.py --window 10 --stride 1 [--spectral]
출력:  data/processed/clean_timeseries.csv
       data/processed/features_w{W}.csv
       data/processed/excluded_segments_w{W}.csv   (윈도우보다 짧아 제외된 구간)
       data/processed/diagnosis.json

팀 합의 규칙
- 시간 공백을 넘어 계산하지 않음: 모든 특징은 segment 내부, 과거 방향(trailing) 윈도우로만 계산
- 전류 음수값은 그대로 유지
- FFT(스펙트럼) 특징은 보류: --spectral 옵션을 줄 때만 생성
- 시간·파일명·행번호·구간길이는 모델 입력에서 제외 (META_COLS)
- 스케일링은 모델 단계에서 fold별 train에만 fit
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# .py 실행 / Jupyter 실행 둘 다 지원
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

RAW = ROOT / "data"
OUT = ROOT / "data" / "processed"

FILES = {"normal": "press_data_normal.csv", "abnormal": "outlier_data.csv"}
SIGNALS = {"AI0_Vibration": "v0", "AI1_Vibration": "v1", "AI2_Current": "cur"}
META_COLS = ["window_id", "segment_id", "source", "label", "split",
             "t_start", "t_end", "pos_in_seg", "seg_len"]
FS = 10.0          # 관측 간격 0.1초 기준
GAP_SEC = 0.15     # 이 간격을 넘으면 새 측정구간
TEST_EVERY = 5     # 시간순 구간 5개마다 1개를 test로 (약 20%)
EPS = 1e-12


# ---------------------------------------------------------------- 1. 로드·정제

def load_and_clean():
    frames, diag = [], {}

    for src, fname in FILES.items():
        df = pd.read_csv(RAW / fname)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
        df["source_row"] = np.arange(len(df))
        n0 = len(df)

        required = ["TimeStamp", *SIGNALS, "Equipment_state"]
        n_missing = int(df[required].isna().sum().sum())
        df = df.dropna(subset=required).copy()

        # full datetime 원본 사용
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
        bad_ts = df["TimeStamp"].isna()
        if bad_ts.any():
            raise ValueError(f"TimeStamp 파싱 실패 {int(bad_ts.sum())}건")

        # 완전중복 제거
        dup_cols = ["TimeStamp", *SIGNALS, "Equipment_state"]
        dup = df.duplicated(subset=dup_cols)
        n_dup = int(dup.sum())
        df = df.loc[~dup].reset_index(drop=True)

        # 원본 순서 그대로 시간축 계산
        df["elapsed_sec"] = (df.TimeStamp - df.TimeStamp.iloc[0]).dt.total_seconds()
        df["dt_sec"] = df.TimeStamp.diff().dt.total_seconds()

        gap = df.dt_sec.gt(GAP_SEC)
        non_inc = df.dt_sec.le(0)
        seg_no = (gap | non_inc).cumsum()

        prefix = "N" if src == "normal" else "A"
        df["segment_id"] = prefix + seg_no.astype(str).str.zfill(4)
        df["pos_in_seg"] = df.groupby("segment_id", sort=False).cumcount()
        df["seg_len"] = df.groupby("segment_id", sort=False).segment_id.transform("size")
        df["split"] = np.where(seg_no % TEST_EVERY == TEST_EVERY // 2, "test", "train")
        df["source"] = src
        df = df.rename(columns={"Equipment_state": "label", **SIGNALS})

        seg_len = df.groupby("segment_id", sort=False).size()
        valid_dt = df.dt_sec.dropna()
        diag[src] = {
            "rows_raw": n0, "missing": n_missing, "duplicates_removed": n_dup,
            "rows_clean": len(df),
            "time_start": str(df.TimeStamp.iloc[0]), "time_end": str(df.TimeStamp.iloc[-1]),
            "median_dt_sec": float(valid_dt.median()) if len(valid_dt) else None,
            "large_gaps": int(gap.sum()), "non_increasing": int(non_inc.sum()),
            "n_segments": int(seg_len.size),
            "segment_len": seg_len.describe().round(2).to_dict(),
            "n_test_segments": int(df.loc[df.split == "test", "segment_id"].nunique()),
        }
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    sig = list(SIGNALS.values())

    ref = data.loc[data.source == "normal", sig]
    ref_std = ref.std().replace(0, np.nan)
    z = (data[sig] - ref.mean()) / ref_std

    for src in FILES:
        m = data.source == src
        diag[src]["signal_stats"] = data.loc[m, sig].describe().round(4).to_dict()
        diag[src]["n_abs_z_gt4_vs_normal"] = (z.loc[m].abs() > 4).sum().astype(int).to_dict()
        diag[src]["n_negative"] = (data.loc[m, sig] < 0).sum().astype(int).to_dict()

    return data, diag

# ---------------------------------------------------------------- 2. 특징량
def channel_features(x, p, spectral=False):
    """x: (n_window, W)"""
    f = {}
    mu = x.mean(1)
    xc = x - mu[:, None]
    m2, m3, m4 = (xc**2).mean(1), (xc**3).mean(1), (xc**4).mean(1)
    rms = np.sqrt((x**2).mean(1))
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
    f[f"{p}_skew"] = m3 / (m2**1.5 + EPS)
    f[f"{p}_kurt"] = m4 / (m2**2 + EPS) - 3
    f[f"{p}_crest"] = absmax / (rms + EPS)
    f[f"{p}_impulse"] = absmax / (absmean + EPS)
    f[f"{p}_diffstd"] = np.diff(x, axis=1).std(1)
    f[f"{p}_mcr"] = (np.diff(np.sign(xc), axis=1) != 0).mean(1)   # 평균 교차율
    denom = (xc**2).sum(1) + EPS
    for k in (1, 2, 3):
        f[f"{p}_ac{k}"] = (xc[:, :-k] * xc[:, k:]).sum(1) / denom

    if spectral:   # 팀 결정으로 보류, 비교용
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
    sig = list(SIGNALS.values())
    out = []
    for seg, g in data.groupby("segment_id", sort=False):
        if len(g) < W:
            continue
        win = sliding_window_view(g[sig].to_numpy(), W, axis=0)[::stride]   # (n, 3, W)
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

        ts = g["TimeStamp"].to_numpy()
        meta = pd.DataFrame({
            "segment_id": seg,
            "source": g["source"].iloc[0],
            "label": g["label"].iloc[0],
            "split": g["split"].iloc[0],
            "t_start": ts[starts],
            "t_end": ts[starts + W - 1],       # 윈도우 판정 시점 = 마지막 샘플
            "pos_in_seg": starts + W - 1,
            "seg_len": len(g),
        })
        out.append(pd.concat([meta, pd.DataFrame(feat)], axis=1))

    feats = pd.concat(out, ignore_index=True)
    feats.insert(0, "window_id", np.arange(len(feats)))
    return feats


def feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


# ---------------------------------------------------------------- main
def main(window=10, stride=1, spectral=False, verbose=True):
    OUT.mkdir(parents=True, exist_ok=True)
    data, diag = load_and_clean()
    feats = make_features(data, window, stride, spectral)

    seg = data.groupby("segment_id").agg(source=("source", "first"), label=("label", "first"),
                                         split=("split", "first"), seg_len=("seg_len", "first"))
    excluded = seg[seg.seg_len < window].reset_index()

    diag["windowing"] = {
        "window": window, "stride": stride, "spectral": spectral,
        "excluded_segments": excluded.groupby("source").size().to_dict(),
        "excluded_rows": excluded.groupby("source")["seg_len"].sum().to_dict(),
        "n_windows": feats.groupby(["split", "source"]).size().unstack().to_dict(),
        "n_features": len(feature_cols(feats)),
    }

    data.to_csv(OUT / "clean_timeseries.csv", index=False)
    feats.to_csv(OUT / f"features_w{window}.csv", index=False)
    excluded.to_csv(OUT / f"excluded_segments_w{window}.csv", index=False)
    with open(OUT / "diagnosis.json", "w", encoding="utf-8") as fp:
        json.dump(diag, fp, ensure_ascii=False, indent=2, default=str)

    if verbose:
        print(json.dumps(diag["windowing"], indent=2, default=str))
    return data, feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--spectral",
        action="store_true",
        help="FFT 특징 포함 (기본 제외)"
    )
    a = ap.parse_args()
    main(a.window, a.stride, a.spectral)


"""
모델 비교
- test: preprocess.py의 고정 segment holdout
- train: StratifiedGroupKFold 5-fold, group=segment_id
- threshold: OOF F1 최대값
- H1 residual / scaling / IQR: 각 fold train에서만 fit
- F5 전류 시간구조는 참고용, 최종 후보 선정 제외
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import load_and_clean, make_features, feature_cols


# ---------------------------------------------------------------- 설정
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

OUT = ROOT / "results" / "models"
SEED, WINDOW, N_FOLDS = 42, 10, 5
EXCLUDE_FROM_SELECTION = ["F5_+전류시간구조"]


# ---------------------------------------------------------------- 특징 세트
def feature_sets(cols):
    cur_time = ["cur_ac1", "cur_ac2", "cur_ac3", "cur_mcr", "cur_diffstd"]
    vib = [c for c in cols if c.startswith(("v0_", "v1_"))]
    cur_mag = [c for c in cols if c.startswith("cur_") and c not in cur_time]
    rel = ["corr_v0_v1",
        "corr_v0_cur", "corr_v1_cur",
        "rms_diff_v0_v1", "rms_ratio_v0_v1"]
    base = vib + cur_mag + rel

    return {
        "F1_진동": (vib, False),
        "F2_+전류크기": (vib + cur_mag, False),
        "F3_+채널관계": (base, False),
        "F4_+H1잔차": (base, True),
        "F5_+전류시간구조": (base + cur_time, True),
    }


# ---------------------------------------------------------------- H1 residual
class H1Residual(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        n = X.loc[np.asarray(y) == 0]
        self.coef_, self.std_ = {}, {}

        for v in ("v0", "v1"):
            coef = np.polyfit(n["cur_rms"], n[f"{v}_rms"], 1)
            resid = n[f"{v}_rms"] - np.polyval(coef, n["cur_rms"])
            self.coef_[v] = coef
            self.std_[v] = max(float(resid.std()), 1e-12)
        return self

    def transform(self, X):
        X = X.copy()
        for v in ("v0", "v1"):
            expected = np.polyval(self.coef_[v], X["cur_rms"])
            X[f"{v}_resid"] = (X[f"{v}_rms"] - expected) / self.std_[v]
        return X


# ---------------------------------------------------------------- RMS-IQR baseline
class RMSIQRRule(BaseEstimator, ClassifierMixin):
    cols = ["v0_rms", "v1_rms", "cur_rms"]

    def fit(self, X, y):
        n = X.loc[np.asarray(y) == 0, self.cols]
        q1, q3 = n.quantile(.25), n.quantile(.75)
        self.iqr_ = (q3 - q1).replace(0, 1e-12)
        self.lo_ = q1 - 1.5 * self.iqr_
        self.hi_ = q3 + 1.5 * self.iqr_
        return self

    def decision_function(self, X):
        x = X[self.cols]
        upper = (x - self.hi_) / self.iqr_
        lower = (self.lo_ - x) / self.iqr_
        return np.maximum(upper, lower).max(axis=1).to_numpy()


# ---------------------------------------------------------------- Isolation Forest
class NormalOnlyIForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=300, random_state=SEED):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        n = X.loc[np.asarray(y) == 0]
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        ).fit(n)
        return self

    def decision_function(self, X):
        return -self.model_.score_samples(X)


# ---------------------------------------------------------------- 모델
MODELS = {
    "LogReg": make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
    ),
    "HistGB": HistGradientBoostingClassifier(
        max_iter=300, learning_rate=.05, max_leaf_nodes=15,
        class_weight="balanced", random_state=SEED
    ),
    "IsolationForest": NormalOnlyIForest(),
}


def build(model, use_resid):
    return make_pipeline(H1Residual(), clone(model)) if use_resid else clone(model)


def score(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    if len(t) == 0:
        return 0.0
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[np.nanargmax(f1)])


def evaluate(y, s, thr, seg, prefix):
    y, s, seg = np.asarray(y), np.asarray(s), np.asarray(seg)
    ok = np.isfinite(s)
    y, s, seg = y[ok], s[ok], seg[ok]

    pred = (s >= thr).astype(int)
    df = pd.DataFrame({"y": y, "pred": pred, "seg": seg})
    sg = df.groupby("seg").agg(y=("y", "first"), alarm=("pred", "max"))

    return {
        f"{prefix}_PR_AUC": average_precision_score(y, s),
        f"{prefix}_F1": f1_score(y, pred, zero_division=0),
        f"{prefix}_Recall": recall_score(y, pred, zero_division=0),
        f"{prefix}_Precision": precision_score(y, pred, zero_division=0),
        f"{prefix}_FalseAlarmRate": pred[y == 0].mean(),
        f"{prefix}_AbnSegDetect": sg.loc[sg.y == 1, "alarm"].mean(),
        f"{prefix}_NormSegFalseAlarm": sg.loc[sg.y == 0, "alarm"].mean(),
    }


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data, _ = load_and_clean()
    f = make_features(data, WINDOW)

    tr = f[f.split == "train"].reset_index(drop=True)
    te = f[f.split == "test"].reset_index(drop=True)
    ytr, yte = tr.label.to_numpy(), te.label.to_numpy()

    print(f"Normal   : {(data.source == 'normal').sum():,} rows / "
          f"{data.loc[data.source == 'normal', 'segment_id'].nunique()} segments")
    print(f"Abnormal : {(data.source == 'abnormal').sum():,} rows / "
          f"{data.loc[data.source == 'abnormal', 'segment_id'].nunique()} segments")
    print(f"Train {len(tr):,} / Test {len(te):,} / Features {len(feature_cols(f))}")

    folds = list(StratifiedGroupKFold(
        N_FOLDS, shuffle=True, random_state=SEED
    ).split(tr, ytr, tr.segment_id))

    runs = [("RMS-IQR", "F0_RMS", RMSIQRRule(), RMSIQRRule.cols, False)]

    for fs_name, (cols, resid) in feature_sets(feature_cols(f)).items():
        for m_name, model in MODELS.items():
            runs.append((m_name, fs_name, model, cols, resid))

    rows, test_scores = [], {}

    for m_name, fs_name, model, cols, resid in runs:
        oof = np.full(len(tr), np.nan)

        for itr, iva in folds:
            est = build(model, resid).fit(tr.loc[itr, cols], ytr[itr])
            oof[iva] = score(est, tr.loc[iva, cols])

        thr = 0.0 if m_name == "RMS-IQR" else best_f1_threshold(ytr, oof)

        est = build(model, resid).fit(tr[cols], ytr)
        s_te = score(est, te[cols])
        test_scores[(m_name, fs_name)] = (s_te, thr)

        result = {
            "model": m_name,
            "features": fs_name,
            "threshold": thr,
            **evaluate(ytr, oof, thr, tr.segment_id, "CV"),
            **evaluate(yte, s_te, thr, te.segment_id, "TEST")
        }
        rows.append(result)

        print(
            f"{m_name:16s} {fs_name:18s} "
            f"CV PR-AUC {result['CV_PR_AUC']:.3f}  "
            f"CV F1 {result['CV_F1']:.3f}  "
            f"TEST F1 {result['TEST_F1']:.3f}"
        )

    # 결과 저장
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "model_results.csv", index=False)

    # F5와 RMS-IQR 제외 후 CV PR-AUC 최고 모델 선정
    cand = res[
        ~res.features.isin(EXCLUDE_FROM_SELECTION)
        & (res.model != "RMS-IQR")
    ]

    best = cand.sort_values("CV_PR_AUC", ascending=False).iloc[0]
    s_te, thr = test_scores[(best.model, best.features)]

    pred = te[["window_id", "segment_id", "t_start", "t_end", "label"]].copy()
    pred["anomaly_score"] = s_te
    pred["alarm"] = (s_te >= thr).astype(int)
    pred.to_csv(OUT / "test_predictions.csv", index=False)

    print("\n최종 후보")
    print(f"{best.model} / {best.features}")
    print(f"threshold : {thr:.4f}")
    print(f"CV PR-AUC : {best.CV_PR_AUC:.4f}")
    print(f"CV F1     : {best.CV_F1:.4f}")
    print(f"TEST F1   : {best.TEST_F1:.4f}")

    return res, pred


if __name__ == "__main__":
    results, test_predictions = main()
