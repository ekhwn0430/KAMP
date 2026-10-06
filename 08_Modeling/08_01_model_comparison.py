"""
08_01 모델 비교 — 1초 윈도우 단위

예측 단위
segment 내부 trailing 1초 윈도우(10샘플, stride 1). 판정 시점 = 윈도우 마지막 행(source_row).
07 cycle table은 Abnormal cycle이 8개뿐이라 학습·검증 단위로 쓰지 않음.

데이터·키
공식 전처리(04_Preprocessing) 결과 사용: Normal 19,999행/599 segment, Abnormal 600행/21 segment
key = source / segment_id / source_row (/ pos_in_seg)

분할
test : source별 시간순 segment 5개마다 1개(segment 번호 % 5 == 2) 고정 홀드아웃 → 최종 평가에만 사용
train: segment 단위 StratifiedGroupKFold 5-fold → out-of-fold(OOF) 예측으로 모델 비교·임계값 결정

학습 방식 2가지
A. 지도학습  : Normal+Abnormal로 학습. 임계값 = OOF에서 F1 최대
B. 정상만 학습: Normal만으로 학습, Abnormal은 평가에만 사용. 임계값 = Normal OOF 점수의 99% 분위(라벨 미사용)
    → Abnormal을 학습에 안 쓰므로 Abnormal 전체(train+test)로도 탐지율을 추가 보고

특징 세트 (뒤로 갈수록 누적)
F1 진동 → F2 +전류 크기 → F3 +채널 관계 → F4 +H1 잔차(정상에서 진동RMS~전류RMS 회귀, fold 내부 fit)
→ F5 +전류 시간구조(자기상관·평균교차율·차분std). F5는 수집조건 차이 혼입 가능성 → 최종 선정 제외, 참고용

실행:  python 08_01_model_comparison.py
출력:  results/model_comparison.csv
    results/test_predictions_A.csv, results/test_predictions_B.csv   (선정 모델의 test 예측, 공식 key 포함)
    results/oof_predictions_A.csv,  results/oof_predictions_B.csv    (오류분석용 train OOF 예측)
    results/model_config.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SIG = {"AI0_Vibration": "v0", "AI1_Vibration": "v1", "AI2_Current": "cur"}
FILES = {"normal": "press_data_normal.csv", "abnormal": "outlier_data.csv"}
WINDOW, STRIDE, GAP_SEC = 10, 1, 0.15
TEST_EVERY, N_FOLDS, SEED = 5, 5, 42
NORMAL_FPR_TARGET = 0.01
EPS = 1e-12
EXCLUDE_FROM_SELECTION = ["F5_+전류시간구조"]


# =============================================================== 데이터
def preprocess_like_04(path, source):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    df["source_row"] = np.arange(len(df))
    df = df.dropna(subset=["TimeStamp", *SIG, "Equipment_state"])
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
    df = df.loc[~df.duplicated(subset=["TimeStamp", *SIG, "Equipment_state"])].reset_index(drop=True)
    dt = df["TimeStamp"].diff().dt.total_seconds()
    seg_no = ((dt > GAP_SEC) | (dt <= 0)).cumsum()
    df["segment_id"] = ("N" if source == "normal" else "A") + seg_no.astype(str).str.zfill(4)
    df["pos_in_seg"] = df.groupby("segment_id", sort=False).cumcount()
    df["seg_len"] = df.groupby("segment_id", sort=False)["segment_id"].transform("size")
    df["source"] = source
    return df


def load():
    for base in [HERE, *HERE.parents, Path.cwd(), *Path.cwd().parents]:
        for p in (base / "data" / "processed" / "preprocessed_data.csv", base / "data" / "preprocessed_data.csv"):
            if p.exists():
                return pd.read_csv(p, parse_dates=["TimeStamp"]), str(p)
        if all((base / "data" / f).exists() for f in FILES.values()):
            d = base / "data"
            return pd.concat([preprocess_like_04(d / f, s) for s, f in FILES.items()], ignore_index=True), str(d)
    raise FileNotFoundError("data/(processed/)preprocessed_data.csv 또는 data/*.csv 없음")


# =============================================================== 윈도우 특징
def channel_features(x, p):
    f = {}
    mu = x.mean(1)
    xc = x - mu[:, None]
    m2, m3, m4 = (xc**2).mean(1), (xc**3).mean(1), (xc**4).mean(1)
    rms, absmean, absmax = np.sqrt((x**2).mean(1)), np.abs(x).mean(1), np.abs(x).max(1)
    f.update({f"{p}_mean": mu, f"{p}_std": np.sqrt(m2), f"{p}_rms": rms, f"{p}_absmean": absmean,
            f"{p}_absmax": absmax, f"{p}_max": x.max(1), f"{p}_min": x.min(1), f"{p}_ptp": x.max(1) - x.min(1),
            f"{p}_skew": m3 / (m2**1.5 + EPS), f"{p}_kurt": m4 / (m2**2 + EPS) - 3,
            f"{p}_crest": absmax / (rms + EPS), f"{p}_impulse": absmax / (absmean + EPS),
            f"{p}_diffstd": np.diff(x, axis=1).std(1),
            f"{p}_mcr": (np.diff(np.sign(xc), axis=1) != 0).mean(1)})
    den = (xc**2).sum(1) + EPS
    for k in (1, 2, 3):
        f[f"{p}_ac{k}"] = (xc[:, :-k] * xc[:, k:]).sum(1) / den
    return f


def corr_rows(a, b):
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (ac * bc).sum(1) / (np.sqrt((ac**2).sum(1) * (bc**2).sum(1)) + EPS)


def make_windows(data):
    out = []
    for (src, seg), g in data.sort_values(["source", "segment_id", "pos_in_seg"]).groupby(["source", "segment_id"], sort=False):
        if len(g) < WINDOW:
            continue
        w = sliding_window_view(g[list(SIG)].to_numpy(float), WINDOW, axis=0)[::STRIDE]
        end = np.arange(WINDOW - 1, len(g), STRIDE)
        v0, v1, cur = w[:, 0], w[:, 1], w[:, 2]
        feat = {}
        for arr, p in ((v0, "v0"), (v1, "v1"), (cur, "cur")):
            feat.update(channel_features(arr, p))
        feat.update(corr_v0_v1=corr_rows(v0, v1), corr_v0_cur=corr_rows(v0, cur), corr_v1_cur=corr_rows(v1, cur),
                    rms_diff_v0_v1=feat["v0_rms"] - feat["v1_rms"],
                    rms_ratio_v0_v1=feat["v0_rms"] / (feat["v1_rms"] + EPS))
        meta = pd.DataFrame({"source": src, "segment_id": seg,
                            "source_row": g["source_row"].to_numpy()[end], "pos_in_seg": end,
                            "seg_len": len(g), "t_end": g["TimeStamp"].to_numpy()[end],
                            "label": int(g["Equipment_state"].iloc[0]),
                            "split": "test" if int(seg[1:]) % TEST_EVERY == TEST_EVERY // 2 else "train"})
        out.append(pd.concat([meta, pd.DataFrame(feat)], axis=1))
    return pd.concat(out, ignore_index=True)


META = ["source", "segment_id", "source_row", "pos_in_seg", "seg_len", "t_end", "label", "split"]


def feature_sets(cols):
    temporal = ["cur_ac1", "cur_ac2", "cur_ac3", "cur_mcr", "cur_diffstd"]
    vib = [c for c in cols if c.startswith(("v0_", "v1_"))]
    cur_mag = [c for c in cols if c.startswith("cur_") and c not in temporal]
    rel = ["corr_v0_v1", "corr_v0_cur", "corr_v1_cur", "rms_diff_v0_v1", "rms_ratio_v0_v1"]
    base = vib + cur_mag + rel
    return {"F1_진동": (vib, False), "F2_+전류크기": (vib + cur_mag, False), "F3_+채널관계": (base, False),
            "F4_+H1잔차": (base, True), "F5_+전류시간구조": (base + temporal, True)}


# =============================================================== 모델 구성요소
class H1Residual(BaseEstimator, TransformerMixin):
    """정상(y==0)에서 진동 RMS ~ 전류 RMS 선형관계 학습 → 표준화 잔차 추가"""
    def fit(self, X, y=None):
        n = X if y is None else X[np.asarray(y) == 0]
        self.coef_ = {v: np.polyfit(n["cur_rms"], n[f"{v}_rms"], 1) for v in ("v0", "v1")}
        self.std_ = {v: (n[f"{v}_rms"] - np.polyval(self.coef_[v], n["cur_rms"])).std() for v in ("v0", "v1")}
        return self

    def transform(self, X):
        X = X.copy()
        for v in ("v0", "v1"):
            X[f"{v}_resid"] = (X[f"{v}_rms"] - np.polyval(self.coef_[v], X["cur_rms"])) / self.std_[v]
        return X


class RMSIQRRule(BaseEstimator, ClassifierMixin):
    """팀 베이스라인: 정상 RMS IQR 범위 이탈(3센서 OR). score>0 = 경보"""
    cols = ["v0_rms", "v1_rms", "cur_rms"]

    def fit(self, X, y):
        n = X.loc[np.asarray(y) == 0, self.cols]
        q1, q3 = n.quantile(0.25), n.quantile(0.75)
        self.iqr_, self.lo_, self.hi_ = q3 - q1, q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        return self

    def decision_function(self, X):
        x = X[self.cols]
        return np.maximum((x - self.hi_) / self.iqr_, (self.lo_ - x) / self.iqr_).max(axis=1).to_numpy()


class NormalOnly(BaseEstimator, ClassifierMixin):
    """정상만 학습하는 이상점수 모델 (iforest | mahalanobis)"""
    def __init__(self, kind="iforest"):
        self.kind = kind

    def fit(self, X, y):
        n = X[np.asarray(y) == 0]
        self.scaler_ = StandardScaler().fit(n)
        z = self.scaler_.transform(n)
        if self.kind == "iforest":
            self.m_ = IsolationForest(n_estimators=300, random_state=SEED).fit(z)
        else:
            self.m_ = LedoitWolf().fit(z)
        return self

    def decision_function(self, X):
        z = self.scaler_.transform(X)
        return -self.m_.score_samples(z) if self.kind == "iforest" else self.m_.mahalanobis(z)


MODELS = {
    "A": {"LogReg": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)),
        "RandomForest": RandomForestClassifier(n_estimators=300,
                                            min_samples_leaf=5,
                                            class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=SEED),
        "HistGB": HistGradientBoostingClassifier(max_iter=300,
                                                learning_rate=0.05,
                                                max_leaf_nodes=15,
                                                class_weight="balanced",
                                                random_state=SEED)
                                                },
    "B": {"IsolationForest": NormalOnly("iforest"), "Mahalanobis": NormalOnly("mahalanobis")},
}


def build(model, resid):
    return make_pipeline(H1Residual(), clone(model)) if resid else clone(model)


def score(est, X):
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)


def thr_best_f1(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + EPS)
    return float(t[np.nanargmax(f[:-1])])


def metrics(y, s, thr, seg, prefix):
    pred = (s >= thr).astype(int)
    d = pd.DataFrame({"y": y, "p": pred, "seg": seg}).groupby("seg").agg(y=("y", "first"), a=("p", "max"))
    return {f"{prefix}_PR_AUC": average_precision_score(y, s) if y.sum() else np.nan,
            f"{prefix}_F1": f1_score(y, pred, zero_division=0), f"{prefix}_Recall": recall_score(y, pred, zero_division=0),
            f"{prefix}_Precision": precision_score(y, pred, zero_division=0),
            f"{prefix}_FalseAlarm": pred[y == 0].mean() if (y == 0).any() else np.nan,
            f"{prefix}_AbnSegDetect": d.loc[d.y == 1, "a"].mean(), f"{prefix}_NormSegFalseAlarm": d.loc[d.y == 0, "a"].mean()}


# =============================================================== main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, src_path = load()
    W = make_windows(data)
    F = [c for c in W.columns if c not in META]
    tr, te = W[W.split == "train"].reset_index(drop=True), W[W.split == "test"].reset_index(drop=True)
    ytr, yte = tr.label.to_numpy(), te.label.to_numpy()
    folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(tr, ytr, tr.segment_id))
    abn_all = W[W.label == 1]

    runs = [("A", "RMS-IQR", "F0_RMS", RMSIQRRule(), RMSIQRRule.cols, False)]
    for fs, (cols, resid) in feature_sets(F).items():
        for fw, ms in MODELS.items():
            for name, m in ms.items():
                runs.append((fw, name, fs, m, cols, resid))

    rows, keep = [], {}
    for fw, name, fs, m, cols, resid in runs:
        oof = np.zeros(len(tr))
        for a, b in folds:
            oof[b] = score(build(m, resid).fit(tr.loc[a, cols], ytr[a]), tr.loc[b, cols])
        if name == "RMS-IQR":
            thr = 0.0
        elif fw == "A":
            thr = thr_best_f1(ytr, oof)
        else:
            thr = float(np.quantile(oof[ytr == 0], 1 - NORMAL_FPR_TARGET))
        est = build(m, resid).fit(tr[cols], ytr)
        s_te = score(est, te[cols])
        r = {"framework": fw, "model": name, "features": fs, "threshold": thr,
             **metrics(ytr, oof, thr, tr.segment_id, "CV"), **metrics(yte, s_te, thr, te.segment_id, "TEST")}
        if fw == "B":   # 정상만 학습 → Abnormal 전체 + test Normal로 추가 평가
            ev = pd.concat([te[te.label == 0], abn_all])
            r.update(metrics(ev.label.to_numpy(), score(est, ev[cols]), thr, ev.segment_id, "ALLABN"))
        rows.append(r)
        keep[(fw, name, fs)] = (oof, s_te, thr)
        print(f"{fw} {name:16s} {fs:18s} CV PR-AUC {r['CV_PR_AUC']:.3f} | TEST F1 {r['TEST_F1']:.3f} "
            f"FA {r['TEST_FalseAlarm']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "model_comparison.csv", index=False)

    # 선정: framework별 CV PR-AUC 최고 (F5·베이스라인 제외, test 미사용)
    chosen = {}
    for fw in ("A", "B"):
        c = res[(res.framework == fw) & (res.model != "RMS-IQR") & ~res.features.isin(EXCLUDE_FROM_SELECTION)]
        best = c.sort_values("CV_PR_AUC", ascending=False).iloc[0]
        chosen[fw] = {"model": best.model, "features": best.features, "threshold": float(best.threshold)}
        oof, s_te, thr = keep[(fw, best.model, best.features)]
        for df, s, name in ((te, s_te, "test"), (tr, oof, "oof")):
            p = df[META].copy()
            p["anomaly_score"], p["alarm"] = s, (s >= thr).astype(int)
            p.to_csv(OUT / f"{name}_predictions_{fw}.csv", index=False)

    cfg = {"data": src_path, "window": WINDOW, "stride": STRIDE, "test_rule": f"segment 번호 % {TEST_EVERY} == {TEST_EVERY // 2}",
        "n_windows": W.groupby(["split", "source"]).size().unstack().to_dict(), "n_features": len(F),
        "folds": N_FOLDS, "seed": SEED, "B_threshold": f"Normal OOF score {1 - NORMAL_FPR_TARGET:.0%} quantile",
        "selected": chosen}
    with open(OUT / "model_config.json", "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, ensure_ascii=False, indent=2, default=str)
    print("\n선정:", json.dumps(chosen, ensure_ascii=False))


if __name__ == "__main__":
    main()
