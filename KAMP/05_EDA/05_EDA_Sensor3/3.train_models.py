"""
모델 비교 (베이스라인 포함)

검증 설계
- test: preprocess.py에서 고정한 구간 홀드아웃 (시간순 5개 구간마다 1개). 최종 평가에만 사용
- train 내부: StratifiedGroupKFold 5-fold (그룹 = segment_id) -> out-of-fold(OOF) 예측
- 임계값: OOF 예측에서 F1 최대가 되는 값으로 정한 뒤 test에 그대로 적용
- H1 잔차 회귀, 스케일링, IQR 범위는 모두 fold의 train(정상)에서만 fit

비교 축
- 모델: RMS-IQR 규칙(팀 베이스라인), 로지스틱회귀, 랜덤포레스트, HistGradientBoosting(LightGBM 방식), IsolationForest(정상만 학습)
- 특징 세트: F1 진동 -> F2 +전류 크기 -> F3 +채널 관계 -> F4 +H1 잔차 -> F5 +전류 시간구조
  F5는 수집조건 차이(날짜) 혼입 가능성이 있어 최종 모델 선정에서 제외하고 참고용으로만 보고

실행:  python train_models.py
출력:  results/models/cv_results.csv, test_results.csv, test_predictions.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import load_and_clean, make_features, feature_cols

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "models"
SEED = 42
WINDOW = 10
N_FOLDS = 5
EXCLUDE_FROM_SELECTION = ["F5_+전류시간구조"]


# ---------------------------------------------------------------- 특징 세트
def feature_sets(cols):
    cur_temporal = ["cur_ac1", "cur_ac2", "cur_ac3", "cur_mcr", "cur_diffstd"]
    vib = [c for c in cols if c.startswith(("v0_", "v1_"))]
    cur_mag = [c for c in cols if c.startswith("cur_") and c not in cur_temporal]
    rel = ["corr_v0_v1", "corr_v0_cur", "corr_v1_cur", "rms_diff_v0_v1", "rms_ratio_v0_v1"]
    base = vib + cur_mag + rel
    return {
        "F1_진동": (vib, False),
        "F2_+전류크기": (vib + cur_mag, False),
        "F3_+채널관계": (base, False),
        "F4_+H1잔차": (base, True),
        "F5_+전류시간구조": (base + cur_temporal, True),
    }


# ---------------------------------------------------------------- 구성요소
class H1Residual(BaseEstimator, TransformerMixin):
    """정상 데이터에서 진동 RMS ~ 전류 RMS 선형관계를 학습하고 표준화 잔차를 특징으로 추가"""
    def fit(self, X, y):
        n = X[np.asarray(y) == 0]
        self.coef_, self.std_ = {}, {}
        for v in ("v0", "v1"):
            self.coef_[v] = np.polyfit(n["cur_rms"], n[f"{v}_rms"], 1)
            self.std_[v] = (n[f"{v}_rms"] - np.polyval(self.coef_[v], n["cur_rms"])).std()
        return self

    def transform(self, X):
        X = X.copy()
        for v in ("v0", "v1"):
            X[f"{v}_resid"] = (X[f"{v}_rms"] - np.polyval(self.coef_[v], X["cur_rms"])) / self.std_[v]
        return X


class RMSIQRRule(BaseEstimator, ClassifierMixin):
    """팀 베이스라인: 정상 RMS의 IQR 범위를 하나라도 벗어나면 이상. score>0 이면 경보"""
    cols = ["v0_rms", "v1_rms", "cur_rms"]

    def fit(self, X, y):
        n = X.loc[np.asarray(y) == 0, self.cols]
        q1, q3 = n.quantile(0.25), n.quantile(0.75)
        self.iqr_ = q3 - q1
        self.lo_, self.hi_ = q1 - 1.5 * self.iqr_, q3 + 1.5 * self.iqr_
        return self

    def decision_function(self, X):
        x = X[self.cols]
        return np.maximum((x - self.hi_) / self.iqr_, (self.lo_ - x) / self.iqr_).max(axis=1).to_numpy()


class NormalOnlyIForest(BaseEstimator, ClassifierMixin):
    """정상 데이터만으로 학습하는 비지도 이상탐지 (라벨·날짜 차이에 덜 의존)"""
    def __init__(self, n_estimators=300, random_state=SEED):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state)
        self.model_.fit(X[np.asarray(y) == 0])
        return self

    def decision_function(self, X):
        return -self.model_.score_samples(X)


MODELS = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)),
    "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=SEED),
    "HistGB": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             class_weight="balanced", random_state=SEED),
    "IsolationForest": NormalOnlyIForest(),
}


def build(model, use_resid):
    return make_pipeline(H1Residual(), clone(model)) if use_resid else clone(model)


def score(est, X):
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = np.nanargmax(f1[:-1])
    return t[i]


def evaluate(y, s, thr, seg, prefix):
    pred = (s >= thr).astype(int)
    df = pd.DataFrame({"y": y, "pred": pred, "seg": seg})
    segs = df.groupby("seg").agg(y=("y", "first"), alarm=("pred", "max"))
    return {
        f"{prefix}_PR_AUC": average_precision_score(y, s),
        f"{prefix}_F1": f1_score(y, pred),
        f"{prefix}_Recall": recall_score(y, pred),
        f"{prefix}_Precision": precision_score(y, pred, zero_division=0),
        f"{prefix}_FalseAlarmRate": pred[y == 0].mean(),
        f"{prefix}_AbnSegDetect": segs.loc[segs.y == 1, "alarm"].mean(),
        f"{prefix}_NormSegFalseAlarm": segs.loc[segs.y == 0, "alarm"].mean(),
    }


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, _ = load_and_clean()
    f = make_features(data, WINDOW)
    tr, te = f[f.split == "train"].reset_index(drop=True), f[f.split == "test"].reset_index(drop=True)
    ytr, yte = tr.label.to_numpy(), te.label.to_numpy()
    folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(tr, ytr, tr.segment_id))

    runs = [("RMS-IQR", "F0_RMS", RMSIQRRule(), RMSIQRRule.cols, False)]
    for fs_name, (cols, resid) in feature_sets(feature_cols(f)).items():
        for m_name, m in MODELS.items():
            runs.append((m_name, fs_name, m, cols, resid))

    rows, test_scores = [], {}
    for m_name, fs_name, m, cols, resid in runs:
        oof = np.zeros(len(tr))
        for itr, iva in folds:
            est = build(m, resid).fit(tr.loc[itr, cols], ytr[itr])
            oof[iva] = score(est, tr.loc[iva, cols])
        thr = 0.0 if m_name == "RMS-IQR" else best_f1_threshold(ytr, oof)
        est = build(m, resid).fit(tr[cols], ytr)
        s_te = score(est, te[cols])
        test_scores[(m_name, fs_name)] = (s_te, thr)
        rows.append({"model": m_name, "features": fs_name, "threshold": thr,
                     **evaluate(ytr, oof, thr, tr.segment_id, "CV"),
                     **evaluate(yte, s_te, thr, te.segment_id, "TEST")})
        print(f"{m_name:16s} {fs_name:16s} CV PR-AUC {rows[-1]['CV_PR_AUC']:.3f}  TEST F1 {rows[-1]['TEST_F1']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "model_results.csv", index=False)

    # 최종 후보: 전류 시간구조(F5) 제외, CV PR-AUC 최고 (test는 선정에 사용하지 않음)
    cand = res[~res.features.isin(EXCLUDE_FROM_SELECTION) & (res.model != "RMS-IQR")]
    best = cand.sort_values("CV_PR_AUC", ascending=False).iloc[0]
    s_te, thr = test_scores[(best.model, best.features)]
    pred = te[["window_id", "segment_id", "t_start", "t_end", "label"]].copy()
    pred["anomaly_score"], pred["alarm"] = s_te, (s_te >= thr).astype(int)
    pred.to_csv(OUT / "test_predictions.csv", index=False)
    print(f"\n최종 후보: {best.model} / {best.features}  (threshold {thr:.4f})")


if __name__ == "__main__":
    main()
