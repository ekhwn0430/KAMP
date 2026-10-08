"""
RMS 미검출 이상 분석

1) 팀 베이스라인(Rolling RMS 10샘플 + 정상 IQR, 3센서 OR)을 segment 내부 계산으로 재현
2) 윈도우 크기별(0.5/1/2/5초) 베이스라인 비교
3) 미검출 이상 vs 정상: 특징별 분리도(AUC)
4) 팀 가설 H1~H3을 순서대로 추가했을 때 미검출이 얼마나 설명되는지(누적)
5) 이상 구간별 요약, 이상 행별 판정 상태

실행:  python analyze_undetected.py      (04_04_04_Preprocessing.py와 같은 폴더)
출력:  results/undetected/*.csv

기준
- 베이스라인 IQR 임계값: 정상 전체 (팀 기존 방식과 동일하게 비교하기 위함)
- 가설 규칙 임계값·잔차 회귀: 정상 train만 사용, 오경보율은 정상 test에서 측정
- 가설 규칙: 정상 train 분포의 0.5% / 99.5% 분위수를 벗어나면 경보 (특징당 약 1%)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# preprocess.py는 04_Data_Processing 폴더에 있으므로 import 경로에 추가한다.
import sys
try: _here = Path(__file__).resolve().parent
except NameError: _here = Path.cwd()
_root = next(p for p in [_here, *_here.parents] if (p / "04_Data_Processing" / "preprocess.py").exists())
sys.path.insert(0, str(_root / "04_Data_Processing"))
from preprocess import load_and_clean, make_features, feature_cols

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "undetected"
RMS_COLS = ["v0_rms", "v1_rms", "cur_rms"]
Q_LO, Q_HI = 0.005, 0.995

HYPOTHESES = [
    ("H1_전류-진동_관계이탈", ["v0_resid", "v1_resid"]),
    ("H2_진동채널_관계", ["corr_v0_v1"]),
    ("H3a_전류_크기", ["cur_mean", "cur_std", "cur_min"]),
    ("H3b_전류_시간구조", ["cur_ac1", "cur_mcr"]),
]


def rms_iqr_flag(feats):
    n = feats[feats.label == 0]
    q1, q3 = n[RMS_COLS].quantile(0.25), n[RMS_COLS].quantile(0.75)
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return ((feats[RMS_COLS] > hi) | (feats[RMS_COLS] < lo)).any(axis=1)


def add_residuals(feats):
    """H1: 정상 train에서 진동 RMS ~ 전류 RMS 선형관계를 학습, 표준화 잔차 계산"""
    tr = feats[(feats.label == 0) & (feats.split == "train")]
    info = {}
    for v in ["v0", "v1"]:
        b = np.polyfit(tr.cur_rms, tr[f"{v}_rms"], 1)
        r = tr[f"{v}_rms"] - np.polyval(b, tr.cur_rms)
        feats[f"{v}_resid"] = (feats[f"{v}_rms"] - np.polyval(b, feats.cur_rms)) / r.std()
        info[v] = {"slope": b[0], "intercept": b[1], "R2": 1 - r.var() / tr[f"{v}_rms"].var()}
    return feats, info


def rule_flag(feats, cols):
    tr = feats[(feats.label == 0) & (feats.split == "train")]
    m = pd.Series(False, index=feats.index)
    for c in cols:
        m |= (feats[c] < tr[c].quantile(Q_LO)) | (feats[c] > tr[c].quantile(Q_HI))
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, _ = load_and_clean()

    # ---------------- 2) 윈도우 크기별 베이스라인
    rows = []
    n_seg = data.groupby("label")["segment_id"].nunique()
    for W in (5, 10, 20, 50):          # 0.5 / 1 / 2 / 5초
        g = make_features(data, W)
        g["flag"] = rms_iqr_flag(g)
        a = g.label == 1
        seg = g.groupby("segment_id").agg(label=("label", "first"), alarm=("flag", "max"))
        rows.append({"window": W, "seconds": W / 10,
                    "abnormal_rows_evaluable": int(a.sum()),
                    "abnormal_rows_not_evaluable": int((data.label == 1).sum() - a.sum()),
                    "abnormal_segments_evaluable": f"{(seg.label == 1).sum()}/{n_seg[1]}",
                    "normal_segments_evaluable": f"{(seg.label == 0).sum()}/{n_seg[0]}",
                    "row_detect_rate": g.loc[a, "flag"].mean(),
                    "abnormal_segment_detect_rate": seg.loc[seg.label == 1, "alarm"].mean(),
                    "normal_row_false_alarm_rate": g.loc[~a, "flag"].mean(),
                    "normal_segment_false_alarm_rate": seg.loc[seg.label == 0, "alarm"].mean()})
    by_window = pd.DataFrame(rows)
    by_window.to_csv(OUT / "baseline_by_window.csv", index=False)

    # ---------------- 1) W=10 베이스라인
    feats = make_features(data, 10)
    feats["rms_flag"] = rms_iqr_flag(feats)
    feats, resid_info = add_residuals(feats)
    A = feats.label == 1
    N = feats.label == 0
    Nt = N & (feats.split == "test")
    U = A & ~feats.rms_flag

    # ---------------- 3) 미검출 vs 정상 분리도
    cand = feature_cols(feats)
    cand = [c for c in cand if c != "rms_flag"]
    y = np.r_[np.zeros(N.sum()), np.ones(U.sum())]
    auc_rows = []
    for c in cand:
        auc = roc_auc_score(y, np.r_[feats.loc[N, c], feats.loc[U, c]])
        auc_rows.append({"feature": c, "auc": max(auc, 1 - auc),
                        "direction": "higher_in_undetected" if auc >= 0.5 else "lower_in_undetected",
                        "normal_median": feats.loc[N, c].median(),
                        "undetected_median": feats.loc[U, c].median(),
                        "detected_median": feats.loc[A & feats.rms_flag, c].median()})
    auc_df = pd.DataFrame(auc_rows).sort_values("auc", ascending=False)
    auc_df.to_csv(OUT / "feature_auc_undetected_vs_normal.csv", index=False)

    # ---------------- 4) 가설 누적 설명력
    cum = feats.rms_flag.copy()
    feats["explained_by"] = np.where(feats.rms_flag, "RMS_baseline", "unexplained")
    cov = [{"step": "RMS_baseline", "abnormal_detected": int(cum[A].sum()),
            "added": int(cum[A].sum()), "normal_test_false_alarm": cum[Nt].mean()}]
    for name, cols in HYPOTHESES:
        new = rule_flag(feats, cols) & ~cum
        feats.loc[new & A, "explained_by"] = name
        cum |= new
        cov.append({"step": f"+{name}", "abnormal_detected": int(cum[A].sum()),
                    "added": int((new & A).sum()), "normal_test_false_alarm": cum[Nt].mean()})
    cov_df = pd.DataFrame(cov)
    cov_df.to_csv(OUT / "hypothesis_coverage.csv", index=False)

    # ---------------- 5) 이상 행별 상태, 구간별 요약
    ab = data[data.label == 1][["TimeStamp", "segment_id", "pos_in_seg", "v0", "v1", "cur"]].copy()
    st = feats.loc[A, ["segment_id", "pos_in_seg", "explained_by"]]
    ab = ab.merge(st, on=["segment_id", "pos_in_seg"], how="left")
    ab["explained_by"] = ab["explained_by"].fillna("not_evaluable")
    ab.to_csv(OUT / "abnormal_rows_status.csv", index=False)

    seg = (ab.groupby(["segment_id", "explained_by"]).size().unstack(fill_value=0))
    med = feats[A].groupby("segment_id")[["v0_rms", "v1_rms", "cur_rms", "v1_resid",
                                        "corr_v0_v1", "cur_mean", "cur_ac1"]].median()
    seg = seg.join(med).round(3)
    seg.to_csv(OUT / "segment_summary.csv")

    # ---------------- 출력
    pd.set_option("display.width", 200)
    print("[윈도우 크기별 베이스라인]\n", by_window.round(4).to_string(index=False))
    print("\n[H1 잔차 회귀 (정상 train)]", {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in resid_info.items()})
    print("\n[미검출 vs 정상 분리도 상위 10]\n", auc_df.head(10).round(3).to_string(index=False))
    print("\n[가설 누적 설명력] (판정 가능 이상 %d행)\n" % A.sum(), cov_df.round(4).to_string(index=False))
    print("\n[이상 600행 판정 상태]\n", ab.explained_by.value_counts().to_string())
    print("\n[이상 구간별 요약]\n", seg.to_string())


if __name__ == "__main__":
    main()
