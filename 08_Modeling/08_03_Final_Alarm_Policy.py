"""
08_03 Final Model / Alarm Policy

08_01에서 고른 두 모델을 현장 경보 규칙으로 묶고, 최종 결과 파일을 만든다.
  A = 지도학습 RandomForest (F4: 진동 + 전류 크기 + 채널 관계 + H1 잔차)
  B = 정상만 학습한 Mahalanobis (F4)

하는 일
  1. 확률보정  : A의 점수를 train OOF 예측으로 isotonic 보정 → 이상 확률(위험도)로 표시
                 B의 점수는 정상 OOF 점수 분포 기준 백분위(정상 대비 이탈 정도)로 변환
  2. 경보 등급  : 정상(0) / 주의(1: A·B 중 하나만 이상) / 경보(2: A·B 모두 이상)
  3. 연속경보  : 같은 segment 안에서 k개 윈도우 연속일 때만 확정 → 오경보 감소
                 k는 train OOF에서만 결정 (test 미사용)
  4. 탐지 지연  : 관측 구간 시작 → 첫 확정 경보까지 시간
  5. 중요도    : segment 단위 5-fold 안에서 feature 묶음 / 개별 feature permutation importance (PR-AUC 감소량)
  6. 최종 test 예측 파일 저장

원칙
  - 모든 선택(보정, k, 경보 규칙)은 train OOF로만 결정하고 test는 평가에만 사용
  - 예측이 없는 행(segment 첫 9행, 10행 미만 segment)은 not_evaluable로 따로 표기, 정상으로 세지 않음

실행:  python 08_03_Final_Alarm_Policy.py      (08_01 실행 후)
출력:  results/final/
  policy_comparison.csv        경보 규칙별 window·segment 성능 (OOF / TEST)
  ablation_table.csv           08_01 특징 계단 비교 + F2·Confusion Matrix
  calibration.csv              A 보정 전후 Brier score
  importance_family.csv        feature 묶음별 permutation importance
  importance_feature.csv       개별 feature permutation importance
  detection_delay.csv          이상 segment별 첫 경보 시간
  test_predictions_final.csv   최종 test 예측 (공식 key + 위험도 + 경보 등급)
  rows_final_status.csv        전체 20,599행 기준 평가 상태 (not_evaluable 포함)
  final_config.json
"""
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "final"
K_CANDIDATES = [1, 2, 3, 5]
PERM_REPEATS = 3
WIN_SEC = 0.1

# 08_01 함수·설정 재사용 (파일명이 숫자로 시작해 importlib로 불러옴)
_spec = importlib.util.spec_from_file_location("m0801", HERE / "08_01_model_comparison.py")
m0801 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m0801)

FAMILIES = {
    "진동0 크기": lambda c: c.startswith("v0_") and c.split("_", 1)[1] in
    {"mean", "std", "rms", "absmean", "absmax", "max", "min", "ptp"},
    "진동1 크기": lambda c: c.startswith("v1_") and c.split("_", 1)[1] in
    {"mean", "std", "rms", "absmean", "absmax", "max", "min", "ptp"},
    "진동 파형형태": lambda c: c.startswith(("v0_", "v1_")) and c.split("_", 1)[1] in
    {"skew", "kurt", "crest", "impulse", "diffstd", "mcr", "ac1", "ac2", "ac3"},
    "전류 크기": lambda c: c.startswith("cur_") and c.split("_", 1)[1] in
    {"mean", "std", "rms", "absmean", "absmax", "max", "min", "ptp"},
    "전류 파형형태": lambda c: c.startswith("cur_") and c.split("_", 1)[1] in {"skew", "kurt", "crest", "impulse"},
    "채널 관계": lambda c: c.startswith(("corr_", "rms_diff", "rms_ratio")),
    "H1 잔차(전류 대비 진동)": lambda c: c.endswith("_resid"),
}


# =============================================================== 공통
def load_preds(code):
    cols = ["source", "segment_id", "source_row", "pos_in_seg", "seg_len", "label", "split", "anomaly_score", "alarm"]
    oof = pd.read_csv(RES / f"oof_predictions_{code}.csv", usecols=cols)
    te = pd.read_csv(RES / f"test_predictions_{code}.csv", usecols=cols)
    oof["eval_set"], te["eval_set"] = "OOF", "TEST"
    return pd.concat([oof, te], ignore_index=True)


def f_beta(p, r, b):
    return (1 + b * b) * p * r / (b * b * p + r) if (p + r) > 0 else 0.0


def window_metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Precision=p, Recall=r, F1=f_beta(p, r, 1), F2=f_beta(p, r, 2),
                FalseAlarmRate=fp / (fp + tn) if fp + tn else np.nan)


def persistent(df, flag_col, k):
    """segment 안에서 flag가 k개 연속일 때부터 확정 경보 (k=1이면 원래 flag)"""
    if k == 1:
        return df[flag_col].astype(int)
    run = df.groupby("segment_id", sort=False)[flag_col].transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumsum())
    return ((run >= k) & df[flag_col].astype(bool)).astype(int)


def segment_metrics(df, col):
    seg = df.groupby("segment_id").agg(label=("label", "first"), alarm=(col, "max"))
    a, n = seg[seg.label == 1], seg[seg.label == 0]
    first = df[df[col] == 1].groupby("segment_id").pos_in_seg.min()
    delay = (first.reindex(a.index) * WIN_SEC).dropna()     # 관측 구간 시작(0초) 기준
    return dict(AbnSegDetect=f"{int(a.alarm.sum())}/{len(a)}", AbnSegDetectRate=a.alarm.mean(),
                NormSegFalseAlarm=f"{int(n.alarm.sum())}/{len(n)}", NormSegFalseAlarmRate=n.alarm.mean(),
                DelaySec_median=delay.median() if len(delay) else np.nan,
                DelaySec_max=delay.max() if len(delay) else np.nan)


# =============================================================== 1. 보정
def calibrate(df):
    oof = df[df.eval_set == "OOF"]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof.A_score, oof.label)
    df["A_prob"] = iso.predict(df.A_score)
    nB = np.sort(oof.loc[oof.label == 0, "B_score"].to_numpy())
    df["B_normal_pct"] = np.searchsorted(nB, df.B_score, side="right") / len(nB) * 100
    te = df[df.eval_set == "TEST"]
    cal = pd.DataFrame([
        dict(model="A RandomForest", score="raw predict_proba", TEST_Brier=brier_score_loss(te.label, te.A_score.clip(0, 1))),
        dict(model="A RandomForest", score="isotonic (OOF fit)", TEST_Brier=brier_score_loss(te.label, te.A_prob)),
    ])
    return df, cal


# =============================================================== 2~4. 경보 규칙
def policies(df):
    df["level"] = df.A_alarm + df.B_alarm                       # 0 정상 / 1 주의 / 2 경보
    df["flag_A"] = df.A_alarm
    df["flag_B"] = df.B_alarm
    df["flag_AND"] = (df.level == 2).astype(int)
    df["flag_OR"] = (df.level >= 1).astype(int)
    rows = []
    for name, base in [("A만", "flag_A"), ("B만", "flag_B"), ("A·B 모두(경보)", "flag_AND"), ("A·B 중 하나(주의)", "flag_OR")]:
        for k in K_CANDIDATES:
            col = f"{base}_k{k}"
            df[col] = persistent(df, base, k)
            for ev in ("OOF", "TEST"):
                d = df[df.eval_set == ev]
                rows.append(dict(eval_set=ev, rule=name, k=k, **window_metrics(d.label.to_numpy(), d[col].to_numpy()),
                                 **segment_metrics(d, col)))
    return df, pd.DataFrame(rows)


def choose(pol):
    """train OOF만 보고 결정: 이상 segment 100% 탐지 유지 → 정상 segment 오경보 최소 → 지연 최소 → k 최소"""
    o = pol[(pol.eval_set == "OOF") & (pol.rule == "A·B 모두(경보)")].copy()
    o = o[o.AbnSegDetectRate == o.AbnSegDetectRate.max()]
    o = o.sort_values(["NormSegFalseAlarmRate", "DelaySec_median", "k"])
    return int(o.iloc[0].k)


# =============================================================== 5. 중요도
def permutation_importance_cv(cfg):
    data, _ = m0801.load()
    W = m0801.make_windows(data)
    F = [c for c in W.columns if c not in m0801.META]
    tr = W[W.split == "train"].reset_index(drop=True)
    y = tr.label.to_numpy()
    cols, resid = m0801.feature_sets(F)[cfg["features"]]
    model = m0801.MODELS["A"][cfg["model"]]
    folds = StratifiedGroupKFold(m0801.N_FOLDS, shuffle=True, random_state=m0801.SEED).split(tr, y, tr.segment_id)
    rng = np.random.default_rng(m0801.SEED)
    fam_rows, feat_rows = [], []
    for fi, (a, b) in enumerate(folds):
        h1 = m0801.H1Residual().fit(tr.loc[a, cols], y[a])
        Xa, Xb = h1.transform(tr.loc[a, cols]), h1.transform(tr.loc[b, cols])
        est = clone(model).fit(Xa, y[a])
        base = average_precision_score(y[b], est.predict_proba(Xb)[:, 1])
        groups = {fam: [c for c in Xb.columns if rule(c)] for fam, rule in FAMILIES.items()}
        groups.update({c: [c] for c in Xb.columns})
        for g, gc in groups.items():
            drops = []
            for _ in range(PERM_REPEATS):
                Xp = Xb.copy()
                idx = rng.permutation(len(Xp))
                Xp[gc] = Xp[gc].to_numpy()[idx]
                drops.append(base - average_precision_score(y[b], est.predict_proba(Xp)[:, 1]))
            row = dict(fold=fi, group=g, n_features=len(gc), base_PR_AUC=base, PR_AUC_drop=float(np.mean(drops)))
            (fam_rows if g in FAMILIES else feat_rows).append(row)
    agg = lambda r: (pd.DataFrame(r).groupby(["group", "n_features"])
                     .agg(PR_AUC_drop_mean=("PR_AUC_drop", "mean"), PR_AUC_drop_std=("PR_AUC_drop", "std"))
                     .reset_index().sort_values("PR_AUC_drop_mean", ascending=False))
    return agg(fam_rows), agg(feat_rows)


# =============================================================== 6. ablation 표
def ablation():
    r = pd.read_csv(RES / "model_comparison.csv")
    cfg = json.load(open(RES / "model_config.json", encoding="utf-8"))
    n_abn, n_norm = cfg["n_windows"]["abnormal"]["test"], cfg["n_windows"]["normal"]["test"]
    t = r.copy()
    t["TEST_TP"] = (t.TEST_Recall * n_abn).round().astype(int)
    t["TEST_FN"] = n_abn - t.TEST_TP
    t["TEST_FP"] = (t.TEST_FalseAlarm * n_norm).round().astype(int)
    t["TEST_TN"] = n_norm - t.TEST_FP
    t["TEST_F2"] = [f_beta(p, q, 2) for p, q in zip(t.TEST_Precision, t.TEST_Recall)]
    t["CV_F2"] = [f_beta(p, q, 2) for p, q in zip(t.CV_Precision, t.CV_Recall)]
    keep = ["framework", "model", "features", "CV_PR_AUC", "CV_F1", "CV_F2", "TEST_PR_AUC", "TEST_F1", "TEST_F2",
            "TEST_Recall", "TEST_Precision", "TEST_FalseAlarm", "TEST_TP", "TEST_FN", "TEST_FP", "TEST_TN",
            "TEST_AbnSegDetect", "TEST_NormSegFalseAlarm"]
    return t[keep]


# =============================================================== main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(RES / "model_config.json", encoding="utf-8"))
    A, B = load_preds("A"), load_preds("B")
    key = ["source", "segment_id", "source_row"]
    df = A.rename(columns={"anomaly_score": "A_score", "alarm": "A_alarm"}).merge(
        B[key + ["anomaly_score", "alarm"]].rename(columns={"anomaly_score": "B_score", "alarm": "B_alarm"}),
        on=key, how="inner", validate="1:1")
    assert len(df) == len(A) == len(B)
    df = df.sort_values(["eval_set", "segment_id", "pos_in_seg"]).reset_index(drop=True)

    df, cal = calibrate(df)
    df, pol = policies(df)
    k = choose(pol)
    df["final_alarm"] = df[f"flag_AND_k{k}"]
    df["alarm_level"] = df.level.map({0: "정상", 1: "주의", 2: "경보"})

    # 탐지 지연 (이상 segment별)
    first = (df[df.final_alarm == 1].groupby("segment_id").pos_in_seg.min() * WIN_SEC).rename("first_alarm_sec")
    first_caution = (df[df.flag_OR == 1].groupby("segment_id").pos_in_seg.min() * WIN_SEC).rename("first_caution_sec")
    delay = (df[df.label == 1].groupby(["eval_set", "segment_id"]).agg(seg_len=("seg_len", "first"),
                                                                        n_windows=("pos_in_seg", "size"))
             .reset_index().merge(first_caution, on="segment_id", how="left").merge(first, on="segment_id", how="left"))
    delay["earliest_possible_sec"] = 0.9

    # 중요도
    fam, feat = permutation_importance_cv(cfg["selected"]["A"])

    # 최종 test 예측
    te = df[df.eval_set == "TEST"]
    final_cols = ["source", "segment_id", "source_row", "pos_in_seg", "label", "A_prob", "B_normal_pct",
                  "A_alarm", "B_alarm", "alarm_level", "final_alarm"]
    te[final_cols].to_csv(OUT / "test_predictions_final.csv", index=False, encoding="utf-8-sig")

    # 전체 행 기준 평가 상태 (not_evaluable 구분)
    data, _ = m0801.load()
    rows = data[["source", "segment_id", "source_row", "pos_in_seg", "seg_len", "Equipment_state"]].rename(
        columns={"Equipment_state": "label"})
    rows = rows.merge(df[key + ["eval_set", "A_prob", "B_normal_pct", "alarm_level", "final_alarm"]], on=key, how="left")
    rows["eval_status"] = np.where(rows.eval_set.isna(), "not_evaluable", "evaluated")
    rows.to_csv(OUT / "rows_final_status.csv", index=False, encoding="utf-8-sig")
    ns = rows.groupby(["source", "eval_status"]).size().unstack(fill_value=0)

    pol.to_csv(OUT / "policy_comparison.csv", index=False, encoding="utf-8-sig")
    ablation().to_csv(OUT / "ablation_table.csv", index=False, encoding="utf-8-sig")
    cal.to_csv(OUT / "calibration.csv", index=False, encoding="utf-8-sig")
    fam.to_csv(OUT / "importance_family.csv", index=False, encoding="utf-8-sig")
    feat.to_csv(OUT / "importance_feature.csv", index=False, encoding="utf-8-sig")
    delay.to_csv(OUT / "detection_delay.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "final_config.json", "w", encoding="utf-8") as fp:
        json.dump({"A": cfg["selected"]["A"], "B": cfg["selected"]["B"],
                   "final_rule": "A·B 모두 이상이면 즉시 경보" if k == 1 else f"A·B 모두 이상이 {k}개 윈도우 연속이면 경보", "k": k,
                   "k_selected_on": "train OOF only", "calibration": "isotonic on A OOF scores",
                   "rows_eval_status": ns.to_dict()}, fp, ensure_ascii=False, indent=2, default=str)

    # ---- 출력
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    show = ["eval_set", "rule", "k", "Recall", "Precision", "F1", "F2", "FalseAlarmRate", "FP", "FN",
            "AbnSegDetect", "NormSegFalseAlarm", "DelaySec_median", "DelaySec_max"]
    print("[경보 규칙 비교]\n", pol[show].round(4).to_string(index=False))
    print(f"\n선택된 k = {k} (train OOF 기준)")
    print("\n[보정]\n", cal.round(5).to_string(index=False))
    print("\n[feature 묶음 중요도]\n", fam.round(4).to_string(index=False))
    print("\n[개별 feature 중요도 상위 15]\n", feat.head(15).round(4).to_string(index=False))
    print("\n[탐지 지연]\n", delay.to_string(index=False))
    print("\n[전체 행 평가 상태]\n", ns.to_string())


if __name__ == "__main__":
    main()
