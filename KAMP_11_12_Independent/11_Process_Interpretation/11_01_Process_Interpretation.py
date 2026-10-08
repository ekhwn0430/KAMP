# 11. 공정해석: 커밋 b196032의 08 산출물을 읽기만 하여 조건별 FP/FN·영향요인·물리해석 증거표를 만든다.
# 모델을 재학습하지 않고 threshold, 예측값, 최종 경보를 변경하지 않는다.
# 원인 메커니즘은 검증된 사실이 아니라 점검 가설로 명시하며 08/07 입력 경로에 쓰지 않는다.
# 실행: python 11_Process_Interpretation/11_01_Process_Interpretation.py --root <KAMP_repo>

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

COMMIT = "b1960321fc598ebb091f1bf4504918c9f0d55f64"
MODEL_BLOBS = {
    "08_01_model_comparison.py": "e79f6b65e61a0253321dc032924e059b151fa154",
    "08_02_Model_Error_Analysis.py": "e3e5152690a9ba7576a85a3bd66101bc2810fce1",
    "08_03_Final_Alarm_Policy.py": "f9c813356b7116b116aa2fa5c7e3b47c3ca4fee3",
}
KEY = ["source", "segment_id", "source_row"]


def load_csv(path: Path, required=(), optional=False):
    if not path.exists():
        if optional:
            return None
        raise FileNotFoundError(f"필수 입력 파일 누락: {path}\n레포 최상단에서 python run_all.py를 먼저 실행하세요.")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: 필수 컬럼 누락 {sorted(missing)}")
    return df


def git_blob(path):
    # Windows(core.autocrlf=true)에서 CRLF로 체크아웃돼도 Git에 저장된 LF 기준으로 비교한다.
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify_08(root: Path):
    details = {}
    for name, expected in MODEL_BLOBS.items():
        path = root / "08_Modeling" / name
        if not path.exists():
            raise FileNotFoundError(f"08 원본 확인 실패: {path}")
        actual = git_blob(path)
        if actual != expected:
            raise RuntimeError(f"08 파일이 기준 커밋과 다름: {name}\nexpected={expected}\nactual={actual}\n원본 모델을 수정하지 말고 기준 브랜치/커밋을 확인하세요.")
        details[name] = actual
    return details


def assert_keys(df, table):
    if df.duplicated(KEY).any():
        raise ValueError(f"{table}: source/segment_id/source_row 중복")
    if df[KEY].isna().any().any():
        raise ValueError(f"{table}: 공식 키에 결측")


def indicator(s):
    txt = s.astype(str).str.strip().str.lower()
    return txt.isin(["true", "1", "1.0", "yes", "y"]) | pd.to_numeric(s, errors="coerce").eq(1)


def pct(a, b):
    return float(a / b) if b else None


def load_predictions(root):
    base = root / "08_Modeling" / "results"
    test = load_csv(base / "final/test_predictions_final.csv", [*KEY, "pos_in_seg", "label", "A_alarm", "B_alarm", "final_alarm", "alarm_level"])
    assert_keys(test, "final/test_predictions_final.csv")
    test = test.copy()
    test["eval_set"] = "TEST"

    # OOF는 최종 선택 정책(k=1, A&B)에 대한 사후 오류 진단용이다.
    # 확률보정된 TEST A_prob와 OOF raw anomaly_score는 비교하지 않는다.
    aa = load_csv(base / "oof_predictions_A.csv", [*KEY, "label", "pos_in_seg", "alarm"], optional=True)
    bb = load_csv(base / "oof_predictions_B.csv", [*KEY, "alarm"], optional=True)
    if aa is not None and bb is not None:
        assert_keys(aa, "OOF A")
        assert_keys(bb, "OOF B")
        z = aa[[*KEY, "label", "pos_in_seg", "alarm"]].rename(columns={"alarm": "A_alarm"}).merge(
            bb[[*KEY, "alarm"]].rename(columns={"alarm": "B_alarm"}), on=KEY, how="inner", validate="one_to_one")
        if len(z) != len(aa) or len(z) != len(bb):
            raise ValueError("OOF 모델 A/B window key 불일치")
        z["final_alarm"] = (indicator(z["A_alarm"]) & indicator(z["B_alarm"])).astype(int)
        z["alarm_level"] = np.select(
            [indicator(z["A_alarm"]) & indicator(z["B_alarm"]), indicator(z["A_alarm"]) | indicator(z["B_alarm"])],
            ["경보", "주의"], default="정상")
        z["eval_set"] = "OOF"
        return pd.concat([z, test], ignore_index=True, sort=False)
    return test


def confusions(df):
    y = pd.to_numeric(df["label"]).astype(int)
    p = indicator(df["final_alarm"]).astype(int)
    return np.select([(y == 1) & (p == 1), (y == 1) & (p == 0), (y == 0) & (p == 1)], ["TP", "FN", "FP"], default="TN")


def metrics(df):
    counts = df["error_type"].value_counts()
    tp, fp, fn, tn = (int(counts.get(t, 0)) for t in ("TP", "FP", "FN", "TN"))
    precision, recall = pct(tp, tp + fp), pct(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return dict(n=int(len(df)), TP=tp, FP=fp, FN=fn, TN=tn, precision=precision, recall=recall, F1=f1,
                normal_FPR=pct(fp, fp + tn), miss_rate=pct(fn, tp + fn))


def main(root: Path, output: Path, allow_changed: bool = False):
    root = root.resolve()
    output = output.resolve()
    base = root / "08_Modeling/results"
    if not allow_changed:
        hashes = verify_08(root)
    else:
        hashes = {name: git_blob(root / "08_Modeling" / name) for name in MODEL_BLOBS}
    cfg = json.loads((base / "final/final_config.json").read_text(encoding="utf-8"))
    if cfg.get("k") != 1 or cfg.get("A", {}).get("model") != "RandomForest" or cfg.get("B", {}).get("model") != "Mahalanobis":
        raise ValueError("이 분석 코드는 08 최종 A&B, k=1을 전제로 함. 다른 정책이면 코드를 검토해야 함")
    pred = load_predictions(root)
    pred["error_type"] = confusions(pred)
    if set(pd.to_numeric(pred.label).dropna().unique()) - {0, 1}:
        raise ValueError("모델 label은 0/1이어야 함")
    manifest = load_csv(root / "data/10_04_manifest_all.csv", [*KEY, "RMS_Detected", "H1_Residual_AI0", "H1_Residual_AI1", "H2_ND", "H3_Current_Amplitude"])
    assert_keys(manifest, "10_04_manifest_all.csv")
    mcols = [*KEY, "RMS_Detected", "H1_Residual_AI0", "H1_Residual_AI1", "H2_ND", "H3_Current_Amplitude"]
    pred = pred.merge(manifest[mcols], on=KEY, how="left", validate="one_to_one", indicator="_manifest_merge")
    if (pred._manifest_merge != "both").any():
        raise ValueError("예측과 manifest 공식 키 매칭 실패")
    pred.drop(columns="_manifest_merge", inplace=True)

    threshold = load_csv(base / "Error_analysis/C_Comparison/common_reason_thresholds.csv", ["metric", "value"])
    th = threshold.set_index("metric").value.to_dict()
    need = ["H1_AI0_Q99", "H1_AI1_Q99", "H2_Q01", "H2_Q99", "CURRENT_Q01", "CURRENT_Q99"]
    if set(need) - set(th):
        raise ValueError(f"08 OOF 정상 기준 Reason Threshold 누락: {sorted(set(need)-set(th))}")
    pred["rms_break"] = indicator(pred.RMS_Detected)
    pred["h1_break"] = (pd.to_numeric(pred.H1_Residual_AI0, errors="coerce") > th["H1_AI0_Q99"]) | (
        pd.to_numeric(pred.H1_Residual_AI1, errors="coerce") > th["H1_AI1_Q99"])
    nd = pd.to_numeric(pred.H2_ND, errors="coerce")
    cur = pd.to_numeric(pred.H3_Current_Amplitude, errors="coerce")
    pred["h2_break"] = nd.lt(th["H2_Q01"]) | nd.gt(th["H2_Q99"])
    pred["current_ood"] = cur.lt(th["CURRENT_Q01"]) | cur.gt(th["CURRENT_Q99"])

    # 공식 전처리의 행 단위 측정시간/세그먼트 정보만 읽고 결합한다. 새 전처리/segment 생성은 없음.
    # 04 실행 결과(data/processed)가 없으면 레포에 커밋된 동일 전처리본(data/preprocessed_data.csv)을 사용
    pp = root / "data/processed/preprocessed_data.csv"
    if not pp.exists():
        pp = root / "data/preprocessed_data.csv"
    src = load_csv(pp, [*KEY, "elapsed_sec", "seg_len"])
    assert_keys(src, "official preprocessing")
    pred.drop(columns=["elapsed_sec", "seg_len"], errors="ignore", inplace=True)
    pred = pred.merge(src[[*KEY, "elapsed_sec", "seg_len"]], on=KEY, how="left", validate="one_to_one")
    if pred.seg_len.isna().any():
        raise ValueError("preprocessed_data.csv official key 불일치")
    pred["regime"] = np.where(
        (pred.source == "normal") & pred.elapsed_sec.between(3600, 4300), "Low-RMS Normal candidate",
        np.where(pred.source == "normal", "Other Normal", "Abnormal (regime unknown)"))
    pred["segment_length_class"] = pd.cut(pred.seg_len, [0, 19, 39, np.inf], labels=["10-19 samples", "20-39 samples", "40+ samples"])
    pred["time_since_first_possible_sec"] = (pd.to_numeric(pred.pos_in_seg) - 9) * .1

    # 07 cycle은 RF 입력이 아니다. 연결 시에도 '설명 맥락'으로만 사용한다.
    cy = load_csv(base / "Error_analysis/C_Comparison/common_cycle_context_by_segment.csv", ["source", "segment_id", "cycle_context_available"], optional=True)
    if cy is not None:
        if cy.duplicated(["source", "segment_id"]).any():
            raise ValueError("cycle context segment key 중복")
        pred = pred.merge(cy, on=["source", "segment_id"], how="left", validate="many_to_one")
        pred["cycle_context_state"] = np.where(pred["cycle_context_available"].fillna(False).astype(bool), "Full-cycle context", "No full-cycle context")
    else:
        pred["cycle_context_state"] = "Unavailable"

    output.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output / "11_window_condition_context.csv", index=False, encoding="utf-8-sig")

    # 상호 중첩되는 조건은 합산 금지, 행별 원인 추정도 아님.
    condition_cols = ["regime", "segment_length_class", "rms_break", "h1_break", "h2_break", "current_ood", "cycle_context_state"]
    condition_rows = []
    for ev, sub in pred.groupby("eval_set", sort=False):
        condition_rows.append(dict(eval_set=ev, condition="ALL", value="ALL", **metrics(sub)))
        for col in condition_cols:
            for val, g in sub.groupby(col, dropna=False, observed=True):
                condition_rows.append(dict(eval_set=ev, condition=col, value=str(val), **metrics(g)))
    pd.DataFrame(condition_rows).to_csv(output / "11_condition_performance.csv", index=False, encoding="utf-8-sig")

    # 변수 간 상호작용: 두 조건을 교차해 정상 윈도우 오경보율·이상 윈도우 미탐률을 본다 (인과효과 아님).
    interaction_pairs = [("current_ood", "h1_break"), ("rms_break", "h1_break"), ("current_ood", "h2_break")]
    inter_rows = []
    for ev, sub in pred.groupby("eval_set", sort=False):
        for a, b in interaction_pairs:
            for (va, vb), g in sub.groupby([a, b], dropna=False):
                m = metrics(g)
                inter_rows.append(dict(eval_set=ev, factor_1=a, value_1=bool(va), factor_2=b, value_2=bool(vb),
                                       n_normal=m["FP"] + m["TN"], n_abnormal=m["TP"] + m["FN"], **m))
    pd.DataFrame(inter_rows).to_csv(output / "11_interaction_conditions.csv", index=False, encoding="utf-8-sig")

    seg = (pred.groupby(["eval_set", "source", "segment_id", "label"], dropna=False, as_index=False)
           .agg(n_windows=("final_alarm", "size"), n_alarm=("final_alarm", "sum"),
                n_fp=("error_type", lambda s: int(s.eq("FP").sum())),
                n_fn=("error_type", lambda s: int(s.eq("FN").sum())),
                rms_break_n=("rms_break", "sum"), h1_break_n=("h1_break", "sum"),
                h2_break_n=("h2_break", "sum"), current_ood_n=("current_ood", "sum"),
                cycle_context=("cycle_context_state", "first"),
                start_sec=("elapsed_sec", "min"), end_sec=("elapsed_sec", "max")))
    seg["is_error_segment"] = (seg.n_fp > 0) | (seg.n_fn > 0)
    seg.sort_values(["eval_set", "is_error_segment", "n_fp", "n_fn"], ascending=[True, False, False, False]).to_csv(
        output / "11_segment_case_review.csv", index=False, encoding="utf-8-sig")
    pred[pred.error_type.isin(["FP", "FN"])].to_csv(output / "11_fp_fn_windows.csv", index=False, encoding="utf-8-sig")

    # feature 영향요인은 이미 08 OOF에서 계산된 permutation importance와 ablation을 재사용한다.
    family = load_csv(base / "final/importance_family.csv", ["group", "PR_AUC_drop_mean"])
    feat = load_csv(base / "final/importance_feature.csv", ["group", "PR_AUC_drop_mean"])
    family.assign(interpretation="OOF permutation importance; 인과기여도 아님").to_csv(
        output / "11_feature_family_importance.csv", index=False, encoding="utf-8-sig")
    feat.assign(interpretation="OOF permutation importance; correlated feature 영향 주의").to_csv(
        output / "11_feature_importance.csv", index=False, encoding="utf-8-sig")
    ablation = load_csv(base / "final/ablation_table.csv", ["framework", "model", "features", "CV_PR_AUC", "TEST_F1"])
    ablation.to_csv(output / "11_model_ablation_reference.csv", index=False, encoding="utf-8-sig")

    # 물리 메커니즘 = 점검 가설. 이름만으로 원인을 식별하지 않음.
    hypothesis = [
        ("R1", "RMS amplitude 이상", "rms_break", "전류/진동 크기 변화", "운전조건·구동계 변동 가능성", "전기량·진동센서 범위 확인", "압력/품질/부품 고장 확정 불가"),
        ("R2", "Current–Vibration 정상관계 이탈", "h1_break", "전류에 비해 진동응답 이탈", "기계적 전달·구속·운전조건 변화 가능성", "전류와 두 진동채널 동시 점검", "전류→진동 인과관계 확정 불가"),
        ("R3", "Vibration 채널 균형 이탈", "h2_break", "AI0/AI1 균형 변화", "측정위치/구조응답/운전조건 가능성", "진동센서 체결·배선·설치위치 확인", "AI0·AI1 물리 위치 미확인"),
        ("R4", "Current 운전영역 이탈", "current_ood", "정상 Current support 벗어남", "구동상태 또는 데이터 수집조건 차이 가능성", "전원·모터·펌프 운전로그 확인", "전류값으로 유압·하중 산정 불가"),
        ("C1", "반복 timing 이탈", "cycle_timing_break", "AI2 약 1.7초 반복구조 변화", "펌프/전기 반복현상 변화 가능성", "PLC/모터 회전·주파수 동시계측", "actual press stroke 및 aliasing 미확인"),
        ("C2", "Cycle current support 이탈", "cycle_current_break", "cycle 수준 전류영역 이탈", "정상 운전범위 밖 가능성", "전기 운전로그 확인", "전류 외삽 회귀 해석 제한"),
        ("C3", "PORD / 응답 이탈", "cycle_response_break", "전류 조건부 진동응답 이탈", "구동-구조 전달상태 변동 가능성", "유압 압력·유량·진동 동기 검증", "물리적 고장원인 확정 불가"),
        ("C4", "AI2 cycle shape 이탈", "cycle_shape_break", "정규화된 파형 구조 변화", "반복 전기/공정 상태 변동 가능성", "원취득률·필터·sampling 및 PLC 비교", "교류 aliasing으로 인한 가상 반복 가능"),
    ]
    pd.DataFrame(hypothesis, columns=["reason_code", "signal_pattern", "source_flag", "observed_meaning", "physical_hypothesis_NOT_PROVEN", "verification_needed", "interpretation_limit"]).to_csv(
        output / "11_physical_hypothesis_matrix.csv", index=False, encoding="utf-8-sig")

    status = load_csv(base / "final/rows_final_status.csv", [*KEY, "label", "eval_status"])
    available = status.groupby(["source", "eval_status"], observed=True).size().rename("rows").reset_index()
    available.to_csv(output / "11_coverage_by_source.csv", index=False, encoding="utf-8-sig")

    summary = {
        "reference_commit": COMMIT, "source_sha": hashes, "model_training": False, "policy_changed": False,
        "model_A": cfg["A"], "model_B": cfg["B"], "k": cfg["k"],
        "metrics": {ev: metrics(df) for ev, df in pred.groupby("eval_set")},
        "case_count": int(len(pred[pred.error_type.isin(["FP", "FN"])])),
        "warnings": [
            "TEST 오차조건 분석은 사후 설명이며, 테스트 성능을 보고 모델·threshold를 수정하지 않는다.",
            "Abnormal과 Normal의 취득일이 달라 session confounding을 배제할 수 없다.",
            "공식 TEST abnormal segment 수가 작고 window가 중첩돼 표본 독립성이 낮다.",
            "Cycle/PORD/Shape은 최종 08 RF의 직접 입력이 아닌 공정설명용 맥락이다.",
            "AI2 약 1.7초 반복은 motor current aliasing 가능성. stroke/phase 확정 불가.",
            "상호 중첩되는 조건별 성능을 독립 집단으로 합산하지 않는다.",
            "11_interaction_conditions.csv는 두 조건의 교차 집계이며 인과적 상호작용 효과가 아니다.",
        ],
    }
    (output / "11_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[11 DONE]", output)
    for name in ["OOF", "TEST"]:
        if name in summary["metrics"]:
            print(name, summary["metrics"][name])
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd(), help="KAMP repository root")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--allow-changed-08", action="store_true", help="검증 목적에서만 08 코드 해시 불일치 허용")
    args = p.parse_args()
    main(args.root, args.output or args.root / "11_Process_Interpretation/results", args.allow_changed_08)
