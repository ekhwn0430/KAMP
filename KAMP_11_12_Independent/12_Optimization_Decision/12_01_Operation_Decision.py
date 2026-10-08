# 12. 최적화·의사결정: 08에서 확정된 A&B 경보를 변경하지 않고 현장 점검 큐와 KPI를 산출한다.
# TEST 결과를 보고 새 임계값이나 k를 고르지 않으며, k=2는 기존 08 결과의 민감도 분석으로만 제공한다.
# 실제 pressure/flow/load setpoint 최적화는 근거 부족으로 수행하지 않는다.
# 실행: python 12_Optimization_Decision/12_01_Operation_Decision.py --root <KAMP_repo>

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ["source", "segment_id", "source_row"]


def read(path: Path, cols=()):
    if not path.exists():
        raise FileNotFoundError(f"필요 파일 없음: {path}. 먼저 08과 11 결과를 확인하세요")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: {sorted(missing)} 누락")
    return df


def is_true(series):
    txt = series.astype(str).str.strip().str.lower()
    return txt.isin(["true", "1", "1.0", "yes", "y"]) | pd.to_numeric(series, errors="coerce").eq(1)


def detect_route(row):
    if row["eval_status"] != "evaluated":
        return "계측·수집 연속성 / 10개 유효 sample 확보 확인"
    parts = []
    if bool(row.get("current_ood", False)):
        parts.append("전류 운전영역/전원·모터·펌프 운전로그 확인")
    if bool(row.get("h1_break", False)):
        parts.append("전류 대비 진동응답·센서체결/설치상태 비교")
    if bool(row.get("h2_break", False)):
        parts.append("두 진동채널 균형·배선·센서 위치 점검")
    if bool(row.get("rms_break", False)):
        parts.append("진동/전류 절대값 변화·동시 운전조건 점검")
    if bool(row.get("cycle_timing_break", False)) or bool(row.get("cycle_shape_break", False)):
        parts.append("AI2 반복파형·PLC·원 샘플링/aliasing 확인")
    return " | ".join(parts) if parts else "08 모델 경보의 센서 파형 재검토(사후 reason code 불명)"


def main(root: Path, in11: Path, output: Path):
    root, in11, output = root.resolve(), in11.resolve(), output.resolve()
    base = root / "08_Modeling/results/final"
    cfg = json.loads((base / "final_config.json").read_text(encoding="utf-8"))
    audit = json.loads((in11 / "11_audit_summary.json").read_text(encoding="utf-8"))
    if cfg.get("k") != 1 or audit.get("k") != 1 or audit.get("policy_changed"):
        raise ValueError("11-12 기준은 08 A&B k=1 고정. 설정 불일치")

    rows = read(base / "rows_final_status.csv", [*KEY, "label", "eval_status", "eval_set", "final_alarm", "alarm_level"])
    detail = read(in11 / "11_window_condition_context.csv", [*KEY, "eval_set", "final_alarm", "rms_break", "h1_break", "h2_break", "current_ood"])
    if rows.duplicated(KEY).any() or detail.duplicated(KEY).any():
        raise ValueError("공식 키 중복")
    detail_cols = [*KEY, "eval_set", "rms_break", "h1_break", "h2_break", "current_ood",
                   "cycle_timing_break", "cycle_shape_break"]
    detail = detail[[c for c in detail_cols if c in detail.columns]]
    rows = rows.merge(detail, on=[*KEY, "eval_set"], how="left", validate="one_to_one")
    status = rows["eval_status"].astype(str)
    not_eval = status == "not_evaluable"
    if (status[~not_eval] != "evaluated").any():
        raise ValueError("허용되지 않은 eval_status")
    if rows.loc[~not_eval, "rms_break"].isna().any():
        raise ValueError("11 결과가 평가 가능 08 window 전체와 일치하지 않음")

    for f in ["rms_break", "h1_break", "h2_break", "current_ood", "cycle_timing_break", "cycle_shape_break"]:
        rows[f] = is_true(rows[f]) if f in rows.columns else False
    alarm = is_true(rows["final_alarm"])
    caution = rows["alarm_level"].astype(str).eq("주의")
    # 주의/경보를 운용 규칙에 없는 true label로 결정하면 leakage가 되므로 사용 금지.
    rows["operational_state"] = np.select(
        [not_eval, alarm, caution], ["평가불가", "경보", "주의"], default="정상")
    rows["priority"] = rows.operational_state.map({"경보": "P1_즉시확인", "주의": "P2_재관찰", "평가불가": "P0_계측확인", "정상": "P3_운전유지"})
    rows["operator_action"] = rows.operational_state.map({
        "경보": "담당자 알림, 최근 파형 및 설비 로그 확인 후 현장 점검; 설비 정지는 별도 안전절차 판단",
        "주의": "동일 segment의 후속 유효 윈도우 관찰, 센서·운전로그 재확인",
        "평가불가": "정상 간주 금지, 센서 누락·gap·윈도우 미완성 확인; 데이터 수집 재확보",
        "정상": "운전 유지 및 정기 모니터링"})
    rows["inspection_route"] = rows.apply(detect_route, axis=1)
    rows["decision_source"] = "08 frozen A&B k=1; 11 reason context is post-hoc"
    # label은 운영 큐에서 제거. 교육/검증용 실제 라벨은 별도 사후평가 파일로만 보존.
    output.mkdir(parents=True, exist_ok=True)
    # 공식 source/segment_id에는 Normal/Abnormal 정답 정보가 담겨 있어 운영자용 화면에는 노출하지 않는다.
    # 재현성 추적은 11의 공식 key 결합표에서만 수행하고, 운영 큐에는 익명 식별자를 쓴다.
    rows["segment_tracking_id"] = rows.apply(
        lambda r: hashlib.sha256(f"{r['source']}|{r['segment_id']}".encode()).hexdigest()[:16], axis=1)
    rows["case_tracking_id"] = rows.apply(
        lambda r: hashlib.sha256(f"{r['source']}|{r['segment_id']}|{r['source_row']}".encode()).hexdigest()[:20], axis=1)
    opcols = ["case_tracking_id", "segment_tracking_id", "pos_in_seg", "seg_len", "eval_set", "eval_status", "A_prob", "B_normal_pct",
              "alarm_level", "final_alarm", "operational_state", "priority", "operator_action", "inspection_route", "decision_source"]
    opcols = [x for x in opcols if x in rows.columns]
    queue = rows.loc[:, opcols].sort_values(["priority", "segment_tracking_id", "pos_in_seg"], kind="stable")
    queue.to_csv(output / "12_operational_decision_all_rows.csv", index=False, encoding="utf-8-sig")
    queue[queue.operational_state.isin(["주의", "경보", "평가불가"])].to_csv(
        output / "12_inspection_queue.csv", index=False, encoding="utf-8-sig")

    # 운영대상 전체 20,599 행의 '평가불가' 관리. 평가불가를 TN으로 넣지 않음.
    coverage = (rows.groupby(["source", "eval_set", "operational_state"], dropna=False).size().rename("n_rows").reset_index())
    coverage.to_csv(output / "12_monitoring_coverage.csv", index=False, encoding="utf-8-sig")

    # 경보/주의/정상/평가불가의 사후 label별 타당성 분석. label이 의사결정에 관여하지 않음.
    audit_table = (rows.groupby(["eval_set", "operational_state", "label"], dropna=False)
                   .size().rename("n_rows").reset_index())
    audit_table.to_csv(output / "12_state_posthoc_label_audit.csv", index=False, encoding="utf-8-sig")

    # 점검 건은 알람 window 개수와 구분. 한 segment 경보가 여러 개면 점검 건 1개로 취급한다.
    seg = (rows.groupby(["eval_set", "source", "segment_id"], dropna=False, as_index=False)
           .agg(total_rows=("source_row", "size"), evaluated_rows=("eval_status", lambda x: int(x.eq("evaluated").sum())),
                n_alert=("operational_state", lambda x: int(x.eq("경보").sum())),
                n_caution=("operational_state", lambda x: int(x.eq("주의").sum())),
                n_not_eval=("operational_state", lambda x: int(x.eq("평가불가").sum()))))
    seg["highest_state"] = np.select([seg.n_alert > 0, seg.n_caution > 0, seg.evaluated_rows == 0],
                                      ["경보", "주의", "평가불가"], default="정상")
    seg.to_csv(output / "12_segment_dispatch.csv", index=False, encoding="utf-8-sig")

    # 08의 OOF에서 이미 정해진 k를 변경하지 않고 대안 k를 비교표로만 제시한다.
    policy = read(base / "policy_comparison.csv", ["eval_set", "rule", "k", "Recall", "FalseAlarmRate", "AbnSegDetectRate", "NormSegFalseAlarmRate"])
    policy["status"] = np.where((policy.rule == "A·B 모두(경보)") & (policy.k == cfg["k"]), "FINAL_FROZEN", "SENSITIVITY_ONLY")
    policy["normal_FP_per_1000_evaluable"] = 1000 * policy["FalseAlarmRate"]
    policy.to_csv(output / "12_policy_tradeoff_READONLY.csv", index=False, encoding="utf-8-sig")

    ablation = read(base / "ablation_table.csv", ["framework", "model", "features", "CV_PR_AUC", "TEST_F1", "TEST_FalseAlarm"])
    # Baseline은 공식 08 ablation, 최종 실제 A&B는 08 policy table의 성능. 별개 benchmark임을 구분.
    base_rms = ablation[ablation.model == "RMS-IQR"]
    final = policy[(policy.eval_set == "TEST") & (policy.status == "FINAL_FROZEN")]
    impact = []
    if len(base_rms):
        b = base_rms.iloc[0]
        impact.append({"method": "RMS-IQR", "evaluation": "TEST", "F1": b.TEST_F1,
                       "Recall": b.TEST_Recall if "TEST_Recall" in b.index else np.nan,
                       "NormalFPR": b.TEST_FalseAlarm, "source": "08 ablation_table.csv"})
    if len(final):
        f = final.iloc[0]
        impact.append({"method": "FINAL_AND_k1", "evaluation": "TEST", "F1": f.F1, "Recall": f.Recall,
                       "NormalFPR": f.FalseAlarmRate, "source": "08 policy_comparison.csv"})
    impact = pd.DataFrame(impact)
    if len(impact) == 2:
        b, f = impact.iloc[0], impact.iloc[1]
        impact["normal_FPR_reduction_vs_RMS"] = [np.nan, float((b.NormalFPR - f.NormalFPR) / b.NormalFPR) if b.NormalFPR else np.nan]
    impact.to_csv(output / "12_baseline_vs_final.csv", index=False, encoding="utf-8-sig")

    # 제4장 KPI는 08 소스 그대로 재사용하고 '후보 조치'라는 한계를 기록한다.
    kpis = [
        {"kpi": "TEST evaluated-window F1", "value": float(final.iloc[0].F1) if len(final) else np.nan, "source": "08 frozen policy"},
        {"kpi": "TEST evaluated-normal FPR", "value": float(final.iloc[0].FalseAlarmRate) if len(final) else np.nan, "source": "08 frozen policy"},
        {"kpi": "TEST abnormal window recall", "value": float(final.iloc[0].Recall) if len(final) else np.nan, "source": "08 frozen policy"},
        {"kpi": "TEST abnormal segment detect rate", "value": float(final.iloc[0].AbnSegDetectRate) if len(final) else np.nan, "source": "08 frozen policy"},
        {"kpi": "Not-evaluable rows (ALL)", "value": int(not_eval.sum()), "source": "08 rows_final_status.csv"},
        {"kpi": "Not-evaluable abnormal rows (ALL)", "value": int((not_eval & (rows.label == 1)).sum()), "source": "08 rows_final_status.csv"},
        {"kpi": "Not-evaluable normal rows (ALL)", "value": int((not_eval & (rows.label == 0)).sum()), "source": "08 rows_final_status.csv"},
    ]
    pd.DataFrame(kpis).to_csv(output / "12_operational_kpi.csv", index=False, encoding="utf-8-sig")

    rules = [
        {"state": "정상", "logical_condition": "evaluable and A_alarm=0 and B_alarm=0", "next_step": "정기 모니터링", "risk": "정상 예측도 실제 설비 상태 확증 아님"},
        {"state": "주의", "logical_condition": "evaluable and exactly one of A_alarm/B_alarm=1", "next_step": "재관찰, 운전로그 비교, 센서 확인", "risk": "주의만으로 부품 고장 확정 금지"},
        {"state": "경보", "logical_condition": "evaluable and A_alarm=1 and B_alarm=1; k=1", "next_step": "우선점검/현장 담당자 확인", "risk": "자동 정지/압력 변경 등 제어 미검증"},
        {"state": "평가불가", "logical_condition": "segment 첫 9행 또는 유효 10 sample 부족", "next_step": "결측/수집 상태 확인 및 재확보", "risk": "정상 처리 금지"},
    ]
    pd.DataFrame(rules).to_csv(output / "12_operating_rules.csv", index=False, encoding="utf-8-sig")
    limit = [
        {"item": "펌프 압력 setpoint", "possible": False, "reason": "압력 계측치, 밸브 제어권한, 영향 실험 없음"},
        {"item": "유량·속도 setpoint", "possible": False, "reason": "유량·램 위치·사이클 물리상태 측정 없음"},
        {"item": "피크전력 최적화", "possible": False, "reason": "AI2 측정 위치·전압·역률 미상; Current != Power"},
        {"item": "진동·전류 관계 기반 검사 우선순위", "possible": True, "reason": "08 예측과 11 사후 EDA explanation에 근거한 점검 추천만 가능"},
        {"item": "k=2 운영모드", "possible": True, "reason": "08에서 계산된 민감도 비교로만 제시, 현재 고정 k=1 변경 금지"},
        {"item": "자동 설비 정지", "possible": False, "reason": "현장 안전·정지비용·가동 검증 없음; 관리자 승인 필수"},
    ]
    pd.DataFrame(limit).to_csv(output / "12_physical_optimization_feasibility.csv", index=False, encoding="utf-8-sig")

    summary = {
        "reference_commit": audit["reference_commit"], "model_modified": False, "policy_modified": False,
        "policy": cfg["final_rule"], "k": cfg["k"], "n_all_rows": int(len(rows)),
        "states": {str(k): int(v) for k, v in rows.operational_state.value_counts().to_dict().items()},
        "limitations": [
            "운영상 inspection_route는 사후 데이터 해석에서 도출한 점검 제안이지 모델에 내장된 원인진단이 아니다.",
            "0.9초는 10-sample 윈도우의 최초 판정 가능 시점이며 실제 고장 발생부터의 조기탐지 지연시간이 아니다.",
            "동일 segment의 겹치는 window는 독립 표본이 아니다.",
            "수집일 차이에 따른 모델 일반화 불확실성이 남아 있다.",
            "압력·유량·품질 인과 검증과 실제 밸브·펌프 제어 최적화는 수행하지 않았다.",
            "현장 대응 표는 제안된 운영 절차이며 산업현장에서 검증된 안전 절차가 아니다.",
        ],
    }
    (output / "12_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[12 DONE]", output)
    print("운영 상태 (평가불가 포함):", summary["states"])
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--input11", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    main(args.root, args.input11 or args.root / "11_Process_Interpretation/results",
         args.output or args.root / "12_Optimization_Decision/results")
