# 11·12 독립 분석 파이프라인 (기준 커밋 `b1960321`)

## 목적

**08 최종 RandomForest + Mahalanobis 모델을 재학습·수정하지 않고**, `08_Modeling/results/`의 확정 예측 결과와 07/EDA 맥락만 읽어 다음 산출물을 계산한다.

- **11 공정해석:** 실제 Feature 중요도, 기존 F0~F5 ablation, OOF/TEST 조건별 FP/FN, 07 Cycle/PORD/Shape 사후 설명, 물리적 고장 가설과 확인 불가 사항 구분
- **12 최적화/의사결정:** 기존 08의 `A&B, k=1` 규칙 그대로 정상/주의/경보/평가불가 → 점검 큐, 점검 우선순위, 운영 KPI, 기존 k=2 정책의 tradeoff 참고
- **13 보고서 지원:** 제3장 영향요인·실패조건 (11 결과 CSV), 제4장 검사·조치 (12), 제5장 수치 기반 차별성 (08 ablation + 11/12)

> **이 패키지는 결과보고서를 자동으로 작성하는 도구가 아니다.** 보고서에 근거로 인용할 수 있는 정량 CSV·감사 JSON을 별도로 출력한다.

## 폴더에 넣기

최종 커밋의 KAMP 레포 최상단에 `KAMP_11_12_Independent` 폴더를 그대로 복사한다. `08_Modeling/` 파일은 복사하거나 변경하지 않는다.

```
KAMP/
├── 08_Modeling/                     # 절대 수정 금지
│   └── results/final/...
├── data/processed/preprocessed_data.csv
├── data/10_04_manifest_all.csv
└── KAMP_11_12_Independent/
    ├── run_11_12.py
    ├── 11_Process_Interpretation/11_01_Process_Interpretation.py
    ├── 12_Optimization_Decision/12_01_Operation_Decision.py
    └── tests/test_independent_pipeline.py
```

실행:

```bash
python KAMP_11_12_Independent/run_11_12.py --root .
```

사용 중 Python의 `pandas` 및 `numpy`가 필요하다. 모델 fitting은 수행하지 않으므로 새로운 ML 패키지나 의존성이 추가되지 않는다. 기존 `requirements.txt`로 충분하다.

**결과는 전부 `KAMP/11_12_results/11/` 및 `KAMP/11_12_results/12/`에만 저장한다.** 코드는 08 내부로 출력을 금지한다. 파일 경로에는 한글·영문 혼용 가능.

## 필수 입력 목록

레포 최상단의 `python run_all.py`가 마지막 단계로 이 패키지를 실행한다. 08 해시 비교는 줄바꿈(CRLF/LF)을 통일해서 하므로 Windows 체크아웃에서도 동작한다.


- `08_Modeling/08_01_model_comparison.py`, `08_02_Model_Error_Analysis.py`, `08_03_Final_Alarm_Policy.py`의 Git blob 해시가 기준 커밋과 일치해야 한다. 불일치 시 **즉시 실패**.
- `08_Modeling/results/final/{test_predictions_final.csv,rows_final_status.csv,final_config.json,policy_comparison.csv,ablation_table.csv,importance_family.csv,importance_feature.csv}`
- `08_Modeling/results/{oof_predictions_A.csv,oof_predictions_B.csv}`: 있을 경우 TEST + OOF 모두 분석
- `08_Modeling/results/Error_analysis/C_Comparison/common_reason_thresholds.csv`: 정상 OOF에서 이미 확정한 Reason Code 기준
- `08_Modeling/results/Error_analysis/C_Comparison/common_cycle_context_by_segment.csv`: 있을 경우 07 cycle 설명 맥락 결합
- `data/10_04_manifest_all.csv`: H1/H2/H3a + RMS row-wise 값
- `data/processed/preprocessed_data.csv`: `elapsed_sec`, `seg_len` 등 원본 공식 키 정보 (없으면 레포의 `data/preprocessed_data.csv` 사용)

## 11단계 결과

| 산출물 | 평가기준 연결 | 내용 |
|---|---|---|
| `11_interaction_conditions.csv` | 제3장 | 두 조건 교차(전류 운전영역×전류-진동 관계, RMS×H1, 전류×H2)별 정상 오경보율·미탐률. 인과효과 아님 |
| `11_condition_performance.csv` | 제3장 | OOF/TEST, Low-RMS Normal, 길이별, RMS 이탈, H1/H2, Current OOD, cycle availability별 TP/FP/FN/TN·Recall·FPR |
| `11_segment_case_review.csv` | 제3장 | FP/FN segment 집중 위치, 각 신호 유형 이탈 개수 |
| `11_fp_fn_windows.csv` | 제3장 | FP/FN 판정시점의 공식 key, 전류/진동 관계, 설명 가능한 범위 |
| `11_feature_family_importance.csv` | 제3·5장 | 08에서 측정한 permutation PR-AUC 감소량 복사 및 해석 제한 |
| `11_feature_importance.csv` | 제3·5장 | 개별 Feature 중요도 |
| `11_model_ablation_reference.csv` | 제3·5장 | 08 모델/Feature ablation 표 그대로 사용 |
| `11_physical_hypothesis_matrix.csv` | 제3·4장 | 관측 패턴, 가능한 메커니즘, 추가 확인할 센서/운전로그, 인과해석 제한 |
| `11_coverage_by_source.csv` | 제3·4장 | 20,599행 중 evaluable / not_evaluable 구분 |
| `11_window_condition_context.csv` | 내부 연계 | OOF/TEST 예측 결과 + 04 manifest + 07 cycle context 결합 |
| `11_audit_summary.json` | 13단계 | 기준 commit·원본 08 SHA·오류분석 수치·한계 감사 기록 |

## 12단계 결과

| 산출물 | 평가기준 연결 | 내용 |
|---|---|---|
| `12_operational_decision_all_rows.csv` | 제4장 | 20,599행 정상/주의/경보/평가불가, 운영 우선순위, 점검 경로 (**source/segment_id/실제 label 제외, 익명화 ID 사용**) |
| `12_inspection_queue.csv` | 제4장 | 주의/경보/평가불가 행의 점검 목록 |
| `12_segment_dispatch.csv` | 제4장 | 중첩 윈도우를 하나의 segment 점검 건으로 요약 |
| `12_monitoring_coverage.csv` | 제4장 | 평가 가능/불가능 현황 및 운영 상태 |
| `12_state_posthoc_label_audit.csv` | 제3장 | 정답 label로 사후 검증하는 감사표 (운영 판단에는 label 미사용) |
| `12_policy_tradeoff_READONLY.csv` | 제4·5장 | 08 OOF/TEST의 k1/k2/k3/k5 원본 결과; **k 재선정 금지** |
| `12_baseline_vs_final.csv` | 제5장 | 08 RMS-IQR vs 08 A&B k1의 TEST F1/Recall/Normal FPR |
| `12_operational_kpi.csv` | 제4장 | F1/Recall/FPR/segment recall/not-evaluable 지표 |
| `12_operating_rules.csv` | 제4장 | 네 가지 운영 상태별 대응 및 한계 |
| `12_physical_optimization_feasibility.csv` | 제4장 | 가능한 검사 우선순위 최적화 vs 불가능한 압력/유량/피크전력 최적화 구분 |
| `12_audit_summary.json` | 13단계 | 정책 불변·분석범위·한계 증거 기록 |

## 분석적으로 지켜야 할 규칙

1. **08이 유일한 모델**. 11·12에서 새 threshold를 TEST에 맞춰 찾지 않는다. 새 분류기를 만들거나 fitting하지 않는다.
2. **08 final `k=1`이 배포 기준**이다. k=2는 08에서 평가한 정책 후보의 차이만 설명하며 운영정책을 바꾸지 않는다.
3. **Full-cycle PORD/Shape은 RF feature가 아니다.** 11단계는 07 사후 해석으로만 합친다. `No full-cycle context`는 정상 판정이 아니다.
4. **원인확정 금지.** AI2 sampling/aliasing, 압력 미관측, 전류 측정 위치 미상, AI0/AI1 설치 위치 미상, Normal·Abnormal 취득일 차이로 유압·금형 고장을 특정할 수 없다.
5. **실제 제어 최적화 금지.** 유압 밸브 조작, 자동 정지, 실제 피크전력 저감은 검증되지 않았다. 12는 검사·정비 *의사결정 지원*에 국한된다.
6. **평가불가 ≠ 정상.** 불완전 1초 window의 데이터 품질 점검이 반드시 필요하다.
7. **데이터 유출 방지.** 운영 큐에는 `label`, `source`, `segment_id`가 들어가지 않는다. `segment_id`의 N/A 접두어 자체가 정답 정보를 포함하므로 익명 tracking ID를 사용한다. 정답과 공식 키는 사후 평가 파일에만 남는다.
8. **조건별 비율은 중첩 조건.** 조건별 수치를 더해 전체 성능을 만들지 않는다. OOF와 TEST를 한 숫자로 섞지 않는다.
9. **검증 수준 제한.** TEST 이상 segment 4개, normal/abnormal 수집일 차이, 중첩 window 때문에 신규 현장 일반화 결론을 내릴 수 없다.

## 간이 테스트

```bash
python KAMP_11_12_Independent/tests/test_independent_pipeline.py
```

이 테스트는 **합성 CSV의 입력/출력 계약을 검증하는 smoke test**다. 합성 테스트가 통과했다고 해서 실제 GitHub 최종 산출물 전체에 대한 재실행 검증이 완료된 것은 아니다. 실제 저장소에서 `run_11_12.py`를 실행한 뒤 `11_audit_summary.json`, `12_audit_summary.json`에서 수치를 확인한다.
