# 진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석

제6회 K-인공지능 제조데이터 분석 경진대회 과제 ③ (소성가공 예지보전 데이터셋)

파인블랭킹 프레스 유압펌프 모터의 진동 2채널(AI0, AI1)과 전류(AI2) 시계열로 설비 이상을 1초 단위로 탐지하고,
오경보를 줄이는 경보 규칙과 미탐지·오경보 발생 조건을 분석한다.

## 0. 제출물 위치

| 항목 | 위치 |
|---|---|
| 소스코드 | `run_all.py`와 `04_Data_Processing/` ~ `08_Modeling/` (실행 순서는 4장) |
| 환경 | `requirements.txt` |
| 학습용 데이터 | `data/press_data_normal.csv`, `data/outlier_data.csv` |
| 테스트데이터 예측결과 | `08_Modeling/results/final/test_predictions_final.csv` |
| 오류조건·운영 의사결정 결과 | `11_12_results/11/`, `11_12_results/12/` |

## 1. 실행 환경

- Python 3.13에서 전체 실행 확인 (3.14에서도 08_01 실행 확인)
- 패키지: `pip install -r requirements.txt` (scikit-learn 1.9.0 고정. 버전이 다르면 HistGB 점수가 달라져 모델 선정이 바뀔 수 있음)

## 2. 실행 방법

레포 최상단에서 한 줄로 전처리부터 최종 결과까지 실행한다 (약 9분).

```bash
pip install -r requirements.txt
python run_all.py
```

특정 단계부터 다시 실행: `python run_all.py --from 08`

## 3. 입력 데이터 (`data/`)

| 파일 | 내용 |
|---|---|
| `press_data_normal.csv` | 정상 20,000행 (2022-07-12 00:00~01:16) |
| `outlier_data.csv` | 이상 600행 (2022-07-17 10:51~10:53) |
| `10_04_manifest_all.csv` | 행 단위 RMS-IQR 분류와 H1/H2/H3a 값 (팀 RMS 분석 결과, 08_02 오류분석에서 사용) |

## 4. 파이프라인

| 순서 | 스크립트 | 하는 일 | 주요 출력 |
|---|---|---|---|
| 1 | `04_Data_Processing/preprocess.py` | 중복 1행 제거, 0.15초 초과 공백에서 segment 분할 (정상 599 / 이상 21) | `data/processed/preprocessed_data.csv` |
| 2 | `07_Feature_Engineering/07_01_Feature.py` | AI2 peak 기준 cycle 단위 feature | `07_Feature_Engineering/Data/` |
| 3 | `05_EDA/05_Cycle/05_EDA_4_Outlier_Cycle.py` | Normal에서 고정한 약 1.7초 AI2 Cycle 기준을 이상 데이터에 적용 | `05_EDA/05_Cycle/results/cycle_outlier/` |
| 4 | `05_EDA/05_Cycle/05_EDA_5_Merge_Cycle.py` | 행 단위 + cycle 단위 Cycle 결과 merge | `merged_cycle_rows_cycle_units.csv` |
| 5 | `07_Feature_Engineering/07_02_Feature_validation.py` | cycle feature 분리도·오경보율 검증 | `07_Feature_Engineering/Experiments/` |
| 6 | `07_Feature_Engineering/07_03_Final_Feature_Selection.py` | 설명용 핵심 feature 선정 (KEEP_CORE 8개) | `07_Feature_Engineering/Final_Selection/` |
| 7 | `08_Modeling/08_01_model_comparison.py` | 1초 윈도우 특징 56개, 모델 5종 × 특징 계단 5단계 비교, A/B 모델 선정 | `08_Modeling/results/` |
| 8 | `08_Modeling/08_02_Model_Error_Analysis.py` | FN/FP 오류분석, Reason Code, A vs B 비교 | `08_Modeling/results/Error_analysis/` |
| 9 | `08_Modeling/08_03_Final_Alarm_Policy.py` | 확률보정, 경보 등급·연속경보 규칙, 탐지 지연, permutation importance, 최종 예측 | `08_Modeling/results/final/` |
| 10 | `KAMP_11_12_Independent/run_11_12.py` | 11 공정해석(조건별·상호작용 오류분석, 점검 가설), 12 운영 의사결정(4단계 상태, 점검 우선순위, KPI) | `11_12_results/11/`, `11_12_results/12/` |

11·12 단계(`KAMP_11_12_Independent/run_11_12.py`)는 08 결과를 읽기만 하며 모델을 다시 학습하거나 임계값을 바꾸지 않는다. 08 코드 3개의 Git 해시로 버전을 고정해, 08 코드가 바뀌면 실행을 멈춘다. 자세한 내용은 `KAMP_11_12_Independent/README_11_12.md`.

07_01을 05_Cycle보다 먼저 실행하는 이유: `05_EDA_4_Outlier_Cycle.py`가 07_01의 cycle 표를 읽어 참고 컬럼(`cycle_id_07`)을 채운다. 07_01은 전처리 데이터만 사용한다.

## 5. 검증 설계

- 분석 단위: segment 내부 1초(10샘플) 윈도우, 판정 시점은 윈도우 마지막 행. segment 경계를 넘어 계산하지 않음
- test: segment 번호 5개마다 1개를 고정 홀드아웃 (평가 가능한 segment 기준 정상 106 / 이상 4). 최종 평가에만 사용
- train: segment 단위 StratifiedGroupKFold 5-fold의 out-of-fold 예측으로 모델·임계값·경보 규칙 결정
- 스케일링, H1 잔차 회귀, IQR 범위는 fold의 train 정상 데이터에서만 fit
- 예측이 없는 행(segment 첫 9행, 10행 미만 segment)은 `not_evaluable`로 따로 표기하고 정상으로 세지 않음 (이상 172행, 정상 5,128행)

## 6. 최종 모델과 경보 규칙

| 구분 | 내용 |
|---|---|
| 모델 A | RandomForest 지도학습, 특징 F4(진동 + 전류 크기 + 채널 관계 + H1 잔차), isotonic 확률보정 |
| 모델 B | Mahalanobis 거리, 정상 데이터만 학습, 임계값 = 정상 OOF 점수 99% 분위 |
| 경보 등급 | 정상(0) / 주의(1: A·B 중 하나만 이상) / 경보(2: A·B 모두 이상) |
| 연속경보 k | train OOF에서 1·2·3·5 비교 → k=1 (k≥2는 1개 윈도우뿐인 이상 segment를 놓침) |

TEST 성능 (윈도우 3,168개: 정상 3,039 / 이상 129)

| 규칙 | Recall | Precision | F1 | F2 | 정상 오경보율 | 이상 segment 탐지 | 정상 segment 오경보 |
|---|---|---|---|---|---|---|---|
| RMS-IQR 베이스라인 | 0.705 | 0.827 | 0.762 | 0.727 | 0.63% | 4/4 | 4/106 |
| A만 | 1.000 | 0.956 | 0.977 | 0.991 | 0.20% | 4/4 | 2/106 |
| B만 | 1.000 | 0.750 | 0.857 | 0.938 | 1.41% | 4/4 | 14/106 |
| **A·B 모두 (최종 경보)** | **1.000** | **0.963** | **0.981** | **0.992** | **0.16%** | **4/4** | **2/106** |

- 모든 이상 segment에서 첫 판정 가능 시점(관측 시작 후 0.9초)에 경보가 발생함
- 확률보정: TEST Brier score 0.00205 → 0.00064
- 운영 옵션: 오경보를 더 줄여야 하는 현장은 A·B 모두 k=2 (TEST 정상 오경보 0.07%, 정상 segment 오경보 1/106, 이상 윈도우 Recall 0.969)

## 7. 결과 파일

| 파일 | 내용 |
|---|---|
| `08_Modeling/results/final/test_predictions_final.csv` | 최종 test 예측 (아래 컬럼 설명) |
| `08_Modeling/results/final/policy_comparison.csv` | 경보 규칙 × k별 window·segment 성능 (OOF/TEST) |
| `08_Modeling/results/final/ablation_table.csv` | 모델 × 특징 계단 비교 (PR-AUC, F1, F2, Confusion Matrix) |
| `08_Modeling/results/final/importance_family.csv` | feature 묶음별 permutation importance |
| `08_Modeling/results/final/rows_final_status.csv` | 전체 20,599행 평가 상태 (not_evaluable 포함) |
| `08_Modeling/results/Error_analysis/` | FN/FP 목록, Reason Code, A vs B 비교 |
| `05_EDA/05_Cycle/results/cycle_outlier/` | Cycle 분석 결과 (segment별 판정, 회의용 숫자) |
| `11_12_results/11/11_condition_performance.csv` | 운전구간·segment 길이·신호 이탈 조건별 TP/FP/FN/TN, Recall, 오경보율 |
| `11_12_results/11/11_interaction_conditions.csv` | 두 조건 교차(전류 운전영역 × 전류-진동 관계 등)별 오경보율·미탐률 |
| `11_12_results/11/11_fp_fn_windows.csv` | 오경보·미탐지 윈도우 목록과 신호 조건 |
| `11_12_results/12/12_operating_rules.csv` | 정상/주의/경보/평가불가 운영 규칙과 대응 |
| `11_12_results/12/12_segment_dispatch.csv` | segment 단위 점검 건 집계 |
| `11_12_results/12/12_policy_tradeoff_READONLY.csv` | 경보 규칙 k별 탐지율·오경보 비교 (08 결과 그대로) |

공통 key: `source` (normal/abnormal) + `segment_id` (N0000/A0000) + `source_row` (원본 CSV 행 번호)

`test_predictions_final.csv` 컬럼

| 컬럼 | 의미 |
|---|---|
| `source`, `segment_id`, `source_row`, `pos_in_seg` | 행 식별 key (윈도우 판정 시점 = 해당 행) |
| `label` | 실제 상태 (0 정상, 1 이상) |
| `A_prob` | 모델 A의 보정된 이상 확률 (0~1) |
| `B_normal_pct` | 모델 B 점수의 정상 데이터 대비 백분위 (100에 가까울수록 정상에서 멂) |
| `A_alarm`, `B_alarm` | 각 모델 임계값 초과 여부 |
| `alarm_level` | 정상 / 주의 / 경보 |
| `final_alarm` | 최종 경보 (1 = 경보) |

## 8. 한계

- 이상 데이터가 하루(7/17) 약 3분, 21개 segment에서 나온 단일 사건이고 정상(7/12)과 수집일이 다르다. 높은 점수가 이상 징후 때문인지 수집조건 차이 때문인지 완전히 분리할 수 없다.
- permutation importance에서 전류 크기·전류 파형형태 묶음의 기여가 가장 크다. 전류 시간구조(자기상관 등)는 수집조건 차이 가능성이 있어 최종 모델에서 제외했다.
- test 이상 segment가 4개뿐이라 test 점수 0.97~1.00 사이 차이는 통계적으로 의미가 작다. 모델 선정은 train 교차검증 기준으로 했다.
- AI2의 약 1.7초 반복 주기는 10Hz 샘플링에 따른 교류 전류 에일리어싱일 가능성이 있어 공정 cycle로 단정하지 않는다.

## 9. 기타 스크립트

레포 최상단의 `00_~07_*.py`, `05_EDA/05_EDA_Sensor2~4`, `05_EDA/05_Physics_Informed`, `05_EDA/05_Cycle/05.EDA_0~3`,
`07_Feature_Engineering/이전실험/`은 탐색 분석 기록이며 위 파이프라인 실행에는 필요하지 않다.
모두 레포 안 어느 위치에서 실행해도 `data/`의 원본 CSV를 찾아 읽고, 결과는 각 스크립트 폴더에 저장한다.
