"""
전처리 → Cycle 분석 → Feature → 모델 비교 → 오류분석 → 최종 경보 규칙까지 한 번에 실행한다.

실행:  python run_all.py            (레포 최상단에서)
       python run_all.py --from 08  (08 단계부터 다시 실행)

순서 메모
  07_01을 05_Cycle보다 먼저 실행한다. 05_EDA_4_Outlier_Cycle.py가 07_01의 cycle 표를 읽어
  참고 컬럼(cycle_id_07)을 채우기 때문이다. 07_01은 전처리 데이터만 사용한다.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("04", "04_Data_Processing/preprocess.py", "원본 CSV 정제, segment 분할 → data/processed/preprocessed_data.csv"),
    ("07", "07_Feature_Engineering/07_01_Feature.py", "AI2 cycle 단위 feature 표"),
    ("05", "05_EDA/05_Cycle/05_EDA_4_Outlier_Cycle.py", "Normal 기준 고정 AI2 Cycle 분석 (Abnormal 적용)"),
    ("05", "05_EDA/05_Cycle/05_EDA_5_Merge_Cycle.py", "cycle_rows + cycle_units merge"),
    ("07", "07_Feature_Engineering/07_02_Feature_validation.py", "cycle feature 검증"),
    ("07", "07_Feature_Engineering/07_03_Final_Feature_Selection.py", "최종 설명 feature 선정"),
    ("08", "08_Modeling/08_01_model_comparison.py", "1초 윈도우 모델 비교, A/B 모델 선정"),
    ("08", "08_Modeling/08_02_Model_Error_Analysis.py", "FN/FP 오류분석, A vs B 비교"),
    ("08", "08_Modeling/08_03_Final_Alarm_Policy.py", "최종 경보 규칙, 중요도, 최종 test 예측"),
    ("11", "KAMP_11_12_Independent/run_11_12.py", "11 공정해석·12 운영 의사결정 (08 결과 읽기 전용, 재학습 없음)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="04", help="이 단계 번호부터 실행 (04/05/07/08/11)")
    args = ap.parse_args()

    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    started = False
    t_all = time.time()
    for no, script, desc in STEPS:
        started = started or no == args.start
        if not started:
            continue
        print(f"\n===== [{no}] {script}\n      {desc}", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env)
        if r.returncode != 0:
            sys.exit(f"\n실패: {script} (exit {r.returncode})")
        print(f"      완료 {time.time() - t:.0f}초", flush=True)
    print(f"\n전체 완료 {time.time() - t_all:.0f}초")
    print("최종 결과: 08_Modeling/results/final/ (test_predictions_final.csv 등), 11_12_results/11, 11_12_results/12")


if __name__ == "__main__":
    main()
