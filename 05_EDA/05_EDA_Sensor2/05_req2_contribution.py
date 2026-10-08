# --- 실행 위치와 관계없이 동작: 결과는 이 스크립트 폴더에 저장, 원본 CSV는 레포 최상단 data/에서 읽음 ---
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parent)
DATA_DIR = next(_p / "data" for _p in [_Path.cwd(), *_Path.cwd().parents]
                if (_p / "data" / "press_data_normal.csv").exists())
# ---------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

print("데이터 로드 및 전처리 중...")
# 1. CSV Load
normal_df = pd.read_csv(DATA_DIR / 'press_data_normal.csv')
outlier_df = pd.read_csv(DATA_DIR / 'outlier_data.csv')

# 2. TimeStamp 파싱 및 Segment ID 생성 함수
def preprocess_and_segment(df, time_threshold=0.15):
    df = df.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df.sort_values('TimeStamp').reset_index(drop=True)
    df['time_diff'] = df['TimeStamp'].diff().dt.total_seconds().fillna(0)
    df['segment_id'] = (df['time_diff'] > time_threshold).cumsum()
    return df

normal_proc = preprocess_and_segment(normal_df)
outlier_proc = preprocess_and_segment(outlier_df)

# 3. Segment 내부 Rolling RMS 계산
cols = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
window_size = 10

def calculate_segment_rms(df, cols, window=10):
    rms_list = []
    for seg_id, group in df.groupby('segment_id'):
        g_rms = group[cols].pow(2).rolling(window=window, min_periods=1).mean().pow(0.5)
        g_rms['segment_id'] = seg_id
        g_rms['TimeStamp'] = group['TimeStamp']
        g_rms['Equipment_state'] = group['Equipment_state']
        rms_list.append(g_rms)
    return pd.concat(rms_list).sort_index()

normal_rms = calculate_segment_rms(normal_proc, cols, window=window_size)
outlier_rms = calculate_segment_rms(outlier_proc, cols, window=window_size)

# 4. IQR Threshold 계산 (정상 데이터 기준) 및 이상치 탐지
thresholds = {}
for col in cols:
    q1 = normal_rms[col].quantile(0.25)
    q3 = normal_rms[col].quantile(0.75)
    iqr = q3 - q1
    thresholds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

detections = {}
for col in cols:
    lower, upper = thresholds[col]
    detections[col] = (outlier_rms[col] < lower) | (outlier_rms[col] > upper)

det_ai0 = detections['AI0_Vibration']
det_ai1 = detections['AI1_Vibration']
det_ai2 = detections['AI2_Current']

# 5. 채널별 단독 및 세부 조합 검출 분석
total_outliers = len(outlier_rms)
combined_detected = det_ai0 | det_ai1 | det_ai2
undetected = ~combined_detected

solo_ai0 = det_ai0 & (~det_ai1) & (~det_ai2)
solo_ai1 = det_ai1 & (~det_ai0) & (~det_ai2)
solo_ai2 = det_ai2 & (~det_ai0) & (~det_ai1)

comb_ai0_ai1 = det_ai0 & det_ai1 & (~det_ai2)
comb_ai0_ai2 = det_ai0 & det_ai2 & (~det_ai1)
comb_ai1_ai2 = det_ai1 & det_ai2 & (~det_ai0)
overlap_all = det_ai0 & det_ai1 & det_ai2

print("\n==========================================")
print("  [RMS 이상탐지 기여도 및 세부 조합 분석]")
print("==========================================")
print(f"총 이상 데이터 수        : {total_outliers}개")
print(f"RMS 검출된 데이터 수     : {combined_detected.sum()}개")
print(f"RMS 미검출 데이터 수     : {undetected.sum()}개")
print("------------------------------------------")
print(f"• AI0 단독 검출          : {solo_ai0.sum()}개")
print(f"• AI1 단독 검출          : {solo_ai1.sum()}개")
print(f"• AI2 단독 검출          : {solo_ai2.sum()}개")
print("------------------------------------------")
print(f"• AI0 + AI1 동시 검출 (AI2는 정상) : {comb_ai0_ai1.sum()}개")
print(f"• AI0 + AI2 동시 검출 (AI1은 정상) : {comb_ai0_ai2.sum()}개")
print(f"• AI1 + AI2 동시 검출 (AI0은 정상) : {comb_ai1_ai2.sum()}개")
print(f"• AI0 + AI1 + AI2 전부 동시 검출   : {overlap_all.sum()}개")
print("==========================================")
print(f"[참고] 개별 채널 임계값 초과 총 횟수:")
print(f"• AI0 총 초과: {det_ai0.sum()}개")
print(f"• AI1 총 초과: {det_ai1.sum()}개")
print(f"• AI2 총 초과: {det_ai2.sum()}개")
print("==========================================")