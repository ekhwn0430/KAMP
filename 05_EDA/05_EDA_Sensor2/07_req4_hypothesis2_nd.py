# --- 실행 위치와 관계없이 동작: 결과는 이 스크립트 폴더에 저장, 원본 CSV는 레포 최상단 data/에서 읽음 ---
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parent)
DATA_DIR = next(_p / "data" for _p in [_Path.cwd(), *_Path.cwd().parents]
                if (_p / "data" / "press_data_normal.csv").exists())
# ---------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 폰트 및 스타일 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

print("======================================================================")
print("                     [가설 2 보강 : ND 지표 분석]")
print("======================================================================")

# 2. 데이터 로드 및 전처리
normal_df = pd.read_csv(DATA_DIR / 'press_data_normal.csv')
outlier_df = pd.read_csv(DATA_DIR / 'outlier_data.csv')

def preprocess_and_segment(df, time_threshold=0.15):
    df = df.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df.sort_values('TimeStamp').reset_index(drop=True)
    df['time_diff'] = df['TimeStamp'].diff().dt.total_seconds().fillna(0)
    df['segment_id'] = (df['time_diff'] > time_threshold).cumsum()
    return df

normal_proc = preprocess_and_segment(normal_df)
outlier_proc = preprocess_and_segment(outlier_df)

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

# 3. IQR Threshold 기반 미검출 이상 데이터(259개) 분리
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

combined_detected = detections['AI0_Vibration'] | detections['AI1_Vibration'] | detections['AI2_Current']
outlier_rms['RMS_Detected'] = combined_detected
undetected_rms = outlier_rms[~outlier_rms['RMS_Detected']].copy()

# 4. Normalized Difference (ND) 지표 계산 함수
def calculate_nd(df, eps=1e-6):
    df = df.copy()
    df['ND'] = (df['AI0_Vibration'] - df['AI1_Vibration']) / (df['AI0_Vibration'] + df['AI1_Vibration'] + eps)
    return df

normal_nd = calculate_nd(normal_rms)
undetected_nd = calculate_nd(undetected_rms)

# 5. 통계량 출력
print("\n[Normalized Difference (ND) 분석 결과]")
print(f"• 정상 데이터 ND 평균       : {normal_nd['ND'].mean():.4f} (표준편차: {normal_nd['ND'].std():.4f})")
print(f"• 미검출 이상 데이터 ND 평균 : {undetected_nd['ND'].mean():.4f} (표준편차: {undetected_nd['ND'].std():.4f})")

# 6. ND 분포 비교 시각화 (KDE Plot) 및 저장
plt.figure(figsize=(10, 6))
sns.kdeplot(normal_nd['ND'], label='Normal', color='blue', fill=True, alpha=0.3)
sns.kdeplot(undetected_nd['ND'], label='Undetected Outlier', color='orange', fill=True, alpha=0.3)
plt.title('Normalized Difference (ND) Distribution Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Normalized Difference (ND)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('07_req4_nd_distribution.png', dpi=300)
plt.close()

print("\n-> [완료] ND 분포 밀도 그래프 '07_req4_nd_distribution.png' 저장 완료!")
print("======================================================================")