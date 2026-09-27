import pandas as pd
import numpy as np

print("1. 데이터 로드 및 전처리 중...")
normal_df = pd.read_csv('press_data_normal.csv')
outlier_df = pd.read_csv('outlier_data.csv')

def preprocess_and_segment(df, time_threshold=0.15):
    df = df.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df.sort_values('TimeStamp').reset_index(drop=True)
    df['time_diff'] = df['TimeStamp'].diff().dt.total_seconds().fillna(0)
    df['segment_id'] = (df['time_diff'] > time_threshold).cumsum()
    return df

normal_proc = preprocess_and_segment(normal_df)
outlier_proc = preprocess_and_segment(outlier_df)

print(f"-> 정상 세그먼트 수: {normal_proc['segment_id'].nunique()}개")
print(f"-> 이상 세그먼트 수: {outlier_proc['segment_id'].nunique()}개")

cols = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
window_size = 10 #5, 10, 20

def calculate_segment_rms(df, cols, window=10):
    rms_list = []
    for seg_id, group in df.groupby('segment_id'):
        g_rms = group[cols].pow(2).rolling(window=window, min_periods=1).mean().pow(0.5)
        g_rms['segment_id'] = seg_id
        g_rms['TimeStamp'] = group['TimeStamp']
        g_rms['Equipment_state'] = group['Equipment_state']
        rms_list.append(g_rms)
    return pd.concat(rms_list).sort_index()

print("\n2. Segment별 Rolling RMS 계산 중...")
normal_rms = calculate_segment_rms(normal_proc, cols, window=window_size)
outlier_rms = calculate_segment_rms(outlier_proc, cols, window=window_size)

# IQR Threshold 탐지 및 미검출 259개 분리
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

print(f"\n3. RMS 미검출 데이터 개수: {len(undetected_rms)}개 <분리 완료>")

print("\n4. 가설[2] 검증: 두 진동 채널 간 관계 피처(차이 및 비율) 분석 중...")

# 가설 2 피처 생성 및 비교
def add_channel_relation_features(df):
    df = df.copy()
    # 두 진동 채널의 차이 (Diff)
    df['Vib_Diff'] = df['AI0_Vibration'] - df['AI1_Vibration']
    # 두 진동 채널의 비율 (Ratio)
    df['Vib_Ratio'] = df['AI0_Vibration'] / (df['AI1_Vibration'].abs() + 1e-6)
    return df

normal_h2 = add_channel_relation_features(normal_rms)
undetected_h2 = add_channel_relation_features(undetected_rms)

print("\n=================================================================")
print("================ [가설 2: 채널 간 관계 분석 결과] ===============")
print("=================================================================")
print(f"정상 데이터 채널 차이(Diff) 평균     : {normal_h2['Vib_Diff'].mean():.4f} (표준편차: {normal_h2['Vib_Diff'].std():.4f})")
print(f"미검출 이상 데이터 차이(Diff) 평균   : {undetected_h2['Vib_Diff'].mean():.4f} (표준편차: {undetected_h2['Vib_Diff'].std():.4f})")
print("-----------------------------------------------------------------")
print(f"정상 데이터 채널 비율(Ratio) 평균    : {normal_h2['Vib_Ratio'].mean():.4f} (표준편차: {normal_h2['Vib_Ratio'].std():.4f})")
print(f"미검출 이상 데이터 비율(Ratio) 평균  : {undetected_h2['Vib_Ratio'].mean():.4f} (표준편차: {undetected_h2['Vib_Ratio'].std():.4f})")
print("=================================================================")

# 채널 간 상관계수 비교
normal_corr = normal_rms['AI0_Vibration'].corr(normal_rms['AI1_Vibration'])
undetected_corr = undetected_rms['AI0_Vibration'].corr(undetected_rms['AI1_Vibration'])
print(f"정상 데이터 채널 간 상관계수 (Corr)     : {normal_corr:.4f}")
print(f"미검출 이상 데이터 채널 간 상관계수 (Corr): {undetected_corr:.4f}")
print("=================================================================")