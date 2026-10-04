import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

print("1. 데이터 로드 및 전처리 중...")
# 1. CSV Load
normal_df = pd.read_csv('press_data_normal.csv')
outlier_df = pd.read_csv('outlier_data.csv')

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

print(f"-> 정상 세그먼트 수: {normal_proc['segment_id'].nunique()}개")
print(f"-> 이상 세그먼트 수: {outlier_proc['segment_id'].nunique()}개")

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

print("\n2. Segment별 Rolling RMS 계산 중...")
normal_rms = calculate_segment_rms(normal_proc, cols, window=window_size)
outlier_rms = calculate_segment_rms(outlier_proc, cols, window=window_size)

# 4. IQR Threshold 생성 및 이상치 탐지 (RMS 기준)
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

# 5. RMS로 탐지된 항목과 미검출 항목 분리
combined_detected = detections['AI0_Vibration'] | detections['AI1_Vibration'] | detections['AI2_Current']
outlier_rms['RMS_Detected'] = combined_detected

undetected_rms = outlier_rms[~outlier_rms['RMS_Detected']].copy()
print(f"\n3. RMS 미검출 데이터 개수: {len(undetected_rms)}개 <분리 완료>")

print("\n4. 가설[1] 검증: 전류-진동 회귀 잔차 분석 중...")
# 6. 정상 데이터 기준 전류(AI2) vs 진동(AI0) 회귀 모델 학습
X_normal = normal_rms[['AI2_Current']].values
y_normal = normal_rms['AI0_Vibration'].values

lr = LinearRegression()
lr.fit(X_normal, y_normal)

# 7. 잔차(실제 진동 - 예측 진동) 계산 함수
def calculate_residuals(rms_df, model):
    df = rms_df.copy()
    X = df[['AI2_Current']].values
    y = df['AI0_Vibration'].values
    pred = model.predict(X)
    df['Residual'] = np.abs(y - pred)
    return df

normal_res = calculate_residuals(normal_rms, lr)
undetected_res = calculate_residuals(undetected_rms, lr)

print("\n========================================")
print("[가설 1: 전류-진동 회귀 잔차 분석 결과]")
print("========================================")
print(f"정상 데이터 잔차 평균         : {normal_res['Residual'].mean():.4f}")
print(f"미검출 이상 데이터 잔차 평균  : {undetected_res['Residual'].mean():.4f}")
print("========================================")