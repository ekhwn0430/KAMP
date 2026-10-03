import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

print('====================================================================')
print('        [공식 전처리 기준: RMS / H1 / H2 / H3 지표 최종 확정]')
print('====================================================================')

# 1. CSV Load
normal_df = pd.read_csv('press_data_normal.csv')
outlier_df = pd.read_csv('outlier_data.csv')


# 2. [공식 전처리 기준] TimeStamp 무결성 보장 및 Segment ID 생성 (Legacy 제거)
def preprocess_and_segment(df, time_threshold=0.15):
  df = df.copy()
  df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], errors='coerce')
  df = df.sort_values('TimeStamp').reset_index(drop=True)
  # 절대 시간 오차 배제를 위한 누적 경과 시간 및 Segment ID 생성
  df['elapsed_sec'] = (
      df['TimeStamp'] - df['TimeStamp'].iloc[0]
  ).dt.total_seconds().fillna(0)
  df['time_diff'] = df['TimeStamp'].diff().dt.total_seconds().fillna(0)
  df['segment_id'] = (df['time_diff'] > time_threshold).cumsum()
  return df


normal_proc = preprocess_and_segment(normal_df)
outlier_proc = preprocess_and_segment(outlier_df)

cols = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
window_size = 10


# 3. Rolling RMS 연산 함수 (공식 전처리 기준 적용)
def calculate_segment_rms(df, cols, window=10):
  rms_list = []
  for seg_id, group in df.groupby('segment_id'):
    g_rms = (
        group[cols].pow(2).rolling(window=window, min_periods=1).mean().pow(0.5)
    )
    g_rms['segment_id'] = seg_id
    g_rms['source_row'] = group.index  # 정통맨 결과와 융합할 수 있는 row 기준 키
    g_rms['Equipment_state'] = group['Equipment_state']
    for c in cols:
      g_rms[f'{c}_Raw'] = group[c].values
    rms_list.append(g_rms)
  return pd.concat(rms_list).sort_index()


print('-> 정상 및 이상 데이터 Rolling RMS 연산 중...')
normal_rms = calculate_segment_rms(normal_proc, cols, window=window_size)
outlier_rms = calculate_segment_rms(outlier_proc, cols, window=window_size)

# 4. 정상 데이터 기준 회귀 모델 학습 (H1 및 H2 산출을 위한 기준 선형 모형)
# AI2(Current) -> AI0, AI1 진동 회귀 학습 (H1용)
lr_ai0 = LinearRegression().fit(
    normal_rms[['AI2_Current']], normal_rms['AI0_Vibration']
)
lr_ai1 = LinearRegression().fit(
    normal_rms[['AI2_Current']], normal_rms['AI1_Vibration']
)


# 5. [RMS / H1 / H2 / H3 통합 지표 생성 함수]
def apply_all_features(rms_df, lr_ai0, lr_ai1):
  df = rms_df.copy()

  # [RMS 지표 확정] 이미 계산된 채널별 Rolling RMS 활용

  # [H1 지표 확정]: 전류-진동 관계 및 잔차 (Residual)
  pred_ai0 = lr_ai0.predict(df[['AI2_Current']].values)
  pred_ai1 = lr_ai1.predict(df[['AI2_Current']].values)
  df['H1_Residual_AI0'] = np.abs(df['AI0_Vibration'] - pred_ai0)
  df['H1_Residual_AI1'] = np.abs(df['AI1_Vibration'] - pred_ai1)

  # [H2 지표 확정]: 진동 채널 간 관계 (Diff, Ratio, Normalized Difference)
  eps = 1e-6
  df['H2_Vib_Diff'] = df['AI0_Vibration'] - df['AI1_Vibration']
  df['H2_Vib_Ratio'] = df['AI0_Vibration'] / (
      df['AI1_Vibration'].abs() + eps
  )
  df['H2_ND'] = (df['AI0_Vibration'] - df['AI1_Vibration']) / (
      df['AI0_Vibration'] + df['AI1_Vibration'] + eps
  )

  # [H3 지표 확정]: 전류 Amplitude 및 파생 특성
  df['H3_Current_Amplitude'] = df['AI2_Current']

  return df


print('-> 공식 전라인 기준 H1, H2, H3 고조파 및 관계 지표 통합 중...')
normal_final = apply_all_features(normal_rms, lr_ai0, lr_ai1)
normal_final['label'] = 'Normal'

outlier_final = apply_all_features(outlier_rms, lr_ai0, lr_ai1)
outlier_final['label'] = 'Outlier'

# 6. 최종 통합 Feature Table 저장
final_integrated_table = pd.concat(
    [normal_final, outlier_final], ignore_index=True
)
final_integrated_table.to_csv(
    'final_sensor_eda_table.csv', index=False
)

print('====================================================================')
print(f'-> [완료] 최종 Sensor EDA 통합 테이블 생성 완료!')
print(f'-> 총 데이터 행 수: {len(final_integrated_table)}개')
print(f'-> 저장 파일명: "final_sensor_eda_table.csv"')
print('====================================================================')