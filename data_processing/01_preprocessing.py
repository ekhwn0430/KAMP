import numpy as np
import pandas as pd

# 1. CSV Load
normal_df = pd.read_csv('press_data_normal.csv')
outlier_df = pd.read_csv('outlier_data.csv')


# 2. Timestamp 무결성 검증 함수
def validate_timestamp(df, name='Data'):
  df = df.copy()
  df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
  is_sorted = df['TimeStamp'].is_monotonic_increasing
  print(f'[{name}] 시간순 정렬 여부: {is_sorted}')

  df['time_diff'] = df['TimeStamp'].diff().dt.total_seconds().fillna(0)
  duplicates = df['TimeStamp'].duplicated().sum()
  gaps = df[df['time_diff'] > 0.15]
  print(
      f'[{name}] 중복 타임스탬프: {duplicates}개 | 0.15초 초과 공백: {len(gaps)}회'
  )
  return df


print('=============== 데이터 무결성 검증 시작 ===============')
normal_v = validate_timestamp(normal_df, 'Normal')
outlier_v = validate_timestamp(outlier_df, 'Outlier')


# 3. Segment ID 생성 및 Rolling RMS 계산 함수
def process_pipeline(df, window_size=10, time_threshold=0.15):
  df = df.copy()
  df = df.sort_values('TimeStamp').reset_index(drop=True)
  df['time_diff'] = df['TimeStamp'].diff().dt.total_seconds().fillna(0)
  df['segment_id'] = (df['time_diff'] > time_threshold).cumsum()

  cols = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
  rms_list = []

  for seg_id, group in df.groupby('segment_id'):
    g_rms = (
        group[cols].pow(2).rolling(window=window_size, min_periods=1).mean().pow(0.5)
    )
    g_rms['segment_id'] = seg_id
    g_rms['TimeStamp'] = group['TimeStamp']
    g_rms['Equipment_state'] = group['Equipment_state']
    rms_list.append(g_rms)

  return pd.concat(rms_list).sort_index()


# 각각 독립적으로 전처리 수행
print('\n============== 세그먼트 및 Rolling RMS 계산 중 ==============')
normal_rms = process_pipeline(normal_v, window_size=10)
outlier_rms = process_pipeline(outlier_v, window_size=10)

# 4. Feature Table 통합 (최종 머신러닝 학습용)
normal_rms['label'] = 0 #정상
outlier_rms['label'] = 1 #이상

final_feature_table = pd.concat([normal_rms, outlier_rms], ignore_index=True)

# 5. 최종 CSV 저장
final_feature_table.to_csv('01_feature_table.csv', index=False)
print(
    '전처리 완료 및 01_feature_table.csv 저장 완료 (총 행 수:'
    f' {len(final_feature_table)}개)'
)