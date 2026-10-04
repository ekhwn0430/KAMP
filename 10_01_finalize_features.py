import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)

print('====================================================================')
print('        [공식 전처리 기준: RMS / H1 / H2 / H3 지표 확정]')
print('====================================================================')

# 1. CSV Load
ROOT = Path('.')
data_path = ROOT / 'data' / 'processed' / 'preprocessed_data.csv'

data = pd.read_csv('preprocessed_data.csv')

normal_proc = data[data['source'] == 'normal'].copy()
outlier_proc = data[data['source'] == 'abnormal'].copy()

cols = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
window_size = 10


# 2. Rolling RMS 연산 함수 (공식 전처리 기준 적용)
def calculate_segment_rms(df, cols, window=10):
  rms_list = []
  for seg_id, group in df.groupby('segment_id'):
    g_rms = (
        group[cols].pow(2).rolling(window=window, min_periods=10).mean().pow(0.5)
    )

    g_rms['TimeStamp'] = group['TimeStamp']
    g_rms['elapsed_sec'] = group['elapsed_sec']
    g_rms['dt_sec'] = group['dt_sec']
    g_rms['segment_id'] = seg_id
    g_rms['pos_in_seg'] = group['pos_in_seg']
    g_rms['source_row'] = group['source_row']
    g_rms['source'] = group['source']

    g_rms.rename(
        columns={
            'AI0_Vibration': 'AI0_Vibration_RMS',
            'AI1_Vibration': 'AI1_Vibration_RMS',
            'AI2_Current': 'AI2_Current_RMS',
        },
        inplace=True,
    )
    
    for c in cols:
      g_rms[f'{c}_Raw'] = group[c].values
      
    rms_list.append(g_rms)
  return pd.concat(rms_list).sort_index()


print(
    '-> 공식 전처리  기반 정상 및 이상 데이터 Rolling RMS 연산 중...'
    ' (min_periods=10)...')
normal_rms = calculate_segment_rms(normal_proc, cols, window=window_size)
outlier_rms = calculate_segment_rms(outlier_proc, cols, window=window_size)

# 4. 정상 데이터 기준 회귀 모델 학습 (H1 및 H2 산출을 위한 기준 선형 모형)
# AI2(Current) -> AI0, AI1 진동 회귀 학습 (H1용)
valid_normal_rms = normal_rms.dropna(
    subset=['AI2_Current_RMS', 'AI0_Vibration_RMS', 'AI1_Vibration_RMS']
)

lr_ai0 = LinearRegression().fit(
    valid_normal_rms[['AI2_Current_RMS']], valid_normal_rms['AI0_Vibration_RMS']
)
lr_ai1 = LinearRegression().fit(
    valid_normal_rms[['AI2_Current_RMS']], valid_normal_rms['AI1_Vibration_RMS']
)


# 5. [RMS / H1 / H2 / H3 통합 지표 생성 함수]
def apply_all_features(rms_df, lr_ai0, lr_ai1):
  df = rms_df.copy()
  
  temp_X = df[['AI2_Current_RMS']].fillna(0)

  # [RMS 지표 확정] 이미 계산된 채널별 Rolling RMS 활용

  # [H1 지표 확정]: 전류-진동 관계 및 잔차 (Residual)
  pred_ai0 = lr_ai0.predict(temp_X.values)
  pred_ai1 = lr_ai1.predict(temp_X.values)
  
  df['H1_Residual_AI0'] = np.abs(df['AI0_Vibration_RMS'] - pred_ai0)
  df['H1_Residual_AI1'] = np.abs(df['AI1_Vibration_RMS'] - pred_ai1)

  # [H2 지표 확정]: 진동 채널 간 관계 (Diff, Ratio, Normalized Difference)
  eps = 1e-6
  df['H2_Vib_Diff'] = df['AI0_Vibration_RMS'] - df['AI1_Vibration_RMS']
  df['H2_Vib_Ratio'] = df['AI0_Vibration_RMS'] / (
      df['AI1_Vibration_RMS'].abs() + eps
  )
  df['H2_ND'] = (df['AI0_Vibration_RMS'] - df['AI1_Vibration_RMS']) / (
      df['AI0_Vibration_RMS'] + df['AI1_Vibration_RMS'] + eps
  )

  # [H3 지표 확정]: 전류 Amplitude 및 파생 특성
  df['H3_Current_Amplitude'] = df['AI2_Current_RMS']

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
    '10_01_final_eda_table.csv', index=False
)

print('====================================================================')
print(f'-> [완료] 최종 Sensor EDA 통합 테이블 생성 완료!')
print(f'-> 총 데이터 행 수: {len(final_integrated_table)}개')
print(f'-> 저장 파일명: "10_01_final_eda_table.csv"')
print('====================================================================')