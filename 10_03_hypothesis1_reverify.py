import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

print(
    '===================================================================='
)
print('          [3번 과제: H1 Current-Vibration 관계 재검증]')
print(
    '===================================================================='
)

# 1. 통합 Feature Table 및 매니페스트 로드
df = pd.read_csv('final_sensor_eda_table.csv')
manifest = pd.read_csv('outlier_final_manifest.csv')

# 2. 데이터 분할 (Normal vs Outlier Categories)
normal_df = df[df['label'] == 'Normal']
outlier_df = df[df['label'] == 'Outlier'].copy()

# 매니페스트의 분류 카테고리 병합
outlier_df = outlier_df.merge(
    manifest[['segment_id', 'source_row', 'Outlier_Category']],
    on=['segment_id', 'source_row'],
    how='left',
)

# 3. 정상 데이터 기준 AI2(Current) -> AI0, AI1(Vibration) 선형회귀 모델 학습
X_norm = normal_df[['AI2_Current']].values
y_norm_ai0 = normal_df['AI0_Vibration'].values
y_norm_ai1 = normal_df['AI1_Vibration'].values

lr_ai0 = LinearRegression().fit(X_norm, y_norm_ai0)
lr_ai1 = LinearRegression().fit(X_norm, y_norm_ai1)


# 4. 잔차(Residual) 산출 함수
def calculate_group_residuals(target_df, model_ai0, model_ai1):
  X = target_df[['AI2_Current']].values
  pred_ai0 = model_ai0.predict(X)
  pred_ai1 = model_ai1.predict(X)

  res_ai0 = np.abs(target_df['AI0_Vibration'] - pred_ai0)
  res_ai1 = np.abs(target_df['AI1_Vibration'] - pred_ai1)
  return res_ai0, res_ai1


# 그룹별 잔차 계산
normal_res_ai0, normal_res_ai1 = calculate_group_residuals(
    normal_df, lr_ai0, lr_ai1
)

missed_df = outlier_df[outlier_df['Outlier_Category'] == 'RMS 미검출']
missed_res_ai0, missed_res_ai1 = calculate_group_residuals(
    missed_df, lr_ai0, lr_ai1
)

detected_df = outlier_df[outlier_df['Outlier_Category'] == 'RMS 검출']
detected_res_ai0, detected_res_ai1 = calculate_group_residuals(
    detected_df, lr_ai0, lr_ai1
)

# 정상 대비 배수(Fold Change) 계산
fc_det_ai0 = detected_res_ai0.mean() / normal_res_ai0.mean()
fc_miss_ai0 = missed_res_ai0.mean() / normal_res_ai0.mean()

fc_det_ai1 = detected_res_ai1.mean() / normal_res_ai1.mean()
fc_miss_ai1 = missed_res_ai1.mean() / normal_res_ai1.mean()

print(
    '===================================================================='
)
print('          [3번 과제: H1 Current-Vibration 관계 재검증]')
print(
    '===================================================================='
)
print('\n[H1 잔차 분석 최종 결과 및 깨짐 정도(배수) 요약]')
print(
    '--------------------------------------------------------------------'
)
print('▶ [AI2 전류 → AI0 진동 잔차 분석]')
print(f'  - 정상(Normal) 그룹 잔차 평균     : {normal_res_ai0.mean():.4f}')
print(
    f'  - RMS 검출 이상 그룹 잔차 평균    : {detected_res_ai0.mean():.4f}'
    f' (정상 대비 {fc_det_ai0:.1f}배 폭등)'
)
print(
    f'  - RMS 미검출 이상 그룹 잔차 평균  : {missed_res_ai0.mean():.4f}'
    f' (정상 대비 {fc_miss_ai0:.1f}배 수준)'
)
print(
    '--------------------------------------------------------------------'
)
print('▶ [AI2 전류 → AI1 진동 잔차 분석]')
print(f'  - 정상(Normal) 그룹 잔차 평균     : {normal_res_ai1.mean():.4f}')
print(
    f'  - RMS 검출 이상 그룹 잔차 평균    : {detected_res_ai1.mean():.4f}'
    f' (정상 대비 {fc_det_ai1:.1f}배 증가)'
)
print(
    f'  - RMS 미검출 이상 그룹 잔차 평균  : {missed_res_ai1.mean():.4f}'
    f' (정상 대비 {fc_miss_ai1:.1f}배 뚜렷한 관계 이탈!)'
)
print(
    '===================================================================='
)