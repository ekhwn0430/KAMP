import pandas as pd

print('====================================================================')
print('       [2번 과제: Outlier 600개 최종 분류 및 매니페스트 생성]')
print('====================================================================')

# 1. 통합 Feature Table 로드 (앞서 1번에서 만든 파일)
df = pd.read_csv('10_01_final_eda_table.csv')

# 2. Outlier 데이터만 필터링 (label이 'Outlier'인 행)
outlier_df = df[df['label'] == 'Outlier'].copy()

# 3. 정상 데이터(Normal) 기준 RMS-IQR Threshold 설정
# [수정] 피드백 반영: Raw가 아닌 RMS 컬럼 기준으로 IQR 임계값 계산
normal_df = df[df['label'] == 'Normal']
rms_cols = ['AI0_Vibration_RMS', 'AI1_Vibration_RMS', 'AI2_Current_RMS']

thresholds = {}
for col in rms_cols:
  q1 = normal_df[col].quantile(0.25)
  q3 = normal_df[col].quantile(0.75)
  iqr = q3 - q1
  thresholds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# 4. 채널별 RMS 검출 여부 확인
detections = {}
for col in rms_cols:
  lower, upper = thresholds[col]
  detections[col] = (outlier_df[col] < lower) | (outlier_df[col] > upper)

# 어느 한 채널이라도 범위를 벗어났으면 RMS 검출로 판정
outlier_df['RMS_Detected'] = (
    detections['AI0_Vibration_RMS']
    | detections['AI1_Vibration_RMS']
    | detections['AI2_Current_RMS']
)


# 5. 3가지 카테고리로 최종 라벨링 함수 정의 (피드백 반영)
def classify_outlier_status(row):
  # [수정] 피드백 반영: min_periods=10에 의해 결측치가 발생했거나 평가 불가 조건인 경우
  if (
      pd.isna(row['AI0_Vibration_RMS'])
      or pd.isna(row['AI1_Vibration_RMS'])
      or pd.isna(row['AI2_Current_RMS'])
  ):
    return 'not_evaluable'
  # RMS 임계값 초과 여부 확인
  elif row['RMS_Detected']:
    return 'RMS 검출'
  else:
    return 'RMS 미검출'


outlier_df['Outlier_Category'] = outlier_df.apply(
    classify_outlier_status, axis=1
)

# 6. 정통맨 결과와 Merge하기 위한 Primary Key 및 핵심 지표 컬럼 정리
outlier_manifest = outlier_df[
    [
        'source',
        'segment_id',
        'source_row',
        'RMS_Detected',
        'Outlier_Category',
        'H1_Residual_AI0',
        'H1_Residual_AI1',
        'H2_ND',
        'H3_Current_Amplitude',
    ]
].copy()

# 7. 최종 매니페스트 CSV 저장
outlier_manifest.to_csv('10_02_outlier_final_manifest.csv', index=False)

# 8. 결과 카운트 출력
print('\n[Outlier 600개 최종 분류 현황]')
print(outlier_manifest['Outlier_Category'].value_counts())
print('--------------------------------------------------------------------')
print('-> [완료] 정통맨 Merge용 "10_02_outlier_final_manifest.csv" 파일 저장 완료!')
print('====================================================================')