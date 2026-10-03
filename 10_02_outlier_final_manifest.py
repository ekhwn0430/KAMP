import pandas as pd

print('====================================================================')
print('       [2번 과제: Outlier 600개 최종 분류 및 매니페스트 생성]')
print('====================================================================')

# 1. 통합 Feature Table 로드
df = pd.read_csv('final_sensor_eda_table.csv')

# 2. Outlier 데이터만 필터링 (label이 'Outlier'인 행)
outlier_df = df[df['label'] == 'Outlier'].copy()

# 3. 정상 데이터(Normal) 기준 IQR Threshold 재확인
normal_df = df[df['label'] == 'Normal']
cols = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']

thresholds = {}
for col in cols:
  q1 = normal_df[col].quantile(0.25)
  q3 = normal_df[col].quantile(0.75)
  iqr = q3 - q1
  thresholds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# 4. 채널별 RMS 검출 여부 확인
detections = {}
for col in cols:
  lower, upper = thresholds[col]
  detections[col] = (outlier_df[col] < lower) | (outlier_df[col] > upper)

# 어느 한 채널이라도 범위를 벗어났으면 RMS 검출로 판정
outlier_df['RMS_Detected'] = (
    detections['AI0_Vibration']
    | detections['AI1_Vibration']
    | detections['AI2_Current']
)


# 5. 3가지 카테고리로 최종 라벨링 함수 정의
def classify_outlier_status(row):
  # 결측치나 평가 불가능한 예외 조건
  if (
      pd.isna(row['AI0_Vibration'])
      or pd.isna(row['AI1_Vibration'])
      or pd.isna(row['AI2_Current'])
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
        'segment_id',
        'source_row',
        'Equipment_state',
        'RMS_Detected',
        'Outlier_Category',
        'H1_Residual_AI0',
        'H1_Residual_AI1',
        'H2_ND',
        'H3_Current_Amplitude',
    ]
].copy()

# 7. 최종 매니페스트 CSV 저장
outlier_manifest.to_csv('outlier_final_manifest.csv', index=False)

# 8. 결과 카운트 출력
print('\n[Outlier 600개 최종 분류 현황]')
print(outlier_manifest['Outlier_Category'].value_counts())
print('--------------------------------------------------------------------')
print('-> [완료] 정통맨 Merge용 "outlier_final_manifest.csv" 파일 저장 완료!')
print('====================================================================')