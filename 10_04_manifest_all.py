import pandas as pd

print('====================================================================')
print('       [2번 과제: Normal + Outlier 전체 통합 매니페스트 생성]')
print('====================================================================')

# 1. 통합 Feature Table 로드 (앞서 1번에서 만든 전체 테이블)
df = pd.read_csv('10_01_final_eda_table.csv')

# 2. 정상 데이터(Normal) 기준 RMS-IQR Threshold 설정
normal_df = df[df['label'] == 'Normal']
rms_cols = ['AI0_Vibration_RMS', 'AI1_Vibration_RMS', 'AI2_Current_RMS']

thresholds = {}
for col in rms_cols:
  q1 = normal_df[col].quantile(0.25)
  q3 = normal_df[col].quantile(0.75)
  iqr = q3 - q1
  # [오탈자 수정 완료] 곱셈 기호 추가 및 올바른 수식 반영
  thresholds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# 3. 채널별 RMS 검출 여부 확인 (전체 데이터 대상)
detections = {}
for col in rms_cols:
  lower, upper = thresholds[col]
  detections[col] = (df[col] < lower) | (df[col] > upper)

# 어느 한 채널이라도 범위를 벗어났으면 RMS 검출로 판정
df['RMS_Detected'] = (
    detections['AI0_Vibration_RMS']
    | detections['AI1_Vibration_RMS']
    | detections['AI2_Current_RMS']
)

# 4. 3가지 카테고리로 최종 라벨링 함수 정의
def classify_status(row):
  # min_periods=10에 의해 결측치가 발생했거나 평가 불가 조건인 경우
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

df['Outlier_Category'] = df.apply(classify_status, axis=1)

# 5. 정통맨 결과와 Merge하기 위한 Primary Key 및 핵심 지표 컬럼 정리 (source 포함)
final_manifest = df[
    [
        'source',              # normal, abnormal(outlier) 모두 포함
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

final_manifest = final_manifest.sort_values(
    by=['source', 'source_row'],
    ascending=[True, True]
    )


# 6. 최종 매니페스트 CSV 저장 (기존 10_02 파일에 덮어쓰기)
final_manifest.to_csv('10_04_manifest_all.csv', index=False)

# 7. 결과 카운트 출력
print('\n[전체 데이터 카테고리별 분류 현황]')
print(final_manifest['Outlier_Category'].value_counts())
print('\n[Source별 데이터 분포]')
print(final_manifest['source'].value_counts())
print('--------------------------------------------------------------------')
print('-> [완료] 전체 데이터 통합 "10_04_manifest_all.csv" 덮어쓰기 완료!')
print('====================================================================')