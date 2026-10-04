import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore', category=UserWarning)

print('====================================================================')
print('   [3번 과제: H1 가설 재검증 - Train/Holdout 및 강건 통계량 보강]')
print('====================================================================')

# 1. 앞서 생성한 통합 Feature Table 로드
df = pd.read_csv('10_01_final_eda_table.csv')

# 평가 불가능한 샘플(not_evaluable) 제외 후 유효 데이터 필터링
valid_df = df.dropna(
    subset=['AI2_Current_RMS', 'AI0_Vibration_RMS', 'AI1_Vibration_RMS']
).copy()

# 2. Normal 데이터 Train / Holdout 분할
normal_df = valid_df[valid_df['label'] == 'Normal']
abnormal_df = valid_df[valid_df['label'] == 'Outlier']

train_normal, holdout_normal = train_test_split(
    normal_df, test_size=0.5, random_state=42
)

# 3. Train 정상 데이터로 Linear Regression 학습 (Current -> Vibration)
lr_ai0 = LinearRegression().fit(
    train_normal[['AI2_Current_RMS']], train_normal['AI0_Vibration_RMS']
)
lr_ai1 = LinearRegression().fit(
    train_normal[['AI2_Current_RMS']], train_normal['AI1_Vibration_RMS']
)


# 4. 잔차(Residual) 계산 함수 정의
def calculate_residuals(data_subset, model_ai0, model_ai1):
  temp_X = data_subset[['AI2_Current_RMS']].fillna(0)
  pred_ai0 = model_ai0.predict(temp_X.values)
  pred_ai1 = model_ai1.predict(temp_X.values)

  res_ai0 = np.abs(data_subset['AI0_Vibration_RMS'] - pred_ai0)
  res_ai1 = np.abs(data_subset['AI1_Vibration_RMS'] - pred_ai1)
  return res_ai0, res_ai1


# 각 그룹별 잔차 추출 (NaN 제거)
holdout_res_ai0, holdout_res_ai1 = calculate_residuals(
    holdout_normal, lr_ai0, lr_ai1
)
abnormal_res_ai0, abnormal_res_ai1 = calculate_residuals(
    abnormal_df, lr_ai0, lr_ai1
)

holdout_res_ai0 = holdout_res_ai0.dropna()
abnormal_res_ai0 = abnormal_res_ai0.dropna()
holdout_res_ai1 = holdout_res_ai1.dropna()
abnormal_res_ai1 = abnormal_res_ai1.dropna()


# 5. 강건 통계량 출력 함수
def print_robust_stats(name, series):
  mean_val = series.mean()
  std_val = series.std()
  median_val = series.median()
  q25 = series.quantile(0.25)
  q75 = series.quantile(0.75)
  iqr_val = q75 - q25
  q95 = series.quantile(0.95)

  print(f'[{name}]')
  print(f'  - Mean ± Std : {mean_val:.4f} ± {std_val:.4f}')
  print(
      f'  - Median (IQR) : {median_val:.4f} (IQR: {iqr_val:.4f} [Q25:'
      f' {q25:.4f}, Q75: {q75:.4f}])'
  )
  print(f'  - 95% Quantile : {q95:.4f}')


# 6. Mann-Whitney U 검정 및 효과 크기 계산 함수
def calculate_mann_whitney(normal_series, abnormal_series, name):
  stat, p_value = stats.mannwhitneyu(
      normal_series, abnormal_series, alternative='two-sided'
  )
  n1 = len(normal_series)
  n2 = len(abnormal_series)
  u_val = stat
  effect_size = 1 - (2 * u_val) / (n1 * n2)

  print(f'[{name} - Mann-Whitney U Test]')
  print(f'  - U statistic : {u_val:.1f}')
  print(f'  - p-value     : {p_value:.4e}')
  print(f'  - Effect Size : {abs(effect_size):.4f}')


print('\n================ [H1 재검증 통계 결과 요약] ================')
print('\n--- [AI0 채널 잔차 (Residual AI0)] ---')
print_robust_stats('Normal Holdout', holdout_res_ai0)
print_robust_stats('Abnormal (Outlier)', abnormal_res_ai0)
calculate_mann_whitney(holdout_res_ai0, abnormal_res_ai0, 'AI0 Residual')

print('\n--- [AI1 채널 잔차 (Residual AI1)] ---')
print_robust_stats('Normal Holdout', holdout_res_ai1)
print_robust_stats('Abnormal (Outlier)', abnormal_res_ai1)
calculate_mann_whitney(holdout_res_ai1, abnormal_res_ai1, 'AI1 Residual')
print('====================================================================')
print('-> [완료] H1 재검증 및 통계 보강 완료!')
print('====================================================================')