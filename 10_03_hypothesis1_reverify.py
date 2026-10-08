import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings('ignore', category=UserWarning)

print('====================================================================')
print('   [3번 과제: H1 가설 재검증 - Segment 단위 Train/Holdout 분할]')
print('====================================================================')

# 1. 통합 Feature Table(10_01) + 전체 매니페스트(10_04) 로드
df = pd.read_csv('10_01_final_eda_table.csv')
manifest = pd.read_csv('10_04_manifest_all.csv')

# source + source_row 가 행 고유키 → 10_04의 RMS 분류(Outlier_Category) 결합
df = df.merge(
    manifest[['source', 'source_row', 'Outlier_Category']],
    on=['source', 'source_row'],
    how='left',
    validate='one_to_one',
)

# 평가 불가능한 샘플(not_evaluable) 제외
valid_df = df[df['Outlier_Category'] != 'not_evaluable'].copy()

normal_df = valid_df[valid_df['label'] == 'Normal']
abnormal_df = valid_df[valid_df['label'] == 'Outlier']

# 2. [수정] Normal 데이터를 segment_id 단위로 Train / Holdout 분할
#    Rolling RMS(window=10)는 이웃 행끼리 원본 값을 공유하므로,
#    행 단위로 섞으면 같은 구간이 Train과 Holdout에 동시에 들어가 정보가 샌다.
#    → 같은 segment의 행은 반드시 한쪽에만 들어가도록 GroupShuffleSplit 사용
gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
train_idx, holdout_idx = next(gss.split(normal_df, groups=normal_df['segment_id']))
train_normal = normal_df.iloc[train_idx]
holdout_normal = normal_df.iloc[holdout_idx]

overlap = set(train_normal['segment_id']) & set(holdout_normal['segment_id'])
print(
    f'-> Train: {train_normal["segment_id"].nunique()}개 segment'
    f' / {len(train_normal)}행'
)
print(
    f'-> Holdout: {holdout_normal["segment_id"].nunique()}개 segment'
    f' / {len(holdout_normal)}행'
)
print(f'-> Train-Holdout 공유 segment 수: {len(overlap)}개 (0이어야 정상)')

# 3. Train 정상 데이터로 Linear Regression 학습 (Current -> Vibration)
lr_ai0 = LinearRegression().fit(
    train_normal[['AI2_Current_RMS']], train_normal['AI0_Vibration_RMS']
)
lr_ai1 = LinearRegression().fit(
    train_normal[['AI2_Current_RMS']], train_normal['AI1_Vibration_RMS']
)


# 4. 잔차(Residual) 계산 함수 (not_evaluable 제외 후라 결측 없음)
def calculate_residuals(data_subset, model_ai0, model_ai1):
  X = data_subset[['AI2_Current_RMS']]
  res_ai0 = np.abs(data_subset['AI0_Vibration_RMS'] - model_ai0.predict(X))
  res_ai1 = np.abs(data_subset['AI1_Vibration_RMS'] - model_ai1.predict(X))
  return res_ai0, res_ai1


holdout_res_ai0, holdout_res_ai1 = calculate_residuals(
    holdout_normal, lr_ai0, lr_ai1
)

# 비교 대상: 이상 전체 / RMS 미검출만 (H1 주장: RMS가 놓친 이상을 잔차가 잡는다)
abnormal_groups = {
    'Abnormal 전체': abnormal_df,
    'Abnormal 중 RMS 미검출': abnormal_df[
        abnormal_df['Outlier_Category'] == 'RMS 미검출'
    ],
}


# 5. 강건 통계량 출력 함수
def print_robust_stats(name, series):
  q25 = series.quantile(0.25)
  q75 = series.quantile(0.75)
  print(f'[{name}] (n={len(series)})')
  print(f'  - Mean ± Std   : {series.mean():.4f} ± {series.std():.4f}')
  print(
      f'  - Median (IQR) : {series.median():.4f} (IQR: {q75 - q25:.4f}'
      f' [Q25: {q25:.4f}, Q75: {q75:.4f}])'
  )
  print(f'  - 95% Quantile : {series.quantile(0.95):.4f}')


# 6. Mann-Whitney U 검정 + 효과크기(rank-biserial) + 정상 95% 기준 초과율
def compare_to_holdout(holdout_series, abnormal_series, name):
  stat, p_value = stats.mannwhitneyu(
      holdout_series, abnormal_series, alternative='two-sided'
  )
  n1, n2 = len(holdout_series), len(abnormal_series)
  effect_size = 1 - (2 * stat) / (n1 * n2)
  threshold = holdout_series.quantile(0.95)
  exceed_rate = (abnormal_series > threshold).mean()

  print(f'[{name} - Mann-Whitney U Test]')
  print(f'  - U statistic : {stat:.1f}')
  print(f'  - p-value     : {p_value:.4e}')
  print(f'  - Effect Size : {abs(effect_size):.4f}')
  print(
      f'  - 정상 Holdout 95% 분위수({threshold:.4f}) 초과 비율 :'
      f' {exceed_rate:.1%}'
  )


print('\n================ [H1 재검증 통계 결과 요약] ================')
for ch, holdout_res in [('AI0', holdout_res_ai0), ('AI1', holdout_res_ai1)]:
  print(f'\n--- [{ch} 채널 잔차 (Residual {ch})] ---')
  print_robust_stats('Normal Holdout', holdout_res)
  for group_name, group_df in abnormal_groups.items():
    res_ai0, res_ai1 = calculate_residuals(group_df, lr_ai0, lr_ai1)
    group_res = res_ai0 if ch == 'AI0' else res_ai1
    print()
    print_robust_stats(group_name, group_res)
    compare_to_holdout(holdout_res, group_res, f'{ch} Residual / {group_name}')

print('\n====================================================================')
print('-> [완료] H1 재검증 및 통계 보강 완료!')
print('====================================================================')
