import pandas as pd

_original_to_datetime = pd.to_datetime


def _parse_elapsed_or_datetime(values, *args, **kwargs):
    if isinstance(values, pd.Series):
        parts = values.astype("string").str.extract(r"^(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d+)?)$")
        if not parts.isna().any().any():
            elapsed_seconds = parts["minutes"].astype(float) * 60 + parts["seconds"].astype(float)
            return pd.to_timedelta(elapsed_seconds, unit="s")
    return _original_to_datetime(values, *args, **kwargs)


pd.to_datetime = _parse_elapsed_or_datetime


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 폰트 및 스타일 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

print("==========================================")
print("        [분포 시각화 및 통계량 분석]")
print("==========================================")

# 2. 데이터 로드
normal_df = pd.read_csv('../../../data/press_data_normal.csv')
outlier_df = pd.read_csv('../../../data/outlier_data.csv')

# 3. TimeStamp 파싱 및 Segment ID 생성 함수
def preprocess_and_segment(df, time_threshold=0.15):
    df = df.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df.sort_values('TimeStamp').reset_index(drop=True)
    df['time_diff'] = df['TimeStamp'].diff().dt.total_seconds().fillna(0)
    df['segment_id'] = (df['time_diff'] > time_threshold).cumsum()
    return df

normal_proc = preprocess_and_segment(normal_df)
outlier_proc = preprocess_and_segment(outlier_df)

# 4. Segment 내부 Rolling RMS 계산 (Raw 값 보존)
cols = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
window_size = 10

def calculate_segment_rms(df, cols, window=10):
    rms_list = []
    for seg_id, group in df.groupby('segment_id'):
        g_rms = group[cols].pow(2).rolling(window=window, min_periods=1).mean().pow(0.5)
        g_rms['segment_id'] = seg_id
        g_rms['TimeStamp'] = group['TimeStamp']
        g_rms['Equipment_state'] = group['Equipment_state']
        for c in cols:
            g_rms[f'{c}_Raw'] = group[c].values
        rms_list.append(g_rms)
    return pd.concat(rms_list).sort_index()

normal_rms = calculate_segment_rms(normal_proc, cols, window=window_size)
outlier_rms = calculate_segment_rms(outlier_proc, cols, window=window_size)

# 5. IQR Threshold 기반 미검출 이상 데이터(259개) 분리
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

combined_detected = detections['AI0_Vibration'] | detections['AI1_Vibration'] | detections['AI2_Current']
outlier_rms['RMS_Detected'] = combined_detected
undetected_rms = outlier_rms[~outlier_rms['RMS_Detected']].copy()

print(f"-> 정상 데이터 행 수        : {len(normal_rms)}개")
print(f"-> RMS 미검출 이상 데이터 수 : {len(undetected_rms)}개")
print("------------------------------------------")

# 6. [시각화 1] 밀도 분포(KDE) 비교 그래프 생성
print("-> [생성 중] 밀도 분포 그래프 ('04_req1_distribution_comparison.png')...")
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
titles = ['AI0 Vibration', 'AI1 Vibration', 'AI2 Current']

for i, col in enumerate(cols):
    sns.kdeplot(normal_rms[f'{col}_Raw'], label='Normal', ax=axes[i, 0], color='blue', fill=True, alpha=0.3)
    sns.kdeplot(undetected_rms[f'{col}_Raw'], label='Undetected Outlier', ax=axes[i, 0], color='red', fill=True, alpha=0.3)
    axes[i, 0].set_title(f'{titles[i]} - Raw Distribution', fontsize=12, fontweight='bold')
    axes[i, 0].legend()
    
    sns.kdeplot(normal_rms[col], label='Normal', ax=axes[i, 1], color='blue', fill=True, alpha=0.3)
    sns.kdeplot(undetected_rms[col], label='Undetected Outlier', ax=axes[i, 1], color='orange', fill=True, alpha=0.3)
    axes[i, 1].set_title(f'{titles[i]} - Rolling RMS Distribution', fontsize=12, fontweight='bold')
    axes[i, 1].legend()

plt.tight_layout()
plt.savefig('step1_distribution_comparison.png', dpi=300)
plt.close()

# 7. [시각화 2] 주요 통계량 박스플롯(Median, Q1, Q3, Mean) 생성
print("-> [생성 중] 통계 박스플롯 ('04_req1_boxplot.png')...")
plot_data = []
for col in cols:
    for val in normal_rms[f'{col}_Raw']:
        plot_data.append({'Channel': col, 'Feature': 'Raw', 'Group': 'Normal', 'Value': val})
    for val in undetected_rms[f'{col}_Raw']:
        plot_data.append({'Channel': col, 'Feature': 'Raw', 'Group': 'Undetected', 'Value': val})
    for val in normal_rms[col]:
        plot_data.append({'Channel': col, 'Feature': 'Rolling_RMS', 'Group': 'Normal', 'Value': val})
    for val in undetected_rms[col]:
        plot_data.append({'Channel': col, 'Feature': 'Rolling_RMS', 'Group': 'Undetected', 'Value': val})

df_plot = pd.DataFrame(plot_data)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for i, col in enumerate(cols):
    sub_df = df_plot[df_plot['Channel'] == col]
    
    sns.boxplot(data=sub_df[sub_df['Feature'] == 'Raw'], x='Group', y='Value', hue='Group', ax=axes[i, 0], palette=['blue', 'orange'], showmeans=True, legend=False,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
    axes[i, 0].set_title(f'{titles[i]} - Raw Statistics (Median/Q1/Q3/Mean)', fontsize=12, fontweight='bold')
    
    sns.boxplot(data=sub_df[sub_df['Feature'] == 'Rolling_RMS'], x='Group', y='Value', hue='Group', ax=axes[i, 1], palette=['blue', 'orange'], showmeans=True, legend=False,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
    axes[i, 1].set_title(f'{titles[i]} - Rolling RMS Statistics (Median/Q1/Q3/Mean)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('step1_boxplot_statistics.png', dpi=300)
plt.close()

print("-> [완료] 모든 시각화 이미지 파일 저장 완료!")
print("==========================================")

# 8. 터미널에 상세 통계량 표(Table) 출력
print("\n\n==============================================================================================================")
print("                                     [채널별 주요 통계량 상세 비교표]")
print("==============================================================================================================")
stats_list = []
for col in cols:
    for name, df in [('Normal', normal_rms), ('Undetected_Outlier', undetected_rms)]:
        s_raw = df[f'{col}_Raw']
        stats_list.append({
            'Channel': col, 'Group': name, 'Type': 'Raw',
            'Mean': s_raw.mean(), 'Std': s_raw.std(), 'Median': s_raw.median(),
            'Q1': s_raw.quantile(0.25), 'Q3': s_raw.quantile(0.75), '95% Q': s_raw.quantile(0.95)
        })
        s_rms = df[col]
        stats_list.append({
            'Channel': col, 'Group': name, 'Type': 'Rolling_RMS',
            'Mean': s_rms.mean(), 'Std': s_rms.std(), 'Median': s_rms.median(),
            'Q1': s_rms.quantile(0.25), 'Q3': s_rms.quantile(0.75), '95% Q': s_rms.quantile(0.95)
        })

stats_df = pd.DataFrame(stats_list)
print(stats_df.to_string(index=False))
print("==========================================")


import pandas as pd
import numpy as np

print("데이터 로드 및 전처리 중...")
# 1. CSV Load
normal_df = pd.read_csv('../../../data/press_data_normal.csv')
outlier_df = pd.read_csv('../../../data/outlier_data.csv')

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

normal_rms = calculate_segment_rms(normal_proc, cols, window=window_size)
outlier_rms = calculate_segment_rms(outlier_proc, cols, window=window_size)

# 4. IQR Threshold 계산 (정상 데이터 기준) 및 이상치 탐지
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

det_ai0 = detections['AI0_Vibration']
det_ai1 = detections['AI1_Vibration']
det_ai2 = detections['AI2_Current']

# 5. 채널별 단독 및 세부 조합 검출 분석
total_outliers = len(outlier_rms)
combined_detected = det_ai0 | det_ai1 | det_ai2
undetected = ~combined_detected

solo_ai0 = det_ai0 & (~det_ai1) & (~det_ai2)
solo_ai1 = det_ai1 & (~det_ai0) & (~det_ai2)
solo_ai2 = det_ai2 & (~det_ai0) & (~det_ai1)

comb_ai0_ai1 = det_ai0 & det_ai1 & (~det_ai2)
comb_ai0_ai2 = det_ai0 & det_ai2 & (~det_ai1)
comb_ai1_ai2 = det_ai1 & det_ai2 & (~det_ai0)
overlap_all = det_ai0 & det_ai1 & det_ai2

print("\n==========================================")
print("  [RMS 이상탐지 기여도 및 세부 조합 분석]")
print("==========================================")
print(f"총 이상 데이터 수        : {total_outliers}개")
print(f"RMS 검출된 데이터 수     : {combined_detected.sum()}개")
print(f"RMS 미검출 데이터 수     : {undetected.sum()}개")
print("------------------------------------------")
print(f"• AI0 단독 검출          : {solo_ai0.sum()}개")
print(f"• AI1 단독 검출          : {solo_ai1.sum()}개")
print(f"• AI2 단독 검출          : {solo_ai2.sum()}개")
print("------------------------------------------")
print(f"• AI0 + AI1 동시 검출 (AI2는 정상) : {comb_ai0_ai1.sum()}개")
print(f"• AI0 + AI2 동시 검출 (AI1은 정상) : {comb_ai0_ai2.sum()}개")
print(f"• AI1 + AI2 동시 검출 (AI0은 정상) : {comb_ai1_ai2.sum()}개")
print(f"• AI0 + AI1 + AI2 전부 동시 검출   : {overlap_all.sum()}개")
print("==========================================")
print(f"[참고] 개별 채널 임계값 초과 총 횟수:")
print(f"• AI0 총 초과: {det_ai0.sum()}개")
print(f"• AI1 총 초과: {det_ai1.sum()}개")
print(f"• AI2 총 초과: {det_ai2.sum()}개")
print("==========================================")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 폰트 및 스타일 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

print("=====================================================================")
print("                     [가설 1 잔차 분석 및 시각화]")
print("=====================================================================")

# 2. 데이터 로드 (필요시 feature_table 또는 원본 데이터 로드)
# 여기선 예시로 기존에 다듬어둔 데이터프레임을 활용하는 흐름으로 구성함.
normal_df = pd.read_csv('../../../data/press_data_normal.csv')
outlier_df = pd.read_csv('../../../data/outlier_data.csv')

# 3. TimeStamp 파싱 및 Segment ID 생성 함수
def preprocess_and_segment(df, time_threshold=0.15):
    df = df.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df.sort_values('TimeStamp').reset_index(drop=True)
    df['time_diff'] = df['TimeStamp'].diff().dt.total_seconds().fillna(0)
    df['segment_id'] = (df['time_diff'] > time_threshold).cumsum()
    return df

normal_proc = preprocess_and_segment(normal_df)
outlier_proc = preprocess_and_segment(outlier_df)

# 4. Segment 내부 Rolling RMS 계산
cols = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
window_size = 10

def calculate_segment_rms(df, cols, window=10):
    rms_list = []
    for seg_id, group in df.groupby('segment_id'):
        g_rms = group[cols].pow(2).rolling(window=window, min_periods=1).mean().pow(0.5)
        g_rms['segment_id'] = seg_id
        g_rms['TimeStamp'] = group['TimeStamp']
        g_rms['Equipment_state'] = group['Equipment_state']
        for c in cols:
            g_rms[f'{c}_Raw'] = group[c].values
        rms_list.append(g_rms)
    return pd.concat(rms_list).sort_index()

normal_rms = calculate_segment_rms(normal_proc, cols, window=window_size)
outlier_rms = calculate_segment_rms(outlier_proc, cols, window=window_size)

# 5. IQR Threshold 기반 미검출 이상 데이터 분리
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

combined_detected = detections['AI0_Vibration'] | detections['AI1_Vibration'] | detections['AI2_Current']
outlier_rms['RMS_Detected'] = combined_detected
undetected_rms = outlier_rms[~outlier_rms['RMS_Detected']].copy()

# 6. 간단한 회귀 잔차 계산 시뮬레이션 (AI2를 기준으로 AI0, AI1과의 잔차 산출)
from sklearn.linear_model import LinearRegression

# Normal 데이터를 기준으로 회귀식 학습
reg_ai0 = LinearRegression().fit(normal_rms[['AI2_Current']], normal_rms['AI0_Vibration'])
reg_ai1 = LinearRegression().fit(normal_rms[['AI2_Current']], normal_rms['AI1_Vibration'])

# Normal 잔차
normal_res_ai0 = np.abs(normal_rms['AI0_Vibration'] - reg_ai0.predict(normal_rms[['AI2_Current']]))
normal_res_ai1 = np.abs(normal_rms['AI1_Vibration'] - reg_ai1.predict(normal_rms[['AI2_Current']]))

# Undetected 잔차
undetected_res_ai0 = np.abs(undetected_rms['AI0_Vibration'] - reg_ai0.predict(undetected_rms[['AI2_Current']]))
undetected_res_ai1 = np.abs(undetected_rms['AI1_Vibration'] - reg_ai1.predict(undetected_rms[['AI2_Current']]))

# 7. [시각화 1] 잔차 비교 박스플롯 생성 ('06_req3_residual_boxplot.png')
print("-> [생성 중] 잔차 박스플롯 그래프 ('06_req3_residual_boxplot.png')...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# AI2 -> AI0 잔차 박스플롯
res_data_ai0 = pd.DataFrame({
    'Absolute Residual': np.concatenate([normal_res_ai0, undetected_res_ai0]),
    'Group': ['Normal']*len(normal_res_ai0) + ['Undetected']*len(undetected_res_ai0)
})
sns.boxplot(data=res_data_ai0, x='Group', y='Absolute Residual', hue='Group', ax=axes[0], palette=['blue', 'orange'], showmeans=True, legend=False,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
axes[0].set_title('AI2 -> AI0 Regression Residual Comparison', fontsize=12, fontweight='bold')

# AI2 -> AI1 잔차 박스플롯
res_data_ai1 = pd.DataFrame({
    'Absolute Residual': np.concatenate([normal_res_ai1, undetected_res_ai1]),
    'Group': ['Normal']*len(normal_res_ai1) + ['Undetected']*len(undetected_res_ai1)
})
sns.boxplot(data=res_data_ai1, x='Group', y='Absolute Residual', hue='Group', ax=axes[1], palette=['blue', 'orange'], showmeans=True, legend=False,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
axes[1].set_title('AI2 -> AI1 Regression Residual Comparison', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('06_req3_residual_boxplot.png', dpi=300)
plt.close()

# 8. 회귀 분석 결과(Slope, R2)와 잔차 통계를 포함한 상세 비교표 생성
from sklearn.metrics import r2_score

# 회귀 모델 평가 지표 추출 함수
def get_regression_stats(X, y_true, model, res):
    slope = model.coef_[0]
    r2 = r2_score(y_true, model.predict(X))
    return {
        'Slope': round(slope, 4),
        'R2': round(r2, 4),
        'Mean_Res': round(res.mean(), 4),
        'Std_Res': round(res.std(), 4),
        'Median_Res': round(res.median(), 4),
        'Q1_Res': round(np.percentile(res, 25), 4),
        'Q3_Res': round(np.percentile(res, 75), 4)
    }

# Normal 및 Undetected 지표 계산
stats_list = []

# AI2 -> AI0 회귀 및 통계
reg_ai0_u = LinearRegression().fit(undetected_rms[['AI2_Current']], undetected_rms['AI0_Vibration'])
undetected_res_ai0 = np.abs(undetected_rms['AI0_Vibration'] - reg_ai0_u.predict(undetected_rms[['AI2_Current']]))

stats_list.append({**{'Pair': 'AI2 -> AI0', 'Group': 'Normal'}, **get_regression_stats(normal_rms[['AI2_Current']], normal_rms['AI0_Vibration'], reg_ai0, normal_res_ai0)})
stats_list.append({**{'Pair': 'AI2 -> AI0', 'Group': 'Undetected'}, **get_regression_stats(undetected_rms[['AI2_Current']], undetected_rms['AI0_Vibration'], reg_ai0_u, undetected_res_ai0)})

# AI2 -> AI1 회귀 및 통계
reg_ai1_u = LinearRegression().fit(undetected_rms[['AI2_Current']], undetected_rms['AI1_Vibration'])
undetected_res_ai1 = np.abs(undetected_rms['AI1_Vibration'] - reg_ai1_u.predict(undetected_rms[['AI2_Current']]))

stats_list.append({**{'Pair': 'AI2 -> AI1', 'Group': 'Normal'}, **get_regression_stats(normal_rms[['AI2_Current']], normal_rms['AI1_Vibration'], reg_ai1, normal_res_ai1)})
stats_list.append({**{'Pair': 'AI2 -> AI1', 'Group': 'Undetected'}, **get_regression_stats(undetected_rms[['AI2_Current']], undetected_rms['AI1_Vibration'], reg_ai1_u, undetected_res_ai1)})

stats_df = pd.DataFrame(stats_list)

print("\n\n===================================================================================")
print("                     [가설 1 보강: 회귀 분석 및 잔차 통계 비교표]")
print("===================================================================================")
print(stats_df.to_string(index=False))
print("===================================================================================")

# 9. [시각화 2] 보강된 통계 표를 이미지 파일('06_req3_stats_table.png')로 저장
print("-> [생성 중] 보강된 통계 표 이미지 ('06_req3_stats_table.png')...")
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.6)

for key, cell in table.get_celld().items():
    if key[0] == 0:
        cell.set_facecolor('#4c72b0')
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
    else:
        cell.set_facecolor('#f9f9f9' if key[0] % 2 == 0 else 'white')

plt.title('Hypothesis 1 Enhancement: Regression & Residual Statistics', fontsize=13, fontweight='bold', pad=20)
plt.savefig('06_req3_stats_table.png', dpi=300, bbox_inches='tight')
plt.close()

print("-> [완료] 가설 1 보강 표 이미지 재생성 완료!")
print("===================================================================================")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 폰트 및 스타일 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

print("======================================================================")
print("                     [가설 2 보강 : ND 지표 분석]")
print("======================================================================")

# 2. 데이터 로드 및 전처리
normal_df = pd.read_csv('../../../data/press_data_normal.csv')
outlier_df = pd.read_csv('../../../data/outlier_data.csv')

def preprocess_and_segment(df, time_threshold=0.15):
    df = df.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df.sort_values('TimeStamp').reset_index(drop=True)
    df['time_diff'] = df['TimeStamp'].diff().dt.total_seconds().fillna(0)
    df['segment_id'] = (df['time_diff'] > time_threshold).cumsum()
    return df

normal_proc = preprocess_and_segment(normal_df)
outlier_proc = preprocess_and_segment(outlier_df)

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

normal_rms = calculate_segment_rms(normal_proc, cols, window=window_size)
outlier_rms = calculate_segment_rms(outlier_proc, cols, window=window_size)

# 3. IQR Threshold 기반 미검출 이상 데이터(259개) 분리
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

combined_detected = detections['AI0_Vibration'] | detections['AI1_Vibration'] | detections['AI2_Current']
outlier_rms['RMS_Detected'] = combined_detected
undetected_rms = outlier_rms[~outlier_rms['RMS_Detected']].copy()

# 4. Normalized Difference (ND) 지표 계산 함수
def calculate_nd(df, eps=1e-6):
    df = df.copy()
    df['ND'] = (df['AI0_Vibration'] - df['AI1_Vibration']) / (df['AI0_Vibration'] + df['AI1_Vibration'] + eps)
    return df

normal_nd = calculate_nd(normal_rms)
undetected_nd = calculate_nd(undetected_rms)

# 5. 통계량 출력
print("\n[Normalized Difference (ND) 분석 결과]")
print(f"• 정상 데이터 ND 평균       : {normal_nd['ND'].mean():.4f} (표준편차: {normal_nd['ND'].std():.4f})")
print(f"• 미검출 이상 데이터 ND 평균 : {undetected_nd['ND'].mean():.4f} (표준편차: {undetected_nd['ND'].std():.4f})")

# 6. ND 분포 비교 시각화 (KDE Plot) 및 저장
plt.figure(figsize=(10, 6))
sns.kdeplot(normal_nd['ND'], label='Normal', color='blue', fill=True, alpha=0.3)
sns.kdeplot(undetected_nd['ND'], label='Undetected Outlier', color='orange', fill=True, alpha=0.3)
plt.title('Normalized Difference (ND) Distribution Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Normalized Difference (ND)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('07_req4_nd_distribution.png', dpi=300)
plt.close()

print("\n-> [완료] ND 분포 밀도 그래프 '07_req4_nd_distribution.png' 저장 완료!")
print("======================================================================")


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

print("===============================================================================================================")
print("                                     [Segment별 종합 Feature Table 구축]")
print("===============================================================================================================")

# 1. 데이터 로드 및 전처리 함수
normal_df = pd.read_csv('../../../data/press_data_normal.csv')
outlier_df = pd.read_csv('../../../data/outlier_data.csv')

def preprocess_and_segment(df, time_threshold=0.15):
    df = df.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df.sort_values('TimeStamp').reset_index(drop=True)
    df['time_diff'] = df['TimeStamp'].diff().dt.total_seconds().fillna(0)
    df['segment_id'] = (df['time_diff'] > time_threshold).cumsum()
    return df

normal_proc = preprocess_and_segment(normal_df)
outlier_proc = preprocess_and_segment(outlier_df)

cols = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
window_size = 10

def calculate_segment_features(df, cols, window=10, lr_model=None):
    feature_rows = []
    
    for seg_id, group in df.groupby('segment_id'):
        # 1. Rolling RMS 계산
        g_rms = group[cols].pow(2).rolling(window=window, min_periods=1).mean().pow(0.5)
        
        # 2. Normalized Difference (ND) 계산
        eps = 1e-6
        nd = (group['AI0_Vibration'] - group['AI1_Vibration']) / (group['AI0_Vibration'] + group['AI1_Vibration'] + eps)
        
        # 3. 회귀 잔차 (AI2 -> AI1 Residual) 계산
        if lr_model is not None:
            pred_ai1 = lr_model.predict(group[['AI2_Current']].values)
            residual = np.abs(group['AI1_Vibration'] - pred_ai1)
        else:
            residual = np.zeros(len(group))

        # 세그먼트 단위 집계 딕셔너리 구성
        feat = {
            'segment_id': seg_id,
            'data_count': len(group),
            
            # RMS Mean & Max
            'ai0_rms_mean': g_rms['AI0_Vibration'].mean(),
            'ai0_rms_max': g_rms['AI0_Vibration'].max(),
            'ai1_rms_mean': g_rms['AI1_Vibration'].mean(),
            'ai1_rms_max': g_rms['AI1_Vibration'].max(),
            'ai2_rms_mean': g_rms['AI2_Current'].mean(),
            'ai2_rms_max': g_rms['AI2_Current'].max(),
            
            # ND Mean
            'nd_mean': nd.mean(),
            'nd_max': nd.max(),
            
            # Residual Mean & Max
            'residual_mean': residual.mean(),
            'residual_max': residual.max(),
            
            'Equipment_state': group['Equipment_state'].iloc[0] if 'Equipment_state' in group.columns else 0
        }
        feature_rows.append(feat)
        
    return pd.DataFrame(feature_rows)

# 2. 정상 데이터 기준으로 회귀 모델 학습 (가설 1 활용)
print("-> 정상 데이터 기반 회귀 모델 학습 중...")
normal_rms_temp = []
for seg_id, group in normal_proc.groupby('segment_id'):
    g_rms = group[cols].pow(2).rolling(window=window_size, min_periods=1).mean().pow(0.5)
    normal_rms_temp.append(g_rms)
normal_rms_df = pd.concat(normal_rms_temp)

X_norm = normal_rms_df[['AI2_Current']].values
y_norm_ai1 = normal_rms_df['AI1_Vibration'].values
lr_ai1 = LinearRegression().fit(X_norm, y_norm_ai1)

# 3. Normal 및 Outlier 세그먼트 피처 테이블 생성
print("-> 세그먼트별 종합 피처 추출 중...")
normal_features = calculate_segment_features(normal_proc, cols, window=window_size, lr_model=lr_ai1)
normal_features['label'] = 'Normal'

outlier_features = calculate_segment_features(outlier_proc, cols, window=window_size, lr_model=lr_ai1)
outlier_features['label'] = 'Outlier'

# 4. 통합 피처 테이블 결합 및 저장
comprehensive_feature_table = pd.concat([normal_features, outlier_features], ignore_index=True)
comprehensive_feature_table.to_csv('segment_feature_table.csv', index=False)

print("===============================================================================================================")
print(f"-> [완료] 종합 Feature Table 생성 완료!")
print(f"-> 총 세그먼트 수: {len(comprehensive_feature_table)}개")
print(f"-> 저장된 파일명: 'segment_feature_table.csv'")
print("===============================================================================================================")
print(comprehensive_feature_table.head(10))
