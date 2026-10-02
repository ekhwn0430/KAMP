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
normal_df = pd.read_csv('press_data_normal.csv')
outlier_df = pd.read_csv('outlier_data.csv')

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

# [중요] Slope/R² 비교를 위한 Undetected 전용 회귀모델 학습은 따로 수행 (PDF 허용 사항)
reg_ai0_u = LinearRegression().fit(undetected_rms[['AI2_Current']], undetected_rms['AI0_Vibration'])
reg_ai1_u = LinearRegression().fit(undetected_rms[['AI2_Current']], undetected_rms['AI1_Vibration'])

# [중요] 통계표의 Residual도 박스플롯과 똑같이 'Normal 모델(reg_ai0, reg_ai1)' 기준 잔차로 완벽 고정!
undetected_res_ai0_fixed = np.abs(undetected_rms['AI0_Vibration'] - reg_ai0.predict(undetected_rms[['AI2_Current']]))
undetected_res_ai1_fixed = np.abs(undetected_rms['AI1_Vibration'] - reg_ai1.predict(undetected_rms[['AI2_Current']]))

# AI2 -> AI0 회귀 및 통계 (Normal은 reg_ai0, Undetected도 잔차는 reg_ai0 기준 적용)
stats_list.append({**{'Pair': 'AI2 -> AI0', 'Group': 'Normal'}, **get_regression_stats(normal_rms[['AI2_Current']], normal_rms['AI0_Vibration'], reg_ai0, normal_res_ai0)})
stats_list.append({**{'Pair': 'AI2 -> AI0', 'Group': 'Undetected'}, **get_regression_stats(undetected_rms[['AI2_Current']], undetected_rms['AI0_Vibration'], reg_ai0_u, undetected_res_ai0_fixed)})

# AI2 -> AI1 회귀 및 통계 (Normal은 reg_ai1, Undetected도 잔차는 reg_ai1 기준 적용)
stats_list.append({**{'Pair': 'AI2 -> AI1', 'Group': 'Normal'}, **get_regression_stats(normal_rms[['AI2_Current']], normal_rms['AI1_Vibration'], reg_ai1, normal_res_ai1)})
stats_list.append({**{'Pair': 'AI2 -> AI1', 'Group': 'Undetected'}, **get_regression_stats(undetected_rms[['AI2_Current']], undetected_rms['AI1_Vibration'], reg_ai1_u, undetected_res_ai1_fixed)})

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