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