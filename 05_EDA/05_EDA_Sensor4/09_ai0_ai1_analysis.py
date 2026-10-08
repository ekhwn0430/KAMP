# --- 실행 위치와 관계없이 동작: 결과는 이 스크립트 폴더에 저장, 원본 CSV는 레포 최상단 data/에서 읽음 ---
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parent)
DATA_DIR = next(_p / "data" for _p in [_Path.cwd(), *_Path.cwd().parents]
                if (_p / "data" / "press_data_normal.csv").exists())
# ---------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 1. 폰트 및 시각화 스타일 설정 (기존 선호하시는 원본 색상 적용)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

print("===================================================================================")
print("                [AI0 및 AI1-AI2 관계 이상특성 분석 및 시각화 시작]")
print("===================================================================================") 

# 2. 데이터 로드
normal_df = pd.read_csv(DATA_DIR / 'press_data_normal.csv')
outlier_df = pd.read_csv(DATA_DIR / 'outlier_data.csv')

# 3. 타임스탬프 무결성 검증 및 세그먼트 분리 전처리 함수
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

# 4. Rolling RMS 계산 함수
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

# 5. IQR Thresholds 산출 및 RMS 검출/미검출(RMS_Detected / RMS_Missed) 분리
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

print(f"-> 전체 이상 행 수: {len(outlier_rms)}개 | RMS 검출: {combined_detected.sum()}개 | RMS 미검출(Missed): {len(undetected_rms)}개")

# ===================================================================================
# [주제 1] AI0 자체 이상특성 분석 및 시각화
# ===================================================================================
print("\n-> [생성 중] AI0 자체 이상특성 분석 시각화 ('ai0_individual_analysis.png')...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: AI0 밀도 분포(KDE) 비교
sns.kdeplot(normal_rms['AI0_Vibration'], label='Normal', ax=axes[0], color='blue', fill=True, alpha=0.3, linewidth=2)
sns.kdeplot(outlier_rms['AI0_Vibration'], label='Outlier (Total)', ax=axes[0], color='red', fill=True, alpha=0.3, linewidth=2)
axes[0].set_title('AI0 Vibration RMS Distribution (Normal vs Total Outlier)', fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlabel('AI0 Vibration RMS', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].legend(fontsize=11)

# Right: 그룹별 비교 박스플롯 (배열 정돈 및 원본 색상 적용)
outlier_rms_copy = outlier_rms.copy()
outlier_rms_copy['Group'] = np.where(outlier_rms_copy['RMS_Detected'], 'RMS_Detected', 'RMS_Missed')
plot_box_data = pd.concat([
    pd.DataFrame({'AI0_Vibration': normal_rms['AI0_Vibration'], 'Group': 'Normal'}),
    pd.DataFrame({'AI0_Vibration': outlier_rms_copy['AI0_Vibration'], 'Group': outlier_rms_copy['Group']})
])
plot_box_data['Group'] = pd.Categorical(plot_box_data['Group'], categories=['Normal', 'RMS_Detected', 'RMS_Missed'], ordered=True)

sns.boxplot(data=plot_box_data, x='Group', y='AI0_Vibration', hue='Group', ax=axes[1], 
            palette=['blue', 'orange', 'red'], showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"},
            boxprops=dict(alpha=0.85), width=0.5)

axes[1].set_title('AI0 RMS Comparison across Groups', fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel('Classification Group', fontsize=11)
axes[1].set_ylabel('AI0 Vibration RMS', fontsize=11)
if axes[1].get_legend():
    axes[1].get_legend().remove()

plt.tight_layout()
plt.savefig('ai0_individual_analysis.png', dpi=300)
plt.close()


# ===================================================================================
# [주제 2] AI1 - AI2 관계 이상특성 (회귀 잔차) 분석 및 시각화
# ===================================================================================
print("-> [생성 중] AI1-AI2 관계 잔차 분석 시각화 ('ai1_ai2_relationship_analysis.png')...")

# Normal 기준 AI2(Current) -> AI1(Vibration) 선형회귀 모델 학습
lr_ai1_ai2 = LinearRegression().fit(normal_rms[['AI2_Current']], normal_rms['AI1_Vibration'])
normal_ai1_res = np.abs(normal_rms['AI1_Vibration'] - lr_ai1_ai2.predict(normal_rms[['AI2_Current']]))
undetected_ai1_res = np.abs(undetected_rms['AI1_Vibration'] - lr_ai1_ai2.predict(undetected_rms[['AI2_Current']]))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: 산점도 (Normal vs RMS-Missed Outlier)
axes[0].scatter(normal_rms['AI2_Current'], normal_rms['AI1_Vibration'], alpha=0.3, color='blue', label='Normal', s=12)
axes[0].scatter(undetected_rms['AI2_Current'], undetected_rms['AI1_Vibration'], alpha=0.6, color='orange', label='RMS-Missed Outlier', s=25, edgecolor='black', linewidth=0.5)
axes[0].set_title('AI2 (Current) vs AI1 (Vibration) Relationship', fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlabel('AI2 Current RMS', fontsize=11)
axes[0].set_ylabel('AI1 Vibration RMS', fontsize=11)
axes[0].legend(fontsize=11)

# Right: 절대 잔차(Absolute Residual) 박스플롯 비교
res_data = pd.DataFrame({
    'Absolute Residual': np.concatenate([normal_ai1_res, undetected_ai1_res]),
    'Group': ['Normal']*len(normal_ai1_res) + ['RMS-Missed']*len(undetected_ai1_res)
})
res_data['Group'] = pd.Categorical(res_data['Group'], categories=['Normal', 'RMS-Missed'], ordered=True)

sns.boxplot(data=res_data, x='Group', y='Absolute Residual', hue='Group', ax=axes[1], 
            palette=['blue', 'orange'], showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"},
            boxprops=dict(alpha=0.85), width=0.4)

axes[1].set_title('AI2 -> AI1 Regression Absolute Residual', fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel('Classification Group', fontsize=11)
axes[1].set_ylabel('Absolute Residual Value', fontsize=11)
if axes[1].get_legend():
    axes[1].get_legend().remove()

plt.tight_layout()
plt.savefig('ai1_ai2_relationship_analysis.png', dpi=300)
plt.close()

print("===================================================================================")
print("-> [완료] 시각화 이미지 생성 완료!")
print("===================================================================================")