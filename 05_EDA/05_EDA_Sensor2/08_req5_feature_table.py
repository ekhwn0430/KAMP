# --- 실행 위치와 관계없이 동작: 결과는 이 스크립트 폴더에 저장, 원본 CSV는 레포 최상단 data/에서 읽음 ---
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parent)
DATA_DIR = next(_p / "data" for _p in [_Path.cwd(), *_Path.cwd().parents]
                if (_p / "data" / "press_data_normal.csv").exists())
# ---------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

print("===============================================================================================================")
print("                                     [Segment별 종합 Feature Table 구축]")
print("===============================================================================================================")

# 1. 데이터 로드 및 전처리 함수
normal_df = pd.read_csv(DATA_DIR / 'press_data_normal.csv')
outlier_df = pd.read_csv(DATA_DIR / 'outlier_data.csv')

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