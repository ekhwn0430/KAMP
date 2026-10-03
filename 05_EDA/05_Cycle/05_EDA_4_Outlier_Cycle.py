"""
Outlier AI2 Cycle 분석 — Normal 기준 freeze

목적
  Normal에서 확정한 약 1.7초 AI2 반복 Cycle 기준을 Outlier에 그대로 적용해
  (a) 1.7초 주기 자체가 깨지는 이상, (b) 주기는 유지되지만 shape/amplitude만 깨지는 이상을 구분한다.

원칙
  - 데이터·segment는 공식 전처리(04_Preprocessing, data/processed/preprocessed_data.csv) 그대로 사용
    Normal 19,999행 / 599 segment, Abnormal 600행 / 21 segment, key = source / segment_id / source_row
    (preprocessed_data.csv가 없으면 data/*.csv에서 04_Preprocessing과 같은 규칙으로 생성)
  - Template_Correlation / Shape_RMSE 정의는 07_01_Feature와 동일 (32-point 위상정규화 + z-score)
  - Peak detector, Template, 모든 threshold는 Normal에서만 결정 후 freeze. Outlier를 보고 재조정하지 않음
  - Global Cycle(긴 segment, peak-to-peak) 우선, 짧은 segment만 Local Form partial matching
  - 신뢰도가 부족하면 not_evaluable. not_evaluable은 정상으로 세지 않음
  - partial 결과는 full-cycle 결과와 같은 의미로 해석하지 않음 (보조 coverage)

실행:  python outlier_ai2_cycle.py      (data/ 폴더를 상위 경로에서 자동 탐색, feature/Archive 브랜치 기준)
출력:  results/cycle_outlier/
  cycle_units.csv            Cycle 단위 최종 Table (Normal+Outlier)
  cycle_rows.csv             행 단위 merge용 Table (source, segment_id, source_row, cycle_id)
  segment_summary_outlier.csv Outlier segment별 결과
  group_comparison.csv       Normal / RMS-detected / RMS-missed 비교
  meeting_numbers.csv        회의용 숫자
  frozen_params.json         Normal에서 결정한 파라미터
  partial_thresholds_by_length.csv  partial 길이별 Normal threshold + 최소 길이 검증

cycle_units.csv 컬럼
  cycle_id        full_cycle은 segment 내 peak 순번(1,2,..), segment 단위 unit(partial 등)은 0
  status          full_cycle(peak-to-peak) / partial(짧은 segment 전체) / global_no_cycle(충분히 긴데 cycle 미검출) / not_evaluable
  cycle_duration  full_cycle: peak-to-peak 시간, partial: segment 길이(주기 아님)
  period          segment ACF 주기 (global segment만)
  return_error    |AI2(t+1.7초)-AI2(t)| 평균 / 1주기 환산 진폭
  template_corr   full: 07과 동일(32점 z-shape vs Normal template), partial: 최적 위상 구간과의 상관
  shape_rmse      z-shape와 template의 RMSE (07과 동일 정의, partial은 최적 위상 구간 기준)
  amplitude       1주기 peak-to-peak (partial은 1주기 환산값),  offset_ratio  1주기 평균 수준 / 진폭
  verdict         period_broken / period_kept_form_broken / period_kept_normal_like /
                  period_unknown_form_broken / period_unknown_normal_like (partial 18샘플 미만: 주기 판정 불가) / not_evaluable
  rms_group       unit 내 RMS 판정 가능 행 다수결 (Normal은 'Normal')
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "cycle_outlier"
FILES = {"normal": "press_data_normal.csv", "abnormal": "outlier_data.csv"}
SIG = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
ANCHOR = "AI2_Current"
FS = 10.0
GAP_SEC = 0.15

# 팀 detector 설정 (5.EDA_0_Cycle / 5.EDA_2_Detector와 동일)
SMOOTH = 3                  # centered rolling median
PEAK_DISTANCE = 12          # samples
DETECTOR_MIN_ROWS = 30      # fixed prominence 산출 시 사용한 기존 조건
PERIOD_SEARCH = (12, 22)    # ACF 주기 탐색 lag 범위 (1.2~2.2초)
RMS_WINDOW = 10
PHASE_POINTS = 32           # 07_01_Feature와 동일
PHASE_STEPS = 4             # partial matching 위상 탐색: 1 sample을 4등분
Q_LO, Q_HI = 0.005, 0.995   # Normal 기준 분위수 threshold
COVERAGE_TARGET = 0.99      # Global 분석 최소 길이 결정 기준
NULL_PASS_MAX = 0.05        # partial 최소 길이 결정 기준 (shuffle null 통과율)
SEED = 42

VERDICT_DETECTED = {"period_broken", "period_kept_form_broken", "period_unknown_form_broken"}


# =============================================================== 0. Load (공식 전처리 기준)
def find_root():
    for base in [HERE, *HERE.parents, Path.cwd(), *Path.cwd().parents]:
        if (base / "data" / "processed" / "preprocessed_data.csv").exists():
            return base, "processed"
        if all((base / "data" / f).exists() for f in FILES.values()):
            return base, "raw"
    raise FileNotFoundError("data/processed/preprocessed_data.csv 또는 data/*.csv를 찾을 수 없음")


def preprocess_like_04(path, source):
    """04_Preprocessing.ipynb load_and_clean과 동일한 규칙"""
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    df["source_row"] = np.arange(len(df))
    df = df.dropna(subset=["TimeStamp", *SIG, "Equipment_state"]).copy()
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
    df = df.loc[~df.duplicated(subset=["TimeStamp", *SIG, "Equipment_state"])].reset_index(drop=True)
    df["elapsed_sec"] = (df["TimeStamp"] - df["TimeStamp"].iloc[0]).dt.total_seconds()
    dt = df["TimeStamp"].diff().dt.total_seconds()
    seg_no = ((dt > GAP_SEC) | (dt <= 0)).cumsum()
    df["segment_id"] = ("N" if source == "normal" else "A") + seg_no.astype(str).str.zfill(4)
    df["pos_in_seg"] = df.groupby("segment_id", sort=False).cumcount()
    df["seg_len"] = df.groupby("segment_id", sort=False)["segment_id"].transform("size")
    df["source"] = source
    return df


def load():
    root, kind = find_root()
    if kind == "processed":
        path = root / "data" / "processed" / "preprocessed_data.csv"
        data = pd.read_csv(path)
    else:
        path = root / "data"
        data = pd.concat([preprocess_like_04(path / f, s) for s, f in FILES.items()], ignore_index=True)
    data = data.sort_values(["source", "segment_id", "pos_in_seg"], kind="stable").reset_index(drop=True)
    return data, str(path)


def seg_num(seg_id):
    return int(str(seg_id)[1:])


# =============================================================== 1. RMS-IQR baseline (segment 내부, 10샘플)
def rms_status(data):
    r = pd.DataFrame(index=data.index)
    for c in SIG:
        r[c] = np.sqrt(data[c].pow(2).groupby([data.source, data.segment_id])
                       .transform(lambda s: s.rolling(RMS_WINDOW, min_periods=RMS_WINDOW).mean()))
    n = r[data.source == "normal"]
    q1, q3 = n.quantile(0.25), n.quantile(0.75)
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    out = ((r < lo) | (r > hi)).any(axis=1)
    ok = r.notna().all(axis=1)
    return np.select([~ok, out], ["RMS_not_evaluable", "RMS_detected"], "RMS_missed")


# =============================================================== 2. 신호 처리 도구
def smooth(y):
    return pd.Series(y).rolling(SMOOTH, center=True, min_periods=1).median().to_numpy()


def detect_peaks(y, prominence):
    p, _ = find_peaks(smooth(y), distance=PEAK_DISTANCE, prominence=prominence)
    return p


def acf_period(y):
    """팀 estimate_period_acf와 동일: lag 12~22에서 상관 최대"""
    lo, hi = PERIOD_SEARCH
    if len(y) <= hi:
        return np.nan, np.nan
    best = (np.nan, -np.inf)
    for lag in range(lo, hi + 1):
        a, b = y[:-lag], y[lag:]
        if len(a) < 5 or a.std() == 0 or b.std() == 0:
            continue
        c = np.corrcoef(a, b)[0, 1]
        if c > best[1]:
            best = (lag, c)
    return (best[0] / FS, best[1]) if np.isfinite(best[1]) else (np.nan, np.nan)


def acf_at(y, lag):
    if len(y) <= lag + 4:
        return np.nan
    a, b = y[:-lag], y[lag:]
    return np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan


def return_error(y, idx, lag, scale):
    """1.7초 복귀 오차: unit 내 t에 대해 |y[t+lag]-y[t]| 평균 / scale (t+lag가 segment 안에 있을 때만)"""
    idx = idx[idx + lag < len(y)]
    if len(idx) == 0 or not np.isfinite(scale) or scale <= 0:
        return np.nan
    return float(np.mean(np.abs(y[idx + lag] - y[idx])) / scale)


def z_shape(x, n=PHASE_POINTS):
    """07_01_Feature normalize_cycle_shape와 동일"""
    y = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), np.asarray(x, float))
    sd = y.std(ddof=1)
    return (y - y.mean()) / sd if sd > 0 else None


class Template:
    """Normal full cycle z-shape 평균 (07과 동일). partial 매칭용으로 주기함수로도 사용"""
    def __init__(self, cycles):
        w = np.vstack([z for z in (z_shape(c) for c in cycles) if z is not None])
        t = w.mean(0)
        self.shape = (t - t.mean()) / t.std(ddof=1)            # 32점, phase 0~1 (양 끝 = peak)
        self.grid = np.linspace(0, 1, PHASE_POINTS)[:-1]       # 주기함수용 (마지막 점 = 첫 점)
        self.values = self.shape[:-1]
        self.ptp = np.ptp(self.values)

    def bank(self, L, period):
        off = np.arange(int(round(period * PHASE_STEPS))) / PHASE_STEPS
        ph = ((off[:, None] + np.arange(L)[None, :]) / period) % 1.0
        return np.interp(ph, self.grid, self.values, period=1.0)        # (J, L)


def match(Y, T):
    """Y (N,L) 각 행을 template bank T (J,L)의 최적 위상에 맞춤 → corr, shape_rmse, amplitude, offset_ratio"""
    Yc = Y - Y.mean(1, keepdims=True)
    Tc = T - T.mean(1, keepdims=True)
    den = np.linalg.norm(Yc, axis=1)[:, None] * np.linalg.norm(Tc, axis=1)[None, :]
    R = np.divide(Yc @ Tc.T, den, out=np.full((len(Y), len(T)), np.nan), where=den > 0)
    j = np.nanargmax(np.nan_to_num(R, nan=-2), axis=1)
    r = R[np.arange(len(Y)), j]
    Tm = T[j]
    yp, tp = np.ptp(Y, 1), np.ptp(Tm, 1)
    ys = Y.std(1, ddof=1, keepdims=True)
    yz = (Y - Y.mean(1, keepdims=True)) / np.where(ys > 0, ys, np.nan)
    tz = (Tm - Tm.mean(1, keepdims=True)) / Tm.std(1, ddof=1, keepdims=True)
    shape_rmse = np.sqrt(np.mean((yz - tz) ** 2, axis=1))
    tmc = Tm - Tm.mean(1, keepdims=True)
    a = (Yc * tmc).sum(1) / (tmc ** 2).sum(1)
    b = Y.mean(1) - a * Tm.mean(1)                     # template은 1주기 평균 0 → b = 1주기 평균 수준
    full_amp = yp / tp * Template_PTP[0]                # 1주기 환산 peak-to-peak
    offset_ratio = b / np.where(full_amp > 0, full_amp, np.nan)
    return r, shape_rmse, full_amp, offset_ratio


Template_PTP = [np.nan]


# =============================================================== 3. Normal에서 파라미터 결정 (freeze)
def segments(data, src):
    d = data[data.source == src]
    return {s: g.sort_values("pos_in_seg") for s, g in d.groupby("segment_id", sort=True)}


def freeze_params(normal_segs):
    P = {}
    # (1) 주기: segment별 ACF 주기 → 중앙값
    per = np.array([acf_period(g[ANCHOR].to_numpy(float))[0] for g in normal_segs.values()])
    per = per[np.isfinite(per)]
    P["period_sec_median"] = float(np.median(per))
    P["return_lag_samples"] = int(round(P["period_sec_median"] * FS))
    # (2) detector prominence: 기존 adaptive(0.5*std)의 Normal 중앙값으로 고정 (팀 5.EDA_2와 동일)
    proms = [max(float(pd.Series(smooth(g[ANCHOR].to_numpy(float))).std()) * 0.5, np.finfo(float).eps)
            for g in normal_segs.values() if len(g) >= DETECTOR_MIN_ROWS]
    P["peak_prominence"] = float(np.median(proms))
    # (3) Normal full cycle 수집
    cyc = []
    for s, g in normal_segs.items():
        y = g[ANCHOR].to_numpy(float)
        pk = detect_peaks(y, P["peak_prominence"])
        for a, b in zip(pk[:-1], pk[1:]):
            cyc.append((s, a, b, y[a:b + 1]))
    dur = np.array([(b - a) for _, a, b, _ in cyc])
    P["cycle_len_samples_mean"] = float(dur.mean())
    P["cycle_duration_range_sec"] = [float(dur.min() / FS - 0.1), float(dur.max() / FS + 0.1)]  # ±1 sample(0.1초 해상도)
    P["n_normal_full_cycles"] = len(cyc)
    # (4) Global 분석 최소 길이: 이 길이 이상 Normal segment의 99%에서 full cycle 1개 이상 검출
    lens = np.array([len(g) for g in normal_segs.values()])
    ncyc = np.array([max(len(detect_peaks(g[ANCHOR].to_numpy(float), P["peak_prominence"])) - 1, 0) for g in normal_segs.values()])
    P["global_min_len"] = next(L for L in range(18, 51) if (ncyc[lens >= L] >= 1).mean() >= COVERAGE_TARGET)
    return P, cyc


def unit_metrics(units, template, period, lag):
    """full: 07 방식(peak 정렬 32점 z-shape), partial: 주기 template 최적 위상 매칭"""
    res = [None] * len(units)
    by_len = {}
    for i, u in enumerate(units):
        if u.get("kind", "partial") == "full":
            y = u["y"][u["idx"]]
            z = z_shape(y)
            amp = float(np.ptp(y))
            res[i] = dict(template_corr=float(np.corrcoef(z, template.shape)[0, 1]) if z is not None else np.nan,
                        shape_rmse=float(np.sqrt(np.mean((z - template.shape) ** 2))) if z is not None else np.nan,
                        amplitude=amp, offset_ratio=float(y[:-1].mean() / amp) if amp > 0 else np.nan,
                        return_error=return_error(u["y"], u["idx"][:-1], lag, amp))
        else:
            by_len.setdefault(len(u["idx"]), []).append(i)
    for L, ids in by_len.items():
        Y = np.vstack([units[i]["y"][units[i]["idx"]] for i in ids])
        r, sr, amp, off = match(Y, template.bank(L, period))
        for k, i in enumerate(ids):
            u = units[i]
            res[i] = dict(template_corr=r[k], shape_rmse=sr[k], amplitude=amp[k], offset_ratio=off[k], return_error=return_error(u["y"], u["idx"], lag, amp[k]))
    return res


def thresholds(df):
    q = lambda c, p: float(df[c].quantile(p)) if df[c].notna().any() else np.nan
    return dict(corr_min=q("template_corr", Q_LO), shape_rmse_max=q("shape_rmse", Q_HI),
                amp_min=q("amplitude", Q_LO), amp_max=q("amplitude", Q_HI),
                abs_offset_max=float(df["offset_ratio"].abs().quantile(Q_HI)),
                return_error_max=q("return_error", Q_HI))


def normal_partial_reference(normal_segs, templates, P, L):
    """길이 L Normal 부분수열 전체(교차 template) → 길이별 threshold + shuffle null 통과율"""
    rng = np.random.default_rng(SEED)
    rows, null_rows = [], []
    for s, g in normal_segs.items():
        y = g[ANCHOR].to_numpy(float)
        if len(y) < L:
            continue
        tpl = templates[seg_num(s) % 2 == 0]            # 다른 parity segment로 만든 template
        for st in range(0, len(y) - L + 1):
            rows.append((tpl, y, np.arange(st, st + L)))
    out = []
    for parity in (True, False):
        sel = [(y, idx) for tpl, y, idx in rows if tpl is templates[parity]]
        if not sel:
            continue
        m = unit_metrics([dict(y=y, idx=idx) for y, idx in sel], templates[parity],
                         P["cycle_len_samples_mean"], P["return_lag_samples"])
        out += m
        Y = np.vstack([y[idx] for y, idx in sel])
        Ys = np.apply_along_axis(rng.permutation, 1, Y)
        null_rows.append(match(Ys, templates[parity].bank(L, P["cycle_len_samples_mean"]))[0])
    ref = pd.DataFrame(out)
    th = thresholds(ref)
    null_pass = float((np.concatenate(null_rows) >= th["corr_min"]).mean())
    return th, null_pass, len(ref)


# =============================================================== 4. Unit 구성·판정
def build_units(segs, src, P, template_for, global_th, full_th, partial_th, partial_min):
    """segment별로 global / partial / not_evaluable 판정 후 unit 목록 생성"""
    units, seg_rows = [], []
    lag, period = P["return_lag_samples"], P["cycle_len_samples_mean"]
    for s, g in segs.items():
        y = g[ANCHOR].to_numpy(float)
        L = len(y)
        rows0 = g["source_row"].to_numpy()
        tpl = template_for(s)
        seg = dict(source=src, segment_id=s, seg_len=L, source_row_start=int(rows0[0]),
                source_row_end=int(rows0[-1]))
        if L >= P["global_min_len"]:
            seg["analysis_type"] = "global"
            per, acfv = acf_period(y)
            seg_ret = return_error(y, np.arange(L), lag, np.ptp(y))
            pk = detect_peaks(y, P["peak_prominence"])
            seg.update(seg_period=per, seg_acf=acfv, seg_acf_lag17=acf_at(y, lag),
                    seg_return_error=seg_ret, seg_n_full_cycles=max(len(pk) - 1, 0))
            seg["seg_period_broken"] = bool(
                not (global_th["period_min"] <= per <= global_th["period_max"]) if np.isfinite(per) else True
            ) or bool(acfv < global_th["acf_min"]) or bool(seg_ret > global_th["return_error_max"]) \
            or seg["seg_n_full_cycles"] == 0
            if len(pk) < 2:
                units.append(dict(seg=seg, status="global_no_cycle", cycle_id=0,
                                idx=np.arange(L), y=y, rows=rows0, duration=np.nan, th=None))
            for k, (a, b) in enumerate(zip(pk[:-1], pk[1:]), start=1):
                units.append(dict(seg=seg, status="full_cycle", cycle_id=k, kind="full",
                                idx=np.arange(a, b + 1), y=y, rows=rows0, duration=(b - a) / FS, th=full_th,
                                start_sec=float(g["elapsed_sec"].iloc[a])))
        elif L >= partial_min:
            seg.update(analysis_type="partial", seg_period_broken=np.nan)
            units.append(dict(seg=seg, status="partial", cycle_id=0,
                            idx=np.arange(L), y=y, rows=rows0, duration=(L - 1) / FS, th=partial_th.get(L)))
        else:
            seg.update(analysis_type="not_evaluable", seg_period_broken=np.nan)
            units.append(dict(seg=seg, status="not_evaluable", cycle_id=0,
                            idx=np.arange(L), y=y, rows=rows0, duration=np.nan, th=None))
        seg_rows.append(seg)

    # metric 계산 (template은 segment별로 다를 수 있어 template 단위로 묶음)
    need = [i for i, u in enumerate(units) if u["status"] in ("full_cycle", "partial")]
    groups = {}
    for i in need:
        groups.setdefault(id(template_for(units[i]["seg"]["segment_id"])), []).append(i)
    for _, ids in groups.items():
        tpl = template_for(units[ids[0]]["seg"]["segment_id"])
        ms = unit_metrics([units[i] for i in ids], tpl, period, lag)
        for i, m in zip(ids, ms):
            units[i].update(m)

    rec = []
    for u in units:
        seg, th = u["seg"], u["th"]
        r = dict(source=src, segment_id=seg["segment_id"], cycle_id=u["cycle_id"],
                cycle_start_sec=u.get("start_sec", np.nan),
                source_row_start=int(u["rows"][u["idx"][0]]), source_row_end=int(u["rows"][u["idx"][-1]]),
                status=u["status"], n_samples=len(u["idx"]), cycle_duration=u["duration"],
                period=seg.get("seg_period", np.nan), seg_acf=seg.get("seg_acf", np.nan),
                return_error=u.get("return_error", np.nan), template_corr=u.get("template_corr", np.nan),
                shape_rmse=u.get("shape_rmse", np.nan), amplitude=u.get("amplitude", np.nan),
                offset_ratio=u.get("offset_ratio", np.nan))
        f_period = f_form = False
        period_tested = False
        if u["status"] == "global_no_cycle":
            f_period, period_tested = True, True
        elif u["status"] == "full_cycle":
            period_tested = True
            lo, hi = P["cycle_duration_range_sec"]
            f_period = (not lo <= u["duration"] <= hi) or bool(seg["seg_period_broken"]) \
                or bool(r["return_error"] > th["return_error_max"])
        elif u["status"] == "partial" and th is not None:
            period_tested = np.isfinite(r["return_error"])
            f_period = bool(period_tested and r["return_error"] > th["return_error_max"])
        if u["status"] in ("full_cycle", "partial") and th is not None:
            f_form = bool(r["template_corr"] < th["corr_min"]) or bool(r["shape_rmse"] > th["shape_rmse_max"]) \
                or bool(not th["amp_min"] <= r["amplitude"] <= th["amp_max"]) \
                or bool(abs(r["offset_ratio"]) > th["abs_offset_max"])
            r.update(flag_shape=bool(r["template_corr"] < th["corr_min"] or r["shape_rmse"] > th["shape_rmse_max"]),
                    flag_amplitude=bool(not th["amp_min"] <= r["amplitude"] <= th["amp_max"]
                                        or abs(r["offset_ratio"]) > th["abs_offset_max"]))
        r["flag_period"] = f_period
        if u["status"] == "not_evaluable":
            v = "not_evaluable"
        elif f_period:
            v = "period_broken"
        elif period_tested:
            v = "period_kept_form_broken" if f_form else "period_kept_normal_like"
        else:
            v = "period_unknown_form_broken" if f_form else "period_unknown_normal_like"
        r["verdict"] = v
        rec.append(r)
    return pd.DataFrame(rec), pd.DataFrame(seg_rows)


# =============================================================== main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, data_dir = load()
    data["rms_status"] = rms_status(data)
    nseg, oseg = segments(data, "normal"), segments(data, "abnormal")

    # ---- Normal에서 freeze
    P, ncyc = freeze_params(nseg)
    tpl_all = Template([c for *_, c in ncyc])
    Template_PTP[0] = tpl_all.ptp
    tpl_even = Template([c for s, *_, c in ncyc if seg_num(s) % 2 == 0])
    tpl_odd = Template([c for s, *_, c in ncyc if seg_num(s) % 2 == 1])
    cross = {True: tpl_odd, False: tpl_even}           # 짝수 segment는 홀수로 만든 template과 비교

    # Global segment-level threshold (Normal global segment)
    gl = []
    for s, g in nseg.items():
        y = g[ANCHOR].to_numpy(float)
        if len(y) >= P["global_min_len"]:
            per, acfv = acf_period(y)
            gl.append(dict(per=per, acf=acfv, ret=return_error(y, np.arange(len(y)), P["return_lag_samples"], np.ptp(y))))
    gl = pd.DataFrame(gl)
    global_th = dict(period_min=float(gl.per.min() - 0.1), period_max=float(gl.per.max() + 0.1),
                     acf_min=float(gl.acf.quantile(Q_LO)), return_error_max=float(gl.ret.quantile(Q_HI)))

    # Full cycle threshold (Normal full cycles, 교차 template)
    fc = []
    for parity in (True, False):
        sel = [dict(y=nseg[s][ANCHOR].to_numpy(float), idx=np.arange(a, b + 1), kind="full")
               for s, a, b, _ in ncyc if (seg_num(s) % 2 == 0) == parity]
        fc += unit_metrics(sel, cross[parity], P["cycle_len_samples_mean"], P["return_lag_samples"])
    full_th = thresholds(pd.DataFrame(fc))

    # Partial: 길이별 threshold + 최소 길이(shuffle null 통과율 <= 5%)
    partial_th, null_tab = {}, []
    for L in range(5, P["global_min_len"]):
        th, null_pass, n_ref = normal_partial_reference(nseg, cross, P, L)
        partial_th[L] = th
        null_tab.append(dict(L=L, n_normal_ref=n_ref, null_pass_rate=null_pass, **th))
    null_tab = pd.DataFrame(null_tab)
    ok = null_tab[null_tab.null_pass_rate <= NULL_PASS_MAX].L
    P["partial_min_len"] = int(ok.min()) if len(ok) else P["global_min_len"]
    partial_th = {L: t for L, t in partial_th.items() if L >= P["partial_min_len"]}

    # ---- 적용: Normal(교차 template, 참고용 오경보) / Outlier(전체 Normal template)
    n_units, n_segs = build_units(nseg, "normal", P, lambda s: cross[seg_num(s) % 2 == 0],
                                global_th, full_th, partial_th, P["partial_min_len"])
    o_units, o_segs = build_units(oseg, "abnormal", P, lambda s: tpl_all,
                                global_th, full_th, partial_th, P["partial_min_len"])
    units = pd.concat([n_units, o_units], ignore_index=True)
    data = data.reset_index(drop=True)

    # ---- 행 단위 merge table
    data["cycle_id"] = pd.NA
    data["cycle_eval"] = "not_evaluable"
    data["cycle_verdict"] = "not_evaluable"
    key = data.set_index(["source", "source_row"]).index
    pos = pd.Series(np.arange(len(data)), index=key)
    segs_all = pd.concat([n_segs, o_segs], ignore_index=True)
    for _, sg in segs_all[segs_all.analysis_type == "global"].iterrows():      # 먼저 segment 수준 판정
        ii = pos.loc[[(sg.source, r) for r in range(sg.source_row_start, sg.source_row_end + 1)]].to_numpy()
        data.loc[ii, "cycle_eval"] = "global_only"
        data.loc[ii, "cycle_verdict"] = "period_broken" if sg.seg_period_broken else "period_kept_form_untested"
    for u in units.sort_values(["source", "source_row_start"], ascending=[True, False]).itertuples():
        rows = range(u.source_row_start, u.source_row_end + (1 if u.status != "full_cycle" else 0))
        if u.status == "full_cycle":
            last = units[(units.source == u.source) & (units.segment_id == u.segment_id)
                        & (units.status == "full_cycle")].source_row_end.max()
            if u.source_row_end == last:
                rows = range(u.source_row_start, u.source_row_end + 1)
        ii = pos.loc[[(u.source, r) for r in rows]].to_numpy()
        data.loc[ii, "cycle_id"] = u.cycle_id
        data.loc[ii, "cycle_eval"] = u.status
        data.loc[ii, "cycle_verdict"] = u.verdict
    data["cycle_detected"] = np.where(data.cycle_eval == "not_evaluable", pd.NA,
                                    data.cycle_verdict.isin(VERDICT_DETECTED))

    # unit RMS 그룹 (unit 내 판정 가능 행 다수결, 동률은 detected)
    rs = data[["source", "source_row", "rms_status"]].set_index(["source", "source_row"]).rms_status
    def unit_rms(u):
        v = rs.loc[[(u.source, r) for r in range(u.source_row_start, u.source_row_end + 1)]]
        d, m = (v == "RMS_detected").sum(), (v == "RMS_missed").sum()
        g = "RMS_not_evaluable" if d + m == 0 else ("RMS_detected" if d >= m else "RMS_missed")
        return pd.Series(dict(rms_detected_rows=d, rms_missed_rows=m,
                            rms_not_evaluable_rows=(v == "RMS_not_evaluable").sum(), rms_group=g))
    units = pd.concat([units, units.apply(unit_rms, axis=1)], axis=1)
    units.loc[units.source == "normal", "rms_group"] = "Normal"

    # ---- 07 cycle_feature_table(adaptive detector)과 대응: 같은 segment·시작시각이면 07의 cycle_id 기록
    units["cycle_id_07"] = pd.NA
    t07 = next((b / "KAMP" / "07_Feature_Engineering" / "Data" / "cycle_feature_table_all.csv"
                for b in [HERE, *HERE.parents] if (b / "KAMP" / "07_Feature_Engineering" / "Data"
                                                    / "cycle_feature_table_all.csv").exists()), None)
    if t07 is not None:
        f07 = pd.read_csv(t07, usecols=["source", "segment_id", "cycle_id", "cycle_start_sec"])
        f07["k"] = f07.cycle_start_sec.round(1)
        m07 = f07.set_index(["source", "segment_id", "k"]).cycle_id
        fu = units.status == "full_cycle"
        keys = list(zip(units.loc[fu, "source"], units.loc[fu, "segment_id"], units.loc[fu, "cycle_start_sec"].round(1)))
        units.loc[fu, "cycle_id_07"] = [m07.get(k, pd.NA) for k in keys]
        P["match_with_07"] = {src: f"{int(units[fu & (units.source == src)].cycle_id_07.notna().sum())}"
                                f"/{int((fu & (units.source == src)).sum())} (07 total {int((f07.source == src).sum())})"
                            for src in ("normal", "abnormal")}

    # ---- 산출물
    cols_u = ["source", "segment_id", "cycle_id", "cycle_id_07", "cycle_start_sec", "source_row_start", "source_row_end", "status", "n_samples",
            "cycle_duration", "period", "seg_acf", "return_error", "template_corr", "shape_rmse", "amplitude",
            "offset_ratio", "flag_period", "flag_shape", "flag_amplitude", "verdict", "rms_group",
            "rms_detected_rows", "rms_missed_rows", "rms_not_evaluable_rows"]
    units[cols_u].to_csv(OUT / "cycle_units.csv", index=False)
    data[["source", "segment_id", "source_row", "pos_in_seg", "seg_len", "cycle_id", "cycle_eval",
        "cycle_verdict", "cycle_detected", "rms_status"]].to_csv(OUT / "cycle_rows.csv", index=False)

    ou = units[units.source == "abnormal"]
    od = data[data.source == "abnormal"]
    seg_sum = (o_segs.set_index("segment_id")
            .join(ou.groupby("segment_id").verdict.agg(lambda v: v.value_counts().to_dict()).rename("unit_verdicts"))
            .join(od.groupby("segment_id").rms_status.value_counts().unstack(fill_value=0))
            .join(od.groupby("segment_id").cycle_verdict.agg(lambda v: v.value_counts().idxmax()).rename("main_row_verdict")))
    seg_sum.reset_index().to_csv(OUT / "segment_summary_outlier.csv", index=False)

    comp_units = units[units.status.isin(["full_cycle", "partial"])]
    met = ["cycle_duration", "return_error", "template_corr", "shape_rmse", "amplitude", "offset_ratio"]
    comp = comp_units.groupby(["status", "rms_group"])[met].median().round(3)
    comp = comp.join(comp_units.groupby(["status", "rms_group"]).size().rename("n_units"))
    comp = comp.join(comp_units.groupby(["status", "rms_group"]).verdict
                    .apply(lambda v: v.isin(VERDICT_DETECTED).mean()).rename("detected_rate").round(3))
    comp.to_csv(OUT / "group_comparison.csv")

    # ---- 회의용 숫자 (행 기준, Outlier 600)
    rms_eval = od.rms_status != "RMS_not_evaluable"
    cyc_eval = od.cycle_eval != "not_evaluable"
    cyc_det = od.cycle_detected.fillna(False).astype(bool)
    nd = data[data.source == "normal"]
    meet = pd.DataFrame([
        dict(method="RMS-IQR (1초, segment 내부)", total_abnormal=len(od), evaluable=int(rms_eval.sum()),
            detected=int((od.rms_status == "RMS_detected").sum()), missed=int((od.rms_status == "RMS_missed").sum()),
            not_evaluable=int((~rms_eval).sum()),
            normal_false_alarm_rows=f"{(nd.rms_status == 'RMS_detected').sum()}/{(nd.rms_status != 'RMS_not_evaluable').sum()}"),
        dict(method="AI2 Cycle (global+partial)", total_abnormal=len(od), evaluable=int(cyc_eval.sum()),
            detected=int(cyc_det.sum()), missed=int((cyc_eval & ~cyc_det).sum()), not_evaluable=int((~cyc_eval).sum()),
            normal_false_alarm_rows=f"{nd.cycle_detected.fillna(False).astype(bool).sum()}/{(nd.cycle_eval != 'not_evaluable').sum()}"),
    ])
    cross_tab = pd.crosstab(od.rms_status, od.cycle_eval.where(~cyc_eval, np.where(cyc_det, "cycle_detected", "cycle_missed")),
                            margins=True)
    meet.to_csv(OUT / "meeting_numbers.csv", index=False)
    cross_tab.to_csv(OUT / "meeting_crosstab.csv")

    P.update(global_thresholds=global_th, full_cycle_thresholds=full_th, data_dir=data_dir,
            normal_segments=len(nseg), abnormal_segments=len(oseg))
    with open(OUT / "frozen_params.json", "w", encoding="utf-8") as fp:
        json.dump(P, fp, ensure_ascii=False, indent=2, default=float)
    null_tab.to_csv(OUT / "partial_thresholds_by_length.csv", index=False)

    # ---- 출력
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("[Frozen params]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in P.items()
                            if k not in ("global_thresholds", "full_cycle_thresholds", "data_dir")})
    print("[Global th]", {k: round(v, 4) for k, v in global_th.items()})
    print("[Full-cycle th]", {k: round(v, 4) for k, v in full_th.items()})
    print("\n[Partial 최소 길이 검증]\n", null_tab[["L", "n_normal_ref", "corr_min", "null_pass_rate"]].head(6).round(3).to_string(index=False))
    print("\n[회의용 숫자]\n", meet.to_string(index=False))
    print("\n[RMS x Cycle 교차표 (Abnormal 행)]\n", cross_tab.to_string())
    print("\n[Abnormal segment 요약]\n", seg_sum[["seg_len", "analysis_type", "seg_period", "seg_acf", "seg_return_error",
                                            "seg_n_full_cycles", "seg_period_broken", "unit_verdicts",
                                            "main_row_verdict"]].to_string())
    print("\n[그룹 비교]\n", comp.to_string())


if __name__ == "__main__":
    main()
