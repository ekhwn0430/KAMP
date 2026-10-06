def segment_peak_diagnostics(df, dataset_name):
    rows = []
    for segment_id, group in df.groupby("segment_id", sort=False):
        values = group[CYCLE_ANCHOR].to_numpy(dtype=float)
        smooth = pd.Series(values).rolling(3, center=True, min_periods=1).median()
        ai2_std = float(smooth.std()) if len(smooth) > 1 else 0.0
        prominence = max(ai2_std * 0.5, np.finfo(float).eps)
        eligible = len(group) >= MIN_SEGMENT_ROWS and np.isfinite(values).all()
        peaks = find_peaks(
            smooth.to_numpy(), distance=CYCLE_MIN_PEAK_DISTANCE_SAMPLES,
            prominence=prominence,
        )[0] if eligible else np.array([], dtype=int)
        rows.append({
            "dataset": dataset_name,
            "segment_id": int(segment_id),
            "n_samples": len(group),
            "AI2_std": ai2_std,
            "current_prominence": prominence,
            "peak_count": len(peaks),
            "candidate_cycle_count": max(len(peaks) - 1, 0),
            "eligible_by_length": eligible,
        })
    return pd.DataFrame(rows)

normal_segment_diagnostics = segment_peak_diagnostics(normal, "Normal")
outlier_segment_diagnostics = segment_peak_diagnostics(outlier, "Outlier")
segment_diagnostics = pd.concat([normal_segment_diagnostics, outlier_segment_diagnostics], ignore_index=True)

for name, diagnostics in [("Normal", normal_segment_diagnostics), ("Outlier", outlier_segment_diagnostics)]:
    print(f"{name}: segment-level detector diagnostics (all segments)")
    display(diagnostics)
    eligible = diagnostics.loc[diagnostics["eligible_by_length"]]
    print(f"{name}: eligible segment counts; segments shorter than {MIN_SEGMENT_ROWS} are skipped by the existing detector")
    display(eligible[["AI2_std", "current_prominence", "peak_count"]].describe().T)

outlier_eligible = outlier_segment_diagnostics.loc[outlier_segment_diagnostics["eligible_by_length"]].copy()
outlier_eligible["peak_group"] = pd.cut(
    outlier_eligible["peak_count"], bins=[-1, 0, 1, np.inf], labels=["peak 0", "peak 1", "peak 2+"],
)
print("Outlier eligible segment counts by peak group:")
display(outlier_eligible["peak_group"].value_counts().reindex(["peak 0", "peak 1", "peak 2+"], fill_value=0).rename("segment_count").to_frame())
print("Outlier eligible segment details:")
display(outlier_eligible[["segment_id", "n_samples", "AI2_std", "current_prominence", "peak_count"]])

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for ax, metric, title in zip(axes, ["AI2_std", "current_prominence", "peak_count"], ["AI2 rolling-median std", "Adaptive prominence", "Detected peaks"]):
    plot_data = [
        normal_segment_diagnostics.loc[normal_segment_diagnostics["eligible_by_length"], metric],
        outlier_segment_diagnostics.loc[outlier_segment_diagnostics["eligible_by_length"], metric],
    ]
    ax.boxplot(plot_data, tick_labels=["Normal", "Outlier"], showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(metric)
plt.show()


peak_categories = [("peak 0", 0), ("peak 1", 1), ("peak 2+", 2)]
fig, axes = plt.subplots(len(peak_categories), 1, figsize=(13, 10), squeeze=False, constrained_layout=True)
for row_plot, (category, minimum_peaks) in enumerate(peak_categories):
    ax = axes[row_plot, 0]
    candidates = outlier_eligible.loc[outlier_eligible["peak_group"] == category]
    if candidates.empty:
        ax.text(0.5, 0.5, f"No eligible Outlier segment in {category}", ha="center", va="center")
        ax.set_axis_off()
        continue
    target_std = candidates["AI2_std"].median()
    chosen = candidates.iloc[(candidates["AI2_std"] - target_std).abs().argsort()[:1]]
    segment_id = int(chosen.iloc[0]["segment_id"])
    group = outlier.loc[outlier["segment_id"] == segment_id]
    raw = group[CYCLE_ANCHOR].to_numpy(dtype=float)
    smooth = pd.Series(raw).rolling(3, center=True, min_periods=1).median().to_numpy()
    threshold = float(chosen.iloc[0]["current_prominence"])
    peaks = find_peaks(smooth, distance=CYCLE_MIN_PEAK_DISTANCE_SAMPLES, prominence=threshold)[0]
    time = np.arange(len(group)) / SAMPLE_RATE_HZ
    ax.plot(time, raw, color="#9ba4a8", linewidth=0.9, label="AI2 raw")
    ax.plot(time, smooth, color="#176b70", linewidth=1.4, label="AI2 rolling median")
    ax.scatter(time[peaks], smooth[peaks], color="#bf4e3d", marker="x", s=65, label="detected peaks", zorder=3)
    ax.set_title(f"Outlier segment_id={segment_id}, length={len(group)}, peaks={len(peaks)}, prominence={threshold:.3g}")
    ax.set_xlabel("Time within segment (sec)")
    ax.set_ylabel("AI2_Current")
    ax.legend(loc="upper right")
plt.show()


FIXED_NORMAL_PROMINENCE = float(
    normal_segment_diagnostics.loc[normal_segment_diagnostics["eligible_by_length"], "current_prominence"].median()
)
print(
    "Frozen threshold rationale: median of per-segment Normal adaptive prominence "
    f"across eligible Normal segments = {FIXED_NORMAL_PROMINENCE:.6g} AI2 units."
)

def detect_peak_method(df, dataset_name, method_name, fixed_prominence=None):
    summary_rows = []
    duration_rows = []
    peak_locations = {}
    for segment_id, group in df.groupby("segment_id", sort=False):
        if len(group) < MIN_SEGMENT_ROWS:
            continue
        raw = group[CYCLE_ANCHOR].to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            continue
        smooth = pd.Series(raw).rolling(3, center=True, min_periods=1).median().to_numpy()
        threshold = (
            max(float(pd.Series(smooth).std()) * 0.5, np.finfo(float).eps)
            if fixed_prominence is None else fixed_prominence
        )
        peaks = find_peaks(smooth, distance=CYCLE_MIN_PEAK_DISTANCE_SAMPLES, prominence=threshold)[0]
        peak_locations[(dataset_name, int(segment_id), method_name)] = peaks
        durations = np.diff(group["elapsed_sec"].to_numpy()[peaks]) if len(peaks) > 1 else np.array([])
        summary_rows.append({
            "dataset": dataset_name,
            "method": method_name,
            "segment_id": int(segment_id),
            "peak_count": len(peaks),
            "candidate_cycle_count": max(len(peaks) - 1, 0),
        })
        duration_rows.extend({
            "dataset": dataset_name, "method": method_name, "segment_id": int(segment_id),
            "duration_sec": float(duration),
        } for duration in durations if duration > 0)
    return pd.DataFrame(summary_rows), pd.DataFrame(duration_rows), peak_locations

method_summaries = []
method_durations = []
method_peak_locations = {}
for dataset_name, df in [("Normal", normal), ("Outlier", outlier)]:
    for method_name, fixed_value in [("Current adaptive", None), ("Fixed Normal-derived", FIXED_NORMAL_PROMINENCE)]:
        summary, durations, locations = detect_peak_method(df, dataset_name, method_name, fixed_value)
        method_summaries.append(summary)
        method_durations.append(durations)
        method_peak_locations.update(locations)
method_summary = pd.concat(method_summaries, ignore_index=True)
method_duration_table = pd.concat(method_durations, ignore_index=True)

comparison_counts = method_summary.groupby(["dataset", "method"])[["peak_count", "candidate_cycle_count"]].sum().rename(columns={
    "peak_count": "total_peaks", "candidate_cycle_count": "total_candidate_cycles",
})
print("Adaptive vs Normal-derived fixed threshold: total peaks and peak-to-peak cycles")
display(comparison_counts)

def duration_stats(values):
    values = pd.Series(values, dtype=float)
    return pd.Series({
        "count": values.count(), "mean": values.mean(), "std": values.std(),
        "min": values.min(), "Q1": values.quantile(.25), "median": values.median(),
        "Q3": values.quantile(.75), "max": values.max(),
    })
print("Cycle duration distributions by dataset and detector:")
display(method_duration_table.groupby(["dataset", "method"])["duration_sec"].apply(duration_stats).unstack())


fig, axes = plt.subplots(3, 1, figsize=(13, 10), squeeze=False, constrained_layout=True)
for row_plot, (category, _) in enumerate(peak_categories):
    ax = axes[row_plot, 0]
    candidates = outlier_eligible.loc[outlier_eligible["peak_group"] == category]
    if candidates.empty:
        ax.text(0.5, 0.5, f"No eligible Outlier segment in {category}", ha="center", va="center")
        ax.set_axis_off()
        continue
    chosen_id = int(candidates.sort_values("segment_id").iloc[len(candidates) // 2]["segment_id"])
    group = outlier.loc[outlier["segment_id"] == chosen_id]
    raw = group[CYCLE_ANCHOR].to_numpy(dtype=float)
    smooth = pd.Series(raw).rolling(3, center=True, min_periods=1).median().to_numpy()
    adaptive = method_peak_locations[("Outlier", chosen_id, "Current adaptive")]
    fixed = method_peak_locations[("Outlier", chosen_id, "Fixed Normal-derived")]
    time = np.arange(len(group)) / SAMPLE_RATE_HZ
    ax.plot(time, raw, color="#aab0b3", linewidth=0.9, label="AI2 raw")
    ax.plot(time, smooth, color="#176b70", linewidth=1.3, label="AI2 rolling median")
    ax.scatter(time[adaptive], smooth[adaptive], marker="o", facecolors="none", edgecolors="#bf4e3d", s=65, label="adaptive peaks")
    ax.scatter(time[fixed], smooth[fixed], marker="x", color="#3359a5", s=65, label="fixed Normal-derived peaks")
    ax.set_title(f"Outlier segment_id={chosen_id}, length={len(group)} | adaptive={len(adaptive)}, fixed={len(fixed)}")
    ax.set_xlabel("Time within segment (sec)")
    ax.set_ylabel("AI2_Current")
    ax.legend(loc="upper right", ncol=2)
plt.show()
print("Peak-count increases under a fixed threshold are not treated as evidence of a better cycle detector; inspect alignment to visible waveform structure.")
