import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt, savgol_filter
import os
from tqdm import tqdm

def process_and_plot(
    df,
    output_csv="elongation_final.csv",
    plot_path="elongation_plot.png",
    min_elongation=100,
    max_elongation=140,
    progress_callback=None,
    cancel_event=None
):
    def notify_progress(p, msg):
        if progress_callback:
            progress_callback(p, msg)
        else:
            print(f"{int(p*100)}% - {msg}")

    notify_progress(0.85, f"🔎 Raw input shape: {df.shape}")

    df = df.loc[(df["elongation_percent"] >= min_elongation) & (df["elongation_percent"] <= max_elongation)].copy()
    notify_progress(0.90, f"✅ After valid elongation range filter ({min_elongation}–{max_elongation}%): {df.shape}")

    if df.empty:
        notify_progress(1.0, "❌ No data left after filtering.")
        return None, None, None

    df["elongation_median"] = medfilt(df["elongation_percent"].to_numpy(), kernel_size=5)

    length = len(df)
    window = min(21, length if length % 2 == 1 else length - 1)
    if window < 5:
        df["elongation_smoothed"] = df["elongation_median"]
    else:
        df["elongation_smoothed"] = savgol_filter(df["elongation_median"], window_length=window, polyorder=2)

    df["elongation_smoothed"] = pd.Series(df["elongation_smoothed"]).cummax()
    df["elongation_smoothed"] = pd.Series(df["elongation_smoothed"]).cummax()

    smoothed = df["elongation_smoothed"].values.copy()
    median_values = df["elongation_median"].values
    window_size = 5
    decay_strength = 0.1

    for i in range(len(smoothed) - 2, -1, -1):
        if cancel_event and cancel_event.is_set():
            notify_progress(1.0, "Processing cancelled by user.")
            break
        local_window = median_values[max(0, i - window_size):i + 1]
        local_median = np.median(local_window)
        if smoothed[i] > smoothed[i + 1]:
            smoothed[i] = smoothed[i + 1]
        if smoothed[i] - local_median > 0.3:
            allowed_drop = smoothed[i + 1] - decay_strength
            target = max(local_median, allowed_drop)
            smoothed[i] = min(smoothed[i], target)

    df["elongation_smoothed_corrected"] = smoothed
    df["strain_rate"] = df["elongation_smoothed_corrected"].diff() / df["timestamp_s"].diff()
    if cancel_event and cancel_event.is_set():
        notify_progress(1.0, "Processing cancelled by user.")
        return
    if df["strain_rate"].isna().all():
        notify_progress(1.0, "❌ Strain rate is all NaNs — not enough valid frames.")
        return df, None, None

    yield_index = df["strain_rate"].idxmax()
    if pd.isna(yield_index):
        notify_progress(1.0, "❌ Could not determine yield point.")
        return df, None, None

    yield_time = df.loc[yield_index, "timestamp_s"]
    yield_elongation = df.loc[yield_index, "elongation_smoothed_corrected"]

    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp_s"], df["elongation_percent"], label="Raw %", linestyle="--", alpha=0.3)
    plt.plot(df["timestamp_s"], df["elongation_median"], label="Median Filtered %", linestyle="--", color="orange", linewidth=1)
    plt.plot(df["timestamp_s"], df["elongation_smoothed"], label="Initial Smoothed %", color="blue", linewidth=1, alpha=0.6)
    plt.plot(df["timestamp_s"], df["elongation_smoothed_corrected"], label="Final Adjusted %", color="green", linewidth=2)

    if yield_time is not None:
        plt.axvline(yield_time, color='red', linestyle='--', label='Yield Point')
        plt.scatter([yield_time], [yield_elongation], color='red')

    plt.title("Rebar Elongation Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Elongation (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(plot_path)
    df.to_csv(output_csv, index=False)

    notify_progress(1.0, f"📊 Saved cleaned data to: {output_csv}, plot to: {plot_path}")
    return df, yield_time, yield_elongation

def main():
    input_csv = "elongation_data.csv"
    if not os.path.exists(input_csv):
        print(f"❌ Input not found: {input_csv}")
        return
    df = pd.read_csv(input_csv)
    process_and_plot(df)

if __name__ == "__main__":
    main()
