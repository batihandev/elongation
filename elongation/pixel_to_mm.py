import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import medfilt, savgol_filter

def calculate_pixel_distance(p1, p2):
    if isinstance(p1, dict):
        x1, y1 = p1['x'], p1['y']
    else:
        x1, y1 = p1[0], p1[1]

    if isinstance(p2, dict):
        x2, y2 = p2['x'], p2['y']
    else:
        x2, y2 = p2[0], p2[1]

    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)

def create_pixel_to_mm_csv(base_name, p1, p2, real_distance_mm):
    pixel_distance = calculate_pixel_distance(p1, p2)
    csv_path = f"results/pixel_to_mm_{base_name}.csv"
    df = pd.DataFrame({
        'pixel_distance': [pixel_distance],
        'real_distance_mm': [real_distance_mm],
        'pixel_to_mm_ratio': [real_distance_mm / pixel_distance if pixel_distance > 0 else None]
    })
    df.to_csv(csv_path, index=False)
    return csv_path

def plot_percent_with_mm_axis(data_csv_path, pixel_to_mm_csv_path, output_plot_path, progress_callback=None):
    def notify_progress(p, msg):
        if progress_callback:
            progress_callback(p, msg)
        else:
            print(f"{int(p*100)}% - {msg}")

    notify_progress(0.0, "Starting plot with percent and mm axis...")

    df = pd.read_csv(data_csv_path)
    ptomm = pd.read_csv(pixel_to_mm_csv_path)
    ratio = ptomm['pixel_to_mm_ratio'].iloc[0]

    # Compute total initial gauge length at %100: ref_bottom_px - ref_top_px (same for all rows assumed)
    initial_gauge_length_px = abs(df.loc[0, "ref_bottom_px"] - df.loc[0, "ref_top_px"])
    mm_per_px = ratio
    mm_per_percent = (initial_gauge_length_px * mm_per_px) / 100.0

    # Add mm columns
    df["elongation_mm"] = (df["elongation_percent"] - 100.0) * mm_per_percent
    df["elongation_median_mm"] = (df["elongation_median"] - 100.0) * mm_per_percent
    df["elongation_smoothed_mm"] = (df["elongation_smoothed"] - 100.0) * mm_per_percent
    df["elongation_monotonic_mm"] = (df["elongation_monotonic"] - 100.0) * mm_per_percent

    # Save extended version
    df.to_csv(data_csv_path, index=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["timestamp_s"], df["elongation_percent"], label="Raw %", linestyle="--", alpha=0.3)
    ax1.plot(df["timestamp_s"], df["elongation_median"], label="Median Filtered %", linestyle="--", color="orange", linewidth=1)
    ax1.plot(df["timestamp_s"], df["elongation_smoothed"], label="LOWESS Smoothed %", color="blue", linewidth=1, alpha=0.6)
    ax1.plot(df["timestamp_s"], df["elongation_monotonic"], label="Final Monotonic %", color="green", linewidth=2)

    # Yield point logic: always use max 2nd derivative (like analyze_elongation.py)
    y_curve = df["elongation_monotonic"].to_numpy()
    t_curve = df["timestamp_s"].to_numpy()
    dy_dt = np.gradient(y_curve, t_curve)
    d2y_dt2 = np.gradient(dy_dt, t_curve)
    yield_index = int(np.argmax(d2y_dt2))
    yield_time = df.loc[yield_index, "timestamp_s"] if yield_index is not None else None
    yield_percent = df.loc[yield_index, "elongation_monotonic"] if yield_index is not None else None
    yield_mm = (yield_percent - 100.0) * mm_per_percent if yield_percent is not None else None
    if yield_time is not None and yield_mm is not None and yield_percent is not None:
        ax1.axvline(yield_time, color='red', linestyle='--', label='Yield Point')
        ax1.scatter([yield_time], [float(yield_percent)], color='red')

    ax1.set_title("Rebar Elongation Over Time (mm)")
    ax1.set_xlabel("Time (s)")
    # Use percent y-ticks but relabel as mm
    percent_ticks = ax1.get_yticks()
    mm_labels = [(v - 100.0) * mm_per_percent for v in percent_ticks]
    ax1.set_yticks(percent_ticks)
    ax1.set_yticklabels([f"{v:.3f} mm" for v in mm_labels])
    ax1.set_ylabel("Elongation (mm)")
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()
    # Add number of frames annotation to main plot
    ax1.text(0.99, 0.01, f"Frames used: {len(df)}", ha='right', va='bottom',
             transform=ax1.transAxes, fontsize=10, color='gray')
    plt.savefig(output_plot_path)
    plt.close()

    notify_progress(1.0, f"Plot saved to: {output_plot_path}")
