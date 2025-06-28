import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import os
from tqdm import tqdm
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    raise ImportError("statsmodels is required for LOWESS smoothing. Please install it with 'pip install statsmodels'.")
try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    raise ImportError("scikit-learn is required for isotonic regression. Please install it with 'pip install scikit-learn'.")

def process_and_plot(
    df,
    output_csv="elongation_final.csv",
    plot_path="elongation_plot.png",
    min_elongation=100,
    max_elongation=140,
    progress_callback=None,
    cancel_event=None,
    zoomed_plot_path=None
):
    # Always save the zoomed plot by default
    if zoomed_plot_path is None and plot_path.endswith('.png'):
        zoomed_plot_path = plot_path.replace('.png', '_zoomed.png')

    def notify_progress(p, msg):
        if progress_callback:
            progress_callback(p, msg)
        else:
            print(f"{int(p*100)}% - {msg}")

    notify_progress(0.85, f"🔎 Raw input shape: {df.shape}")

    df = df.loc[(df["elongation_percent"] >= min_elongation) & (df["elongation_percent"] <= max_elongation)].copy()
    df = df.reset_index(drop=True)
    notify_progress(0.90, f"✅ After valid elongation range filter ({min_elongation}–{max_elongation}%): {df.shape}")

    if df.empty:
        notify_progress(1.0, "❌ No data left after filtering.")
        return None, None, None

    df["elongation_median"] = medfilt(df["elongation_percent"].to_numpy(), kernel_size=5)

    # LOWESS smoothing
    frac = 0.05 if len(df) > 20 else 0.3  # Use larger window for small datasets
    smoothed = lowess(df["elongation_median"], df["timestamp_s"], frac=frac, return_sorted=False)
    df["elongation_smoothed"] = smoothed

    # Isotonic regression for monotonicity
    ir = IsotonicRegression(increasing=True)
    # Drop NaN/None and non-finite values for isotonic regression
    valid_mask = (~pd.isnull(df["elongation_smoothed"])) & (~pd.isnull(df["timestamp_s"]))
    x_valid = np.array(df.loc[valid_mask, "timestamp_s"], dtype=float)
    y_valid = np.array(df.loc[valid_mask, "elongation_smoothed"], dtype=float)
    finite_mask = np.isfinite(x_valid) & np.isfinite(y_valid)
    x_valid = x_valid[finite_mask]
    y_valid = y_valid[finite_mask]
    df["elongation_monotonic"] = np.nan
    if len(x_valid) >= 2:
        y_monotonic = ir.fit_transform(np.array(x_valid, dtype=float), np.array(y_valid, dtype=float))
        # Only fill back the valid, finite indices
        valid_indices = df.loc[valid_mask].index[finite_mask]
        df.loc[valid_indices, "elongation_monotonic"] = y_monotonic
    else:
        # Not enough data for isotonic regression, just copy smoothed values
        valid_indices = df.loc[valid_mask].index[finite_mask]
        df.loc[valid_indices, "elongation_monotonic"] = y_valid

    # Flatten the end if last N median values are flat
    N = 5
    if len(df) >= N and np.allclose(df["elongation_median"].values[-N:], df["elongation_median"].values[-1]):
        df["elongation_monotonic"].values[-N:] = df["elongation_median"].values[-1]

    # --- Yield Point Detection: Maximum Change in Slope (Second Derivative) ---
    # Use the monotonic curve for robust detection
    y_curve = df["elongation_monotonic"].to_numpy()
    t_curve = df["timestamp_s"].to_numpy()
    
    # Compute first and second derivatives
    dy_dt = np.gradient(y_curve, t_curve)
    d2y_dt2 = np.gradient(dy_dt, t_curve)

    # Yield point: index of maximum second derivative (sharpest change in slope)
    yield_index = int(np.argmax(d2y_dt2))
    yield_time = df.loc[yield_index, "timestamp_s"] if yield_index is not None else None
    yield_elongation = df.loc[yield_index, "elongation_monotonic"] if yield_index is not None else None

    # Logging for yield point detection
    notify_progress(0.95, f"[Yield Detection] Method: max 2nd derivative (change in slope)")
    notify_progress(0.96, f"[Yield Detection] Index: {yield_index}, Time: {yield_time:.3f}s, Elongation: {yield_elongation:.3f}%")
    if yield_index == 0 or yield_index == len(df) - 1:
        notify_progress(0.97, f"[Yield Detection][Warning] Yield point at boundary (index={yield_index}). This may indicate a detection issue.")

    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp_s"], df["elongation_percent"], label="Raw %", linestyle="--", alpha=0.3)
    plt.plot(df["timestamp_s"], df["elongation_median"], label="Median Filtered %", linestyle="--", color="orange", linewidth=1)
    plt.plot(df["timestamp_s"], df["elongation_smoothed"], label="LOWESS Smoothed %", color="blue", linewidth=1, alpha=0.6)
    plt.plot(df["timestamp_s"], df["elongation_monotonic"], label="Final Monotonic %", color="green", linewidth=2)

    if yield_time is not None and yield_elongation is not None:
        plt.axvline(yield_time, color='red', linestyle='--', label='Yield Point')
        plt.scatter([yield_time], [yield_elongation], color='red')

    plt.title("Rebar Elongation Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Elongation (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Add number of frames annotation to main plot
    plt.text(0.99, 0.01, f"Frames used: {len(df)}", ha='right', va='bottom',
             transform=plt.gca().transAxes, fontsize=10, color='gray')

    plt.savefig(plot_path)
    df.to_csv(output_csv, index=False)

    # --- Auto-zoomed plot ---
    # Use percentiles to ignore outliers for zoomed plot
    all_vals = pd.concat([df["elongation_monotonic"], df["elongation_smoothed"]]).dropna()
    if len(all_vals) > 0:
        q_low, q_high = np.percentile(all_vals, [1, 99])
        margin = 0.5
        zoom_min = max(q_low - margin, 0)
        zoom_max = q_high + margin
        if zoom_max - zoom_min < 2:
            zoom_max = zoom_min + 2
        # Only apply clipping to the zoomed plot
        plt.ylim(zoom_min, zoom_max)
        plt.yticks(np.arange(np.floor(zoom_min), np.ceil(zoom_max)+0.1, 0.5))
        # Add number of frames annotation to zoomed plot
        plt.text(0.99, 0.01, f"Frames used: {len(df)}", ha='right', va='bottom',
                 transform=plt.gca().transAxes, fontsize=10, color='gray')
        if zoomed_plot_path is not None:
            plt.savefig(zoomed_plot_path)
    plt.close()

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
