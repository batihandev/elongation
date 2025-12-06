# Re-import necessary libraries after code execution state reset
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_force_curve(df, yield_strength_mpa, rebar_diameter_mm, plot_path):

    E = 200_000  # MPa, Young's modulus for steel

    area_mm2 = np.pi * (rebar_diameter_mm / 2) ** 2
    area_m2 = area_mm2 * 1e-6

    df = df.copy()
    df["strain"] = (df["elongation_smoothed_corrected"] - 100) / 100

    df["stress_mpa"] = np.minimum(df["strain"] * E, yield_strength_mpa)
    df["estimated_force_kN"] = df["stress_mpa"] * area_m2

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp_s"], df["estimated_force_kN"], color="purple", label=f"Estimated Force (Yield = {yield_strength_mpa} MPa)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (kN)")
    plt.title("Estimated Axial Force Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
