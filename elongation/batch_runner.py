import os
import pandas as pd
from extract_frames import extract_frames
from mark_frames import process_images
from analyze_elongation import process_and_plot
from compute_force_curve import compute_force_curve
from generate_pdf_report import generate_pdf_report  # Ensure this exists

# 🔧 Update this list with video-specific parameters
every_n_frames = 5  # Extract every 5th frame
VIDEO_CONFIG = [
    {
        "video_file": "sample/40kn-1.mp4",
        "skip_start": 4,
        "skip_end": 4,
        "yield_strength": 420,  # MPa
        "rebar_diameter": 16    # mm
    },
    {
        "video_file": "sample/40kn-2.mp4",
        "skip_start": 25,
        "skip_end": 12,
        "yield_strength": 365,
        "rebar_diameter": 16
    }
]

def batch_run():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    results = []

    for config in VIDEO_CONFIG:
        video_file = config["video_file"]
        skip_start = config["skip_start"]
        skip_end = config["skip_end"]
        yield_strength = config["yield_strength"]
        rebar_diameter = config["rebar_diameter"]
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f"\n=== Processing: {video_file} ===")

        input_frames = os.path.join(results_dir, f"output_frames_{base_name}")
        marked_output = os.path.join(results_dir, f"elongation_marked_{base_name}")
        csv_output = os.path.join(results_dir, f"elongation_data_{base_name}.csv")
        final_csv = os.path.join(results_dir, f"elongation_final_{base_name}.csv")
        plot_path = os.path.join(results_dir, f"elongation_plot_{base_name}.png")
        force_path = os.path.join(results_dir, f"force_plot_{base_name}.png")

        extract_frames(video_file, input_frames, every_n_frames)

        process_images(
            input_folder=input_frames,
            output_folder=marked_output,
            csv_output_path=csv_output,
            skip_start_frames=skip_start,
            skip_end_frames=skip_end
        )

        if os.path.exists(csv_output):
            df = pd.read_csv(csv_output)
            df_final, yield_time, yield_elongation = process_and_plot(df, final_csv, plot_path)
            compute_force_curve(df_final, yield_strength, rebar_diameter, force_path)


            results.append({
                "name": base_name,
                "yield_strength": yield_strength,
                "rebar_diameter": rebar_diameter,
                "elongation_plot": plot_path,
                "force_plot": force_path
            })
        else:
            print(f"⚠️ CSV not found for {base_name}, skipping analysis.")

    print("\n📦 Batch Completed. Results:")
    for r in results:
        print(f"🔹 {r['name']}: Elongation → {r['elongation_plot']}, Force → {r['force_plot']}")

    # 📄 Generate PDF summary report inside results/
    pdf_path = os.path.join(results_dir, "elongation_summary_report.pdf")
    generate_pdf_report(results, output_path=pdf_path)
    print(f"\n📑 PDF Report saved to: {pdf_path}")

if __name__ == "__main__":
    batch_run()
