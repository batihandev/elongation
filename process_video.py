#!/usr/bin/env python3
import sys
import os
from elongation.extract_frames import extract_frames
from elongation.mark_frames import process_images
from elongation.analyze_elongation import process_and_plot
import pandas as pd
import subprocess
import shutil

def process_video(
    video_path, 
    every_n_frames=5, 
    min_elong=100, 
    max_elong=140,
    skip_start_frames=0,
    skip_start_seconds=0.0,
    auto_detect_focus=True,
    auto_detect_motion=True,
    focus_threshold=50,
    motion_threshold=200,
    min_consecutive_stable=50
):
    """Process a video file directly like the backend does"""
    
    # Get base name from video path
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Setup paths
    frames_dir = f"results/output_frames_{base_name}"
    marked_dir = f"results/elongation_marked_{base_name}"
    csv_path = f"results/elongation_data_{base_name}.csv"
    final_csv_path = f"results/elongation_final_{base_name}.csv"
    plot_path = f"results/elongation_plot_{base_name}.png"
    
    # Clean up previous outputs if they exist
    for path in [frames_dir, marked_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
    for path in [csv_path, final_csv_path, plot_path]:
        if os.path.exists(path):
            os.remove(path)
    
    print(f"🎬 Processing: {video_path}")
    print(f"📁 Output: {base_name}")
    
    # Step 1: Extract frames with focus and motion detection
    print("📸 Extracting frames...")
    extraction_result = extract_frames(
        video_path, 
        frames_dir, 
        every_n_frames=every_n_frames,
        skip_start_frames=skip_start_frames,
        skip_start_seconds=skip_start_seconds,
        auto_detect_focus=auto_detect_focus,
        auto_detect_motion=auto_detect_motion,
        focus_threshold=focus_threshold,
        motion_threshold=motion_threshold,
        edge_threshold=30,  # Match test parameters
        bg_threshold=20,    # Match test parameters
        min_consecutive_stable=min_consecutive_stable
    )
    
    print(f"   Frames saved: {extraction_result['frames_saved']}")
    print(f"   Frames skipped: {extraction_result['frames_skipped']}")
    print(f"   Time skipped: {extraction_result['time_skipped']:.2f}s")
    
    # Step 2: Process images and detect markers
    print("🔍 Detecting markers...")
    process_images(
        input_folder=frames_dir,
        output_folder=marked_dir,
        csv_output_path=csv_path,
        skip_start_frames=0,  # Already handled in extract_frames
        skip_end_frames=0
    )
    
    # Step 3: Analyze elongation
    print("📊 Analyzing elongation...")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        result = process_and_plot(
            df,
            output_csv=final_csv_path,
            plot_path=plot_path,
            min_elongation=min_elong,
            max_elongation=max_elong
        )
        
        if result is not None:
            df_final, yield_time, yield_elongation = result
            print(f"✅ Done! Yield time: {yield_time:.2f}s, Yield elongation: {yield_elongation:.2f}%")
            print(f"📈 Plot saved: {plot_path}")
            print(f"📊 Data saved: {final_csv_path}")
            # Open plot image for quick viewing
            if os.path.exists(plot_path):
                subprocess.Popen(['feh', plot_path])
        else:
            print("❌ Analysis failed")
    else:
        print("❌ No data generated")

if __name__ == "__main__":
    # Default to sample 1, can be overridden with command line argument
    video_file = "sample/40kn-1.mp4"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "1":
            video_file = "sample/40kn-1.mp4"
        elif sys.argv[1] == "2":
            video_file = "sample/40kn-2.mp4"
        elif sys.argv[1] == "2_sensitive":
            # More sensitive motion detection for sample2
            video_file = "sample/40kn-2.mp4"
            process_video(
                video_file,
                every_n_frames=5,
                min_elong=100,
                max_elong=140,
                auto_detect_focus=True,
                auto_detect_motion=True,
                focus_threshold=50,
                motion_threshold=200,  # Much more sensitive (default is 1000)
                min_consecutive_stable=20  # Require longer stable period
            )
            sys.exit(0)
        else:
            video_file = sys.argv[1]
    
    if not os.path.exists(video_file):
        print(f"❌ Video file not found: {video_file}")
        print("Usage: python process_video.py [1|2|2_sensitive|path_to_video]")
        sys.exit(1)
    
    process_video(video_file) 