#!/usr/bin/env python3
"""
Test script for enhanced motion detection methods.
This script demonstrates the different detection approaches for camera movement.
"""

import os
import sys
from elongation.extract_frames import (
    detect_motion_quality, 
    detect_edge_movement, 
    detect_background_movement,
    detect_cumulative_movement,
    find_enhanced_stable_start_frame,
    detect_orb_rotation
)
import cv2

def test_enhanced_detection(video_path="sample/40kn-2.mp4"):
    """Test all enhanced detection methods on the video."""
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"🎬 Testing enhanced motion detection on: {video_path}")
    print("=" * 60)
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Test 2: Enhanced stability detection
    print("\n2️⃣ Testing Enhanced Stability Detection:")
    print("-" * 40)
    
    start_frame, start_time, stability_info = find_enhanced_stable_start_frame(
        video_path, fps, max_check_frames=300,
        focus_threshold=50,
        motion_threshold=200,  # More sensitive
        edge_threshold=30,     # More sensitive
        bg_threshold=20,       # More sensitive
        min_consecutive_stable=10
    )
    
    print(f"   Start frame: {start_frame}")
    print(f"   Start time: {start_time:.2f}s")
    if start_frame > 0 or start_time > 0:
        print(f"   Skipped {start_frame} frames ({start_time:.2f} seconds) before stable frames begin.")
    else:
        print("   No frames skipped; video is stable from the start.")
    if stability_info:
        print(f"   Focus scores: {min(stability_info['focus_scores']):.1f}-{max(stability_info['focus_scores']):.1f}")
        print(f"   Motion scores: {min(stability_info['motion_scores']):.1f}-{max(stability_info['motion_scores']):.1f}")
        print(f"   Edge scores: {min(stability_info['edge_scores']):.1f}-{max(stability_info['edge_scores']):.1f}")
        print(f"   Background scores: {min(stability_info['bg_scores']):.1f}-{max(stability_info['bg_scores']):.1f}")

    # Test 3: Individual frame analysis
    print("\n3️⃣ Testing Individual Frame Analysis:")
    print("-" * 40)
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    for frame_idx in range(10):  # Test first 10 frames
        success, frame = cap.read()
        if not success:
            break
        timestamp = frame_idx / fps
        if prev_frame is not None:
            is_motion_stable, motion_score, _ = detect_motion_quality(frame, prev_frame, 200)
            is_edge_stable, edge_score, _ = detect_edge_movement(frame, prev_frame, 30)
            is_bg_stable, bg_score, rebar_score = detect_background_movement(frame, prev_frame, None, 20)
            # Only print for first 2 frames, then every 5th frame
            if frame_idx < 2 or frame_idx % 5 == 0:
                print(f"   Frame {frame_idx:2d} ({timestamp:5.2f}s):")
                print(f"      Motion: {is_motion_stable} (score: {motion_score:6.1f})")
                print(f"      Edge: {is_edge_stable} (score: {edge_score:6.1f})")
                print(f"      Background: {is_bg_stable} (score: {bg_score:6.1f}, rebar: {rebar_score:6.1f})")
        prev_frame = frame.copy()
    cap.release()
    print("\n   Individual frame analysis complete. (Logs shortened)")

    # --- All other tests and scenarios here ---
    test_specific_scenarios()

    # --- FINAL: Cumulative Movement and ORB Analysis ---
    print("\n================ FINAL GLOBAL MOVEMENT CHECKS ================")
    # Cumulative Movement Detection (first half of video)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print("\n🧭 Cumulative Movement Detection (first half of video):")
    half_frames = total_frames // 2
    is_stable, total_movement, movement_trend = detect_cumulative_movement(
        video_path, fps, max_check_frames=half_frames, check_interval=30
    )
    print(f"   Cumulative movement: {total_movement:.1f} pixels (first {half_frames} frames)")
    print(f"   Movement trend: {movement_trend}")
    print(f"   Is stable: {is_stable}")

    # ORB-based individual frame analysis summary (at end)
    cap = cv2.VideoCapture(video_path)
    frame_gap = 5
    max_frames = min(300, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    orb_first_unstable = None
    orb_first_unstable_time = None
    orb_last_unstable = None
    orb_last_unstable_time = None
    orb_first_stable_after_unstable = None
    orb_first_stable_after_unstable_time = None
    orb_in_unstable = False
    consecutive_stable = 0
    required_consecutive_stable = 5
    print("\n🔬 ORB-Based Edge-Only, Sensitive Analysis (every 5th frame, up to 300 frames, stop at 5 consecutive stable):")
    for frame_idx in range(frame_gap, max_frames, frame_gap):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            break
        timestamp = frame_idx / fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - frame_gap)
        _, prev_frame = cap.read()
        is_orb_stable, orb_angle, orb_trans, orb_matches = detect_orb_rotation(
            frame, prev_frame, translation_threshold=0.2, frame_gap=frame_gap, mask_center=True
        )
        if frame_idx < 2*frame_gap or frame_idx % (5*frame_gap) == 0:
            print(f"   Frame {frame_idx:3d} vs {frame_idx-frame_gap:3d} ({timestamp:5.2f}s): ORB: {is_orb_stable} (angle: {orb_angle:.2f} deg, trans: {orb_trans}, matches: {orb_matches})")
        if not is_orb_stable:
            if orb_first_unstable is None:
                orb_first_unstable = frame_idx
                orb_first_unstable_time = timestamp
            orb_last_unstable = frame_idx
            orb_last_unstable_time = timestamp
            orb_in_unstable = True
            consecutive_stable = 0
        elif orb_in_unstable:
            consecutive_stable += 1
            if consecutive_stable == required_consecutive_stable and orb_first_stable_after_unstable is None:
                orb_first_stable_after_unstable = frame_idx - (required_consecutive_stable - 1) * frame_gap
                orb_first_stable_after_unstable_time = (orb_first_stable_after_unstable) / fps
                break
        else:
            consecutive_stable = 0
    cap.release()
    if orb_first_unstable is not None:
        print(f"\n   ORB: First unstable frame at {orb_first_unstable} ({orb_first_unstable_time:.2f}s). Stable up to this point.")
        print(f"   ORB: Last unstable frame at {orb_last_unstable} ({orb_last_unstable_time:.2f}s).")
        if orb_first_stable_after_unstable is not None:
            print(f"   ORB: First stable frame after last unstable at {orb_first_stable_after_unstable} ({orb_first_stable_after_unstable_time:.2f}s) (after {required_consecutive_stable} consecutive stable frames).")
        else:
            print(f"   ORB: No stable frames detected after instability.")
    else:
        print(f"\n   ORB: All analyzed frames are stable.")
    print("============================================================\n")

def test_specific_scenarios():
    """Test specific scenarios with different thresholds."""
    
    print("\n🎯 Testing Specific Scenarios:")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Very Sensitive",
            "motion_threshold": 100,
            "edge_threshold": 20,
            "bg_threshold": 10
        },
        {
            "name": "Moderately Sensitive", 
            "motion_threshold": 300,
            "edge_threshold": 40,
            "bg_threshold": 25
        },
        {
            "name": "Standard",
            "motion_threshold": 500,
            "edge_threshold": 60,
            "bg_threshold": 40
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Testing: {scenario['name']}")
        print("-" * 30)
        
        start_frame, start_time, stability_info = find_enhanced_stable_start_frame(
            "sample/40kn-2.mp4", 30.0, max_check_frames=300,
            focus_threshold=50,
            motion_threshold=scenario["motion_threshold"],
            edge_threshold=scenario["edge_threshold"],
            bg_threshold=scenario["bg_threshold"],
            min_consecutive_stable=10
        )
        
        print(f"   Start frame: {start_frame}")
        print(f"   Start time: {start_time:.2f}s")
        
        if stability_info:
            max_motion = max(stability_info['motion_scores'])
            max_edge = max(stability_info['edge_scores'])
            max_bg = max(stability_info['bg_scores'])
            print(f"   Max scores - Motion: {max_motion:.1f}, Edge: {max_edge:.1f}, BG: {max_bg:.1f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "sample/40kn-2.mp4"
    
    test_enhanced_detection(video_path) 