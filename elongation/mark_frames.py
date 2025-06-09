import os
import cv2
import numpy as np
import pandas as pd
from elongation.detect_markers import detect_markers
from tqdm import tqdm

def process_images(
    input_folder,
    output_folder,
    csv_output_path,
    skip_start_frames=0,
    skip_end_frames=0,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    color_ref=(0, 255, 255),    # Yellow
    color_curr=(255, 100, 100), # Blue
    color_grid=(200, 200, 200),
    color_middle=(0, 0, 255),   # Red
    max_pattern_height=12,
    pattern_width=8,
    search_margin_y=5,
    search_margin_x=5,
    pattern_capture_frames=10,
    pattern_capture_step=5,
    prune_threshold=1e7,
    top_n_to_keep=5,
    scan_width=10,
    threshold_ratio=0.3,
    min_valid_distance=50,
    max_band_thickness=10,
    progress_callback=None,
    cancel_event=None
):
    def notify_progress(p, msg):
        if progress_callback:
            progress_callback(p, msg)
        else:
            print(f"{int(p*100)}% - {msg}")

    def filter_consistent_pairs(pairs, pattern_capture):
        consistent = []
        pair_score_map = []

        notify_progress(0.4,f"\n⏳ Scoring {len(pairs)} pattern pairs...")

        for i, (pt, pb) in enumerate(tqdm(pairs, desc="Evaluating pairs")):
            total_score = 0
            count_valid = 0
            if cancel_event and cancel_event.is_set():
                notify_progress(1.0, "Processing cancelled by user.")
                break
            for gray, center_x, *_ in pattern_capture:
                if cancel_event and cancel_event.is_set():
                    notify_progress(1.0, "Processing cancelled by user.")
                    break
                h, w = gray.shape
                x1 = max(center_x - pattern_width // 2 - search_margin_x, 0)
                x2 = min(center_x + pattern_width // 2 + search_margin_x, w)
                strip = gray[:, x1:x2]

                _, score_top = match_pattern(strip, pt, (10, h // 3))
                _, score_bottom = match_pattern(strip, pb, (2 * h // 3, h - 10))

                if score_top is None or score_bottom is None:
                    continue

                total_score += score_top + score_bottom
                count_valid += 1

            if count_valid == 0:
                continue
      
            avg_score = total_score / count_valid
            pair_score_map.append(((pt, pb), avg_score))

        pair_score_map.sort(key=lambda x: x[1])
        consistent = [pair for pair, score in pair_score_map[:top_n_to_keep]]
        return consistent

    def detect_rebar_center_x(gray):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        energy = np.sum(np.abs(sobel), axis=0)
        w = gray.shape[1]
        middle_band = energy[w // 4: w * 3 // 4]
        threshold = 0.4 * np.max(middle_band)
        indices = np.where(middle_band > threshold)[0]
        if len(indices) == 0:
            return w // 2
        left = indices[0] + w // 4
        right = indices[-1] + w // 4
        return (left + right) // 2

    def extract_pattern_candidates(gray, center_x, top_y, bottom_y, pattern_candidates_top, pattern_candidates_bottom):
        h, w = gray.shape
        x1 = max(center_x - pattern_width // 2 - search_margin_x, 0)
        x2 = min(center_x + pattern_width // 2 + search_margin_x, w)
        strip = gray[:, x1:x2]

        for dy in range(0, search_margin_y + 1):
            ty = top_y + dy
            if ty + max_pattern_height <= h:
                pattern = strip[ty:ty + max_pattern_height, :].copy()
                pattern_candidates_top.append(pattern)

        for dy in range(0, search_margin_y + 1):
            by = bottom_y - dy
            if by - max_pattern_height >= 0:
                pattern = strip[by - max_pattern_height:by, :].copy()
                pattern_candidates_bottom.append(pattern)

    def match_pattern(strip, pattern, search_range):
        best_score = float('inf')
        best_y = None

        for y in range(search_range[0], search_range[1] - pattern.shape[0]):
            candidate = strip[y:y + pattern.shape[0], :]
            if candidate.shape != pattern.shape:
                continue

            score = np.linalg.norm(candidate.astype(float) - pattern.astype(float))
            if score < best_score:
                best_score = score
                best_y = y

        if best_y is None:
            return None, None
        return best_y, best_score

    def draw_reference_grid(img, ref_top, ref_bottom, ref_distance):
        h, w = img.shape[:2]
        for i in range(11):
            y = int(ref_top + i * (ref_distance / 10))
            cv2.line(img, (0, y), (w, y), color_grid, 1)
        for y in range(0, ref_top, int(ref_distance / 10)):
            cv2.line(img, (0, y), (w, y), color_grid, 1)
        for y in range(ref_bottom, h, int(ref_distance / 10)):
            cv2.line(img, (0, y), (w, y), color_grid, 1)

    def score_candidate_pair(pt, pb, frames, precomputed):
        total_score = 0
        valid_matches = 0
        temp_results = {}
        for frame in frames:
            if cancel_event and cancel_event.is_set():
                notify_progress(1.0, "Processing cancelled by user.")
                break
            img, gray = precomputed[frame]
            center_x = detect_rebar_center_x(gray)
            x1 = max(center_x - pattern_width // 2 - search_margin_x, 0)
            x2 = min(center_x + pattern_width // 2 + search_margin_x, gray.shape[1])
            strip = gray[:, x1:x2]

            top_y, score_top = match_pattern(strip, pt, (10, gray.shape[0] // 3))
            bottom_y, score_bottom = match_pattern(strip, pb, (2 * gray.shape[0] // 3, gray.shape[0] - 10))

            if top_y is not None and bottom_y is not None:
                score = score_top + score_bottom
                total_score += score
                valid_matches += 1
                temp_results[frame] = (top_y, bottom_y, score)

            if total_score > prune_threshold:
                return float('inf'), None

        return total_score if valid_matches > 0 else float('inf'), temp_results
    if cancel_event and cancel_event.is_set():
        notify_progress(1.0, "Processing cancelled by user.")
        return
    pattern_candidates_top = []
    pattern_candidates_bottom = []
    ref_top = None
    ref_bottom = None
    ref_distance = None
    match_results = {}

    os.makedirs(output_folder, exist_ok=True)
    frames = sorted(f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png')))
    frames = frames[skip_start_frames: len(frames) - skip_end_frames]
    notify_progress(0.1,f"🔎 Total usable frames after skip: {len(frames)}")

    precomputed = {}
    for frame in tqdm(frames, desc="Precomputing grayscale"):
        if cancel_event and cancel_event.is_set():
            notify_progress(1.0, "Processing cancelled by user.")
            break
        img_path = os.path.join(input_folder, frame)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        precomputed[frame] = (img, gray)

    if cancel_event and cancel_event.is_set():
        notify_progress(1.0, "Processing cancelled by user.")
        return
    pattern_capture = []
    notify_progress(0.2,"🗕 Collecting pattern candidates from initial frames...")
    for idx in range(0, pattern_capture_frames * pattern_capture_step, pattern_capture_step):
        if cancel_event and cancel_event.is_set():
            notify_progress(1.0, "Processing cancelled by user.")
            break
        if idx >= len(frames):
            break
        frame = frames[idx]
        img, gray = precomputed[frame]
        center_x = detect_rebar_center_x(gray)
        top_y, bottom_y = detect_markers(gray, center_x,scan_width, threshold_ratio, min_valid_distance, max_band_thickness)
        if top_y is not None and bottom_y is not None:
            extract_pattern_candidates(gray, center_x, top_y, bottom_y, pattern_candidates_top, pattern_candidates_bottom)
            pattern_capture.append((gray, center_x, top_y, bottom_y))
    if cancel_event and cancel_event.is_set():
        notify_progress(1.0, "Processing cancelled by user.")
        return
    if not pattern_candidates_top or not pattern_candidates_bottom:
        notify_progress(1.0,"❌ Could not collect pattern candidates. Exiting.")
        return

    pattern_pairs = [(pt, pb) for pt in pattern_candidates_top for pb in pattern_candidates_bottom]
    notify_progress(0.35,f"🔄 Testing {len(pattern_pairs)} pattern pairs before filtering...")
    pattern_pairs = filter_consistent_pairs(pattern_pairs, pattern_capture)
    notify_progress(0.45,f"✅ Filtered to {len(pattern_pairs)} consistent pattern pairs.")

    args_list = [(pt, pb, frames, precomputed) for pt, pb in pattern_pairs]
    best_pattern = None
    best_score = float('inf')

    for i, (pt, pb) in enumerate(tqdm(pattern_pairs, desc="Scoring candidates")):
        if cancel_event and cancel_event.is_set():
            notify_progress(1.0, "Processing cancelled by user.")
            break
        score, temp_results = score_candidate_pair(pt, pb, frames, precomputed)
        if score < best_score:
            best_score = score
            best_pattern = (pt, pb)
            match_results = temp_results
    if cancel_event and cancel_event.is_set():
        notify_progress(1.0, "Processing cancelled by user.")
        return
    if best_pattern is None:
        notify_progress(1.0,"❌ Could not lock best global pattern.")
        return

    selected_top, selected_bottom = best_pattern
    notify_progress(0.75,"✅ Global best pattern selected with score: {best_score}")

    data = []
    print("🖼 Drawing overlays and saving output frames...")
    for frame in tqdm(frames, desc="Drawing pass"):
        if frame not in match_results:
            continue
        if cancel_event and cancel_event.is_set():
            notify_progress(1.0, "Processing cancelled by user.")
            break

        top_y, bottom_y, _ = match_results[frame]
        img, gray = precomputed[frame]
        center_x = detect_rebar_center_x(gray)

        if ref_top is None:
            ref_top, ref_bottom = top_y, bottom_y
            ref_distance = ref_bottom - ref_top

        elongation_px = bottom_y - top_y - ref_distance
        elongation_percent = ((bottom_y - top_y) / ref_distance) * 100

        cv2.line(img, (center_x, 0), (center_x, img.shape[0]), color_middle, 1)
        draw_reference_grid(img, ref_top, ref_bottom, ref_distance)

        cv2.line(img, (0, ref_top), (img.shape[1], ref_top), color_ref, 2)
        cv2.line(img, (0, ref_bottom), (img.shape[1], ref_bottom), color_ref, 2)
        cv2.putText(img, f"{ref_distance}px (100%)", (10, ref_top - 10), font, font_scale, color_ref, 2)

        cv2.line(img, (0, top_y), (img.shape[1], top_y), color_curr, 2)
        cv2.line(img, (0, bottom_y), (img.shape[1], bottom_y), color_curr, 2)
        cv2.putText(img, f"{bottom_y - top_y}px ({elongation_percent:.1f}%)", (img.shape[1] - 250, top_y - 10), font, font_scale, color_curr, 2)

        output_path = os.path.join(output_folder, frame)
        cv2.imwrite(output_path, img)

        timestamp_str = os.path.splitext(frame)[0].split('_')[-1].replace('s', '')
        data.append({
            "frame": frame,
            "timestamp_s": float(timestamp_str),
            "ref_top_px": ref_top,
            "ref_bottom_px": ref_bottom,
            "curr_top_px": top_y,
            "curr_bottom_px": bottom_y,
            "elongation_px": elongation_px,
            "elongation_percent": elongation_percent
        })
    if cancel_event and cancel_event.is_set():
        notify_progress(1.0, "Processing cancelled by user.")
        return
    pd.DataFrame(data).to_csv(csv_output_path, index=False)
    notify_progress(0.80,f"✔ All done! Marked frames in '{output_folder}', data in '{csv_output_path}'")

if __name__ == "__main__":
    process_images("output_frames", "elongation_marked_frames", "elongation_data.csv")
