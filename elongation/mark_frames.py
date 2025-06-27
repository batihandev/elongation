import os
import cv2
import numpy as np
import pandas as pd
from elongation.detect_markers import detect_markers
from tqdm import tqdm
from elongation.subpixel import subpixel_refine

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
    max_pattern_height=20,
    pattern_width=15,
    search_margin_y=1, # affect initial collection of pattern candidates not the final matching
    search_margin_x=1, # affect initial collection of pattern candidates not the final matching
    pattern_capture_frames=5,
    pattern_capture_step=15,
    prune_threshold=1e7,
    top_n_to_keep=5,
    scan_width=10,
    threshold_ratio=0.3,
    min_valid_distance=50,
    max_band_thickness=10,
    progress_callback=None,
    cancel_event=None,
    pattern_top_grid=7,
    pattern_bottom_grid=3
):
    """
    pattern_top_grid: int (default 10) - grid line (0=bottom, 10=top) for top pattern extraction
    pattern_bottom_grid: int (default 0) - grid line (0=bottom, 10=top) for bottom pattern extraction
    """
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

    def normalize_and_denoise(img):
        # Apply Gaussian blur for denoising
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        # Normalize to zero mean, unit variance
        img_norm = img_blur.astype(float)
        img_norm = (img_norm - np.mean(img_norm)) / (np.std(img_norm) + 1e-8)
        return img_norm

    def match_pattern(strip, pattern, search_range):
        best_score = float('inf')
        best_y = None
        scores = {}
        pattern_proc = normalize_and_denoise(pattern)
        for y in range(search_range[0], search_range[1] - pattern.shape[0]):
            iy = int(round(y))
            candidate = strip[iy:iy + pattern.shape[0], :]
            if candidate.shape != pattern.shape:
                continue
            candidate_proc = normalize_and_denoise(candidate)
            score = np.linalg.norm(candidate_proc - pattern_proc)
            scores[y] = score
            if score < best_score:
                best_score = score
                best_y = y
        if best_y is None:
            return None, None
        # Subpixel refinement
        def score_func(y_):
            iy_ = int(round(y_))
            if iy_ in scores:
                return scores[iy_]
            if iy_ < search_range[0] or iy_ > search_range[1] - pattern.shape[0]:
                return float('inf')
            candidate = strip[iy_:iy_ + pattern.shape[0], :]
            if candidate.shape != pattern.shape:
                return float('inf')
            candidate_proc = normalize_and_denoise(candidate)
            return np.linalg.norm(candidate_proc - pattern_proc)
        if best_y > search_range[0] and best_y < search_range[1] - pattern.shape[0] - 1:
            best_y_subpixel = subpixel_refine(score_func, best_y, mode='callable')
        else:
            best_y_subpixel = float(best_y)
        return best_y_subpixel, best_score

    def draw_reference_grid(img, ref_top, ref_bottom, ref_distance):
        h, w = img.shape[:2]
        # Draw 10 white grid lines evenly spaced across the entire image height (0=bottom, 9=near top)
        for i in range(10):  # Only 0-9, do not draw the 10th (topmost) line
            y = int((9 - i) * h / 9)  # 0 at bottom, 9 near top
            cv2.line(img, (0, y), (w, y), color_grid, 1)
            # Draw grid numbers (0=bottom, 9=near top) at left margin
            cv2.putText(img, str(i), (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

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

            h = gray.shape[0]
            # Use grid system: 0=bottom, 10=top
            y_top_center = int((10 - pattern_top_grid) * h / 10)
            y_bottom_center = int((10 - pattern_bottom_grid) * h / 10)
            search_margin = 20  # or use pattern_search_margin if you want it configurable
            # Search for top pattern near y_top_center
            top_y, score_top = match_pattern(strip, pt, (max(0, y_top_center - search_margin), min(h, y_top_center + search_margin)))
            # Search for bottom pattern near y_bottom_center
            bottom_y, score_bottom = match_pattern(strip, pb, (max(0, y_bottom_center - search_margin), min(h, y_bottom_center + search_margin)))

            if top_y is not None and bottom_y is not None and score_top is not None and score_bottom is not None:
                score = score_top + score_bottom
                total_score += score
                valid_matches += 1
                temp_results[frame] = (top_y, bottom_y, score)

            if total_score > prune_threshold:
                return float('inf'), None

        return total_score if valid_matches > 0 else float('inf'), temp_results

    def find_pattern_near_reference(strip, pattern, reference_y, max_search_distance=20):
        h = strip.shape[0]
        best_score = float('inf')
        best_y = None
        scores = {}
        pattern_proc = normalize_and_denoise(pattern)
        for distance in range(max_search_distance + 1):
            # Try positive offset
            test_y = reference_y + distance
            if test_y >= 0 and test_y + pattern.shape[0] <= h:
                iy = int(round(test_y))
                candidate = strip[iy:iy + pattern.shape[0], :]
                if candidate.shape == pattern.shape:
                    candidate_proc = normalize_and_denoise(candidate)
                    score = np.linalg.norm(candidate_proc - pattern_proc)
                    scores[test_y] = score
                    if score < best_score:
                        best_score = score
                        best_y = test_y
            # Try negative offset (skip distance=0 to avoid duplicate)
            if distance > 0:
                test_y = reference_y - distance
                if test_y >= 0 and test_y + pattern.shape[0] <= h:
                    iy = int(round(test_y))
                    candidate = strip[iy:iy + pattern.shape[0], :]
                    if candidate.shape == pattern.shape:
                        candidate_proc = normalize_and_denoise(candidate)
                        score = np.linalg.norm(candidate_proc - pattern_proc)
                        scores[test_y] = score
                        if score < best_score:
                            best_score = score
                            best_y = test_y
        if best_y is None:
            return None, None
        # Subpixel refinement
        def score_func(y_):
            iy_ = int(round(y_))
            if iy_ in scores:
                return scores[iy_]
            if iy_ < 0 or iy_ + pattern.shape[0] > h:
                return float('inf')
            candidate = strip[iy_:iy_ + pattern.shape[0], :]
            if candidate.shape != pattern.shape:
                return float('inf')
            candidate_proc = normalize_and_denoise(candidate)
            return np.linalg.norm(candidate_proc - pattern_proc)
        if best_y > 0 and best_y < h - pattern.shape[0] - 1:
            best_y_subpixel = subpixel_refine(score_func, best_y, mode='callable')
        else:
            best_y_subpixel = float(best_y)
        return best_y_subpixel, best_score

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
        h = gray.shape[0]
        # Use grid system: 0=bottom, 10=top
        y_top = int((10 - pattern_top_grid) * h / 10)
        y_bottom = int((10 - pattern_bottom_grid) * h / 10)
        if idx == 0:
            print(f"[DEBUG] First frame pattern extraction: y_top={y_top}, y_bottom={y_bottom}, image height={h}")
        extract_pattern_candidates(gray, center_x, y_top, y_bottom, pattern_candidates_top, pattern_candidates_bottom)
        pattern_capture.append((gray, center_x, y_top, y_bottom))
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
            # Store the best pattern y-positions from the first frame
            if frames and match_results is not None:
                first_frame = frames[0]
                first_result = match_results.get(first_frame)
                if first_result is not None and len(first_result) == 3:
                    ref_top_fixed, ref_bottom_fixed, _ = first_result
                    print(f"[DEBUG] Reference top (y_3): {ref_top_fixed}, Reference bottom (y_7): {ref_bottom_fixed}, Gauge length: {ref_bottom_fixed - ref_top_fixed}")
    if cancel_event and cancel_event.is_set():
        notify_progress(1.0, "Processing cancelled by user.")
        return
    if best_pattern is None:
        notify_progress(1.0,"❌ Could not lock best global pattern.")
        return

    selected_top, selected_bottom = best_pattern
    notify_progress(0.75, f"✅ Global best pattern selected with score: {best_score}")

    # Find the reference positions where the best pattern was originally detected
    # Use the first frame to establish the reference positions for yellow lines
    ref_top_fixed = None
    ref_bottom_fixed = None
    if frames and match_results is not None:
        first_frame = frames[0]
        first_result = match_results.get(first_frame)
        if first_result is not None and len(first_result) == 3:
            ref_top_fixed, ref_bottom_fixed, _ = first_result
            print(f"[DEBUG] Yellow line reference positions - Top: {ref_top_fixed}, Bottom: {ref_bottom_fixed}, Gauge length: {ref_bottom_fixed - ref_top_fixed}")
        else:
            # Fallback: use the original detected markers if available
            if ref_top is not None and ref_bottom is not None:
                ref_top_fixed, ref_bottom_fixed = ref_top, ref_bottom
                print(f"[DEBUG] Using fallback reference positions - Top: {ref_top_fixed}, Bottom: {ref_bottom_fixed}")
            else:
                notify_progress(1.0, "❌ Could not establish reference positions for yellow lines.")
                return

    data = []
    print("🖼 Drawing overlays and saving output frames...")
    for frame in tqdm(frames, desc="Drawing pass"):
        if cancel_event and cancel_event.is_set():
            notify_progress(1.0, "Processing cancelled by user.")
            break

        img, gray = precomputed[frame]
        center_x = detect_rebar_center_x(gray)
        
        # Ensure we have valid reference positions
        if ref_top_fixed is None or ref_bottom_fixed is None:
            continue

        # Search for the best pattern near the reference positions
        x1 = max(center_x - pattern_width // 2 - search_margin_x, 0)
        x2 = min(center_x + pattern_width // 2 + search_margin_x, gray.shape[1])
        strip = gray[:, x1:x2]

        # Find top pattern near the reference top position
        top_y, score_top = find_pattern_near_reference(strip, selected_top, ref_top_fixed, max_search_distance=20)
        
        # Find bottom pattern near the reference bottom position
        bottom_y, score_bottom = find_pattern_near_reference(strip, selected_bottom, ref_bottom_fixed, max_search_distance=20)

        # Ensure we have valid coordinates
        if top_y is None or bottom_y is None:
            continue

        # Calculate elongation using the reference gauge length
        ref_gauge_length = ref_bottom_fixed - ref_top_fixed
        elongation_px = bottom_y - top_y - ref_gauge_length
        elongation_percent = ((bottom_y - top_y) / ref_gauge_length) * 100

        cv2.line(img, (center_x, 0), (center_x, img.shape[0]), color_middle, 1)
        
        # Draw reference grid and yellow lines at the fixed reference positions
        draw_reference_grid(img, int(round(ref_top_fixed)), int(round(ref_bottom_fixed)), ref_gauge_length)
        cv2.line(img, (0, int(round(ref_top_fixed))), (img.shape[1], int(round(ref_top_fixed))), color_ref, 2)
        cv2.line(img, (0, int(round(ref_bottom_fixed))), (img.shape[1], int(round(ref_bottom_fixed))), color_ref, 2)
        cv2.putText(img, f"{ref_gauge_length}px (100%)", (10, int(round(ref_top_fixed)) - 10), font, font_scale, color_ref, 2)

        # Draw blue lines at the newly found optimal positions
        cv2.line(img, (0, int(round(top_y))), (img.shape[1], int(round(top_y))), color_curr, 2)
        cv2.line(img, (0, int(round(bottom_y))), (img.shape[1], int(round(bottom_y))), color_curr, 2)
        cv2.putText(img, f"{bottom_y - top_y}px ({elongation_percent:.1f}%)", (img.shape[1] - 250, int(round(top_y)) - 10), font, font_scale, color_curr, 2)

        # Draw 50% transparent pattern overlays matching the pattern's actual size
        # Green overlay for yellow line patterns (reference)
        green_overlay = img.copy()
        cv2.rectangle(
            green_overlay,
            (center_x - pattern_width // 2, int(round(ref_top_fixed))),
            (center_x + pattern_width // 2, int(round(ref_top_fixed)) + max_pattern_height),
            (0, 255, 0),
            -1
        )
        cv2.rectangle(
            green_overlay,
            (center_x - pattern_width // 2, int(round(ref_bottom_fixed)) - max_pattern_height),
            (center_x + pattern_width // 2, int(round(ref_bottom_fixed))),
            (0, 255, 0),
            -1
        )
        cv2.addWeighted(green_overlay, 0.5, img, 0.5, 0, img)

        # Purple overlay for blue line patterns (current)
        purple_overlay = img.copy()
        cv2.rectangle(
            purple_overlay,
            (center_x - pattern_width // 2, int(round(top_y))),
            (center_x + pattern_width // 2, int(round(top_y)) + max_pattern_height),
            (255, 0, 255),
            -1
        )
        cv2.rectangle(
            purple_overlay,
            (center_x - pattern_width // 2, int(round(bottom_y)) - max_pattern_height),
            (center_x + pattern_width // 2, int(round(bottom_y))),
            (255, 0, 255),
            -1
        )
        cv2.addWeighted(purple_overlay, 0.5, img, 0.5, 0, img)

        output_path = os.path.join(output_folder, frame)
        cv2.imwrite(output_path, img)

        timestamp_str = os.path.splitext(frame)[0].split('_')[-1].replace('s', '')
        data.append({
            "frame": frame,
            "timestamp_s": float(timestamp_str),
            "ref_top_px": ref_top_fixed,
            "ref_bottom_px": ref_bottom_fixed,
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
