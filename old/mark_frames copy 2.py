import os
import cv2
import numpy as np
import pandas as pd
from detect_markers import detect_markers
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Parameters
input_folder = "output_frames"
output_folder = "elongation_marked_frames"
csv_output_path = "elongation_data.csv"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
color_ref = (0, 255, 255)    # Yellow
color_curr = (255, 100, 100) # Blue
color_grid = (200, 200, 200)
color_middle = (0, 0, 255)   # Red

skip_start_frames = 30
skip_end_frames = 20
max_pattern_height = 5
pattern_width = 12
search_margin_y = 10
search_margin_x = 5
pattern_capture_frames = 20
prune_threshold = 1e7  # Early pruning: discard candidates that get too bad

pattern_candidates_top = []
pattern_candidates_bottom = []
ref_top = None
ref_bottom = None
ref_distance = None
match_results = {}  # frame → (top_y, bottom_y)


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


def extract_pattern_candidates(gray, center_x, top_y, bottom_y):
    h, w = gray.shape
    x1 = max(center_x - pattern_width // 2 - search_margin_x, 0)
    x2 = min(center_x + pattern_width // 2 + search_margin_x, w)
    strip = gray[:, x1:x2]

    for dy in range(-search_margin_y, search_margin_y + 1):
        ty = top_y + dy
        by = bottom_y + dy
        if 0 <= ty <= h - max_pattern_height:
            pattern_candidates_top.append(strip[ty:ty + max_pattern_height, :].copy())
        if 0 <= by <= h - max_pattern_height:
            pattern_candidates_bottom.append(strip[by:by + max_pattern_height, :].copy())


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


def score_candidate_pair(args):
    pt, pb, frames, precomputed = args
    total_score = 0
    valid_matches = 0
    temp_results = {}
    for frame in frames:
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


def process_images():
    global ref_top, ref_bottom, ref_distance

    os.makedirs(output_folder, exist_ok=True)
    frames = sorted(f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png')))
    frames = frames[skip_start_frames: len(frames) - skip_end_frames]
    print(f"🔎 Total usable frames after skip: {len(frames)}")

    precomputed = {}
    for frame in tqdm(frames, desc="Precomputing grayscale"):
        img_path = os.path.join(input_folder, frame)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        precomputed[frame] = (img, gray)

    pattern_capture = []
    print("📥 Collecting pattern candidates from initial frames...")
    for frame in tqdm(frames[:pattern_capture_frames], desc="Collecting candidates"):
        img, gray = precomputed[frame]
        center_x = detect_rebar_center_x(gray)
        top_y, bottom_y = detect_markers(gray, center_x)
        if top_y is not None and bottom_y is not None:
            extract_pattern_candidates(gray, center_x, top_y, bottom_y)
            pattern_capture.append((gray, center_x, top_y, bottom_y))

    if not pattern_candidates_top or not pattern_candidates_bottom:
        print("❌ Could not collect pattern candidates. Exiting.")
        return

    print(f"📊 Evaluating {len(pattern_candidates_top) * len(pattern_candidates_bottom)} candidate pattern pairs...")
    args_list = [(pt, pb, frames, precomputed) for pt in pattern_candidates_top for pb in pattern_candidates_bottom]
    best_pattern = None
    best_score = float('inf')
    global match_results

    with Pool(processes=cpu_count()) as pool:
        for result, (pt, pb) in tqdm(zip(pool.imap(score_candidate_pair, args_list), [(pt, pb) for pt in pattern_candidates_top for pb in pattern_candidates_bottom]), total=len(args_list), desc="Scoring candidates"):
            score, temp_results = result
            if score < best_score:
                best_score = score
                best_pattern = (pt, pb)
                match_results = temp_results

    if best_pattern is None:
        print("❌ Could not lock best global pattern.")
        return

    selected_top, selected_bottom = best_pattern
    print("✅ Global best pattern selected with score:", best_score)

    data = []
    print("🖼 Drawing overlays and saving output frames...")
    for frame in tqdm(frames, desc="Drawing pass"):
        if frame not in match_results:
            continue

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

    pd.DataFrame(data).to_csv(csv_output_path, index=False)
    print(f"✔ All done! Marked frames in '{output_folder}', data in '{csv_output_path}'")

if __name__ == "__main__":
    process_images()
