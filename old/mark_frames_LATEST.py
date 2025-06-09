import os
import cv2
import numpy as np
import pandas as pd
from elongation.detect_markers import detect_markers
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

max_pattern_height = 12
pattern_width = 8
search_margin_y = 5
search_margin_x = 5
pattern_capture_frames = 10
pattern_capture_step = 5  # capture every 3rd frame
prune_threshold = 1e7  # Early pruning: discard candidates that get too bad


def filter_consistent_pairs(
    pairs,
    pattern_capture,
    pattern_width,
    search_margin_x,
    top_n_to_keep=5
):
    consistent = []
    pair_score_map = []
    failed_scores = []


    print(f"\n⏳ Scoring {len(pairs)} pattern pairs...")

    for pt, pb in tqdm(pairs, desc="Evaluating pairs"):
        total_score = 0
        count_valid = 0

        for gray, center_x, *_ in pattern_capture:
            h, w = gray.shape
            x1 = max(center_x - pattern_width // 2 - search_margin_x, 0)
            x2 = min(center_x + pattern_width // 2 + search_margin_x, w)
            strip = gray[:, x1:x2]

            _, score_top = match_pattern(strip, pt, (10, h // 3))
            _, score_bottom = match_pattern(strip, pb, (2 * h // 3, h - 10))

            if score_top is None or score_bottom is None:
                continue  # Skip bad frames

            total_score += score_top + score_bottom
            count_valid += 1

        if count_valid == 0:
            continue  # Reject if no valid frame

        avg_score = total_score / count_valid
        pair_score_map.append(((pt, pb), avg_score))

    # === Sort and keep best N ===
    pair_score_map.sort(key=lambda x: x[1])
    consistent = [pair for pair, score in pair_score_map[:top_n_to_keep]]
    all_scores = [score for _, score in pair_score_map[:top_n_to_keep]]

    # === Log filtered-out worst scores ===
    for pair, score in pair_score_map[top_n_to_keep:]:
        failed_scores.append(score)

    # === Logging ===
    print(f"\n✅ Top {len(consistent)} consistent pairs selected.")

    if all_scores:
        print("\n📊 [TOP MATCHED PAIR STATS]")
        print(f"   min:  {min(all_scores):.2f}")
        print(f"   max:  {max(all_scores):.2f}")
        print(f"   mean: {np.mean(all_scores):.2f}")
        print(f"   std:  {np.std(all_scores):.2f}")
    else:
        print("\n⚠️ No consistent pattern pairs matched.")

    if failed_scores:
        failed_scores.sort()
        print(f"\n📉 [Top failing scores (best 5)]")
        for i, score in enumerate(failed_scores[:5]):
            print(f"   {i+1:02d}: {score:.2f}")

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

def extract_pattern_candidates(gray, center_x, top_y, bottom_y,pattern_candidates_top, pattern_candidates_bottom):
    h, w = gray.shape
    x1 = max(center_x - pattern_width // 2 - search_margin_x, 0)
    x2 = min(center_x + pattern_width // 2 + search_margin_x, w)
    strip = gray[:, x1:x2]

    # Extract only DOWNWARD from top marker
    for dy in range(0, search_margin_y + 1):  # Only positive offset
        ty = top_y + dy
        if ty + max_pattern_height <= h:
            pattern = strip[ty:ty + max_pattern_height, :].copy()
            pattern_candidates_top.append(pattern)

    # Extract only UPWARD from bottom marker
    for dy in range(0, search_margin_y + 1):  # Only positive offset
        by = bottom_y - dy
        if by - max_pattern_height >= 0:
            pattern = strip[by - max_pattern_height:by, :].copy()
            pattern_candidates_bottom.append(pattern)

    print(f"   ⬆ Top patterns added (downward): {len(pattern_candidates_top)}")
    print(f"   ⬇ Bottom patterns added (upward): {len(pattern_candidates_bottom)}")



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
        return None, None  # instead of 0

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

def process_images(input_folder, output_folder, csv_output_path,skip_start_frames=0, skip_end_frames=0):
    pattern_candidates_top = []
    pattern_candidates_bottom = []
    ref_top = None
    ref_bottom = None
    ref_distance = None
    match_results = {}  # frame → (top_y, bottom_y)

    os.makedirs(output_folder, exist_ok=True)
    frames = sorted(f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png')))
    frames = frames[skip_start_frames: len(frames) - skip_end_frames]
    print(f"🔎 Total usable frames after skip: {len(frames)}")
    print(f"🧠 Using {cpu_count()} CPU cores for parallel processing")

    precomputed = {}
    for frame in tqdm(frames, desc="Precomputing grayscale"):
        img_path = os.path.join(input_folder, frame)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        precomputed[frame] = (img, gray)

    pattern_capture = []
    print("📅 Collecting pattern candidates from initial frames...")
    for idx in range(0, pattern_capture_frames * pattern_capture_step, pattern_capture_step):
        if idx >= len(frames):
            break
        frame = frames[idx]
        img, gray = precomputed[frame]
        center_x = detect_rebar_center_x(gray)
        top_y, bottom_y = detect_markers(gray, center_x)
        if top_y is not None and bottom_y is not None:
            extract_pattern_candidates(gray, center_x, top_y, bottom_y,pattern_candidates_top, pattern_candidates_bottom)
            pattern_capture.append((gray, center_x, top_y, bottom_y))

    if not pattern_candidates_top or not pattern_candidates_bottom:
        print("❌ Could not collect pattern candidates. Exiting.")
        return

    # Form all candidate pairs
    pattern_pairs = [(pt, pb) for pt in pattern_candidates_top for pb in pattern_candidates_bottom]
    print(f"🔄 Testing {len(pattern_pairs)} pattern pairs before filtering...")

    # Filter valid pairs only
    pattern_pairs = filter_consistent_pairs(
        pattern_pairs,
        pattern_capture,
        pattern_width,
        search_margin_x,
    )

    print(f"✅ Filtered to {len(pattern_pairs)} consistent pattern pairs.")
    args_list = [(pt, pb, frames, precomputed) for pt, pb in pattern_pairs]

    best_pattern = None
    best_score = float('inf')

    with Pool(processes=cpu_count()) as pool:
        for result, (pt, pb) in tqdm(
            zip(pool.imap(score_candidate_pair, args_list), [(pt, pb) for pt in pattern_candidates_top for pb in pattern_candidates_bottom]),
            total=len(args_list),
            desc="Scoring candidates",
            mininterval=1
        ):
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
                # === Draw visual pixel scale bars ===
        bar_y = img.shape[0] - 20  # Bottom margin
        bar_margin_right = 20
        font_thickness = 1
        bar_thickness = 2

        # Draw 5px bar
        bar5_len = 5
        bar5_x_start = img.shape[1] - bar5_len - bar_margin_right-45
        cv2.line(img, (bar5_x_start, bar_y), (bar5_x_start + bar5_len, bar_y), (255, 255, 255), bar_thickness)
        cv2.putText(img, "5 px", (bar5_x_start - 5, bar_y - 10), font, font_scale, (255, 255, 255), font_thickness)

        # Draw 50px bar
        bar50_len = 50
        bar50_x_start = img.shape[1] - bar50_len - bar_margin_right
        cv2.line(img, (bar50_x_start, bar_y - 30), (bar50_x_start + bar50_len, bar_y - 30), (255, 255, 255), bar_thickness)
        cv2.putText(img, "50 px", (bar50_x_start - 5, bar_y - 40), font, font_scale, (255, 255, 255), font_thickness)


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
    process_images(input_folder, output_folder, csv_output_path)
