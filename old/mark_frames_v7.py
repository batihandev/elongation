import os
import cv2
import numpy as np
import pandas as pd

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

pattern_candidates_top = []
pattern_candidates_bottom = []
selected_patterns = {}
ref_top = None
ref_bottom = None
ref_distance = None


def detect_rebar_center_x(gray):
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    energy = np.sum(np.abs(sobel), axis=0)
    w = gray.shape[1]
    middle_band = energy[w // 4 : w * 3 // 4]
    threshold = 0.4 * np.max(middle_band)
    indices = np.where(middle_band > threshold)[0]
    if len(indices) == 0:
        return w // 2
    left = indices[0] + w // 4
    right = indices[-1] + w // 4
    return (left + right) // 2


def detect_markers(gray, center_x, scan_width=15, threshold_ratio=0.4, min_valid_distance=50, max_band_thickness=15):
    h, w = gray.shape
    x1, x2 = max(center_x - scan_width, 0), min(center_x + scan_width, w)
    strip = gray[:, x1:x2]
    strip = cv2.equalizeHist(strip)
    sobel_vertical = cv2.Sobel(strip, cv2.CV_64F, 0, 1, ksize=3)
    gradient_strength = np.abs(sobel_vertical).sum(axis=1)
    threshold = np.max(gradient_strength) * threshold_ratio

    def find_band(start_y, end_y, step):
        for y in range(start_y, end_y, step):
            if gradient_strength[y] > threshold:
                for dy in range(1, max_band_thickness):
                    next_y = y + dy * step
                    if 0 <= next_y < h and gradient_strength[next_y] > threshold:
                        return (y + next_y) // 2
        return None

    mid = h // 2
    top_y = find_band(10, mid, 1)
    bottom_y = find_band(h - 10, mid, -1)

    if top_y is not None and bottom_y is not None:
        if (bottom_y - top_y) < min_valid_distance:
            return None, None

    return top_y, bottom_y


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
    return best_y


def draw_reference_grid(img, ref_top, ref_bottom, ref_distance):
    h, w = img.shape[:2]
    for i in range(11):
        y = int(ref_top + i * (ref_distance / 10))
        cv2.line(img, (0, y), (w, y), color_grid, 1)
    for y in range(0, ref_top, int(ref_distance / 10)):
        cv2.line(img, (0, y), (w, y), color_grid, 1)
    for y in range(ref_bottom, h, int(ref_distance / 10)):
        cv2.line(img, (0, y), (w, y), color_grid, 1)


def process_images():
    global selected_patterns, ref_top, ref_bottom, ref_distance

    os.makedirs(output_folder, exist_ok=True)
    frames = sorted(f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png')))
    original_frames = frames[skip_start_frames: len(frames) - skip_end_frames]
    pattern_capture = []

    # First loop: collect candidates for first 20 frames
    for i, frame in enumerate(original_frames[:pattern_capture_frames]):
        img_path = os.path.join(input_folder, frame)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center_x = detect_rebar_center_x(gray)
        top_y, bottom_y = detect_markers(gray, center_x)
        if top_y is not None and bottom_y is not None:
            extract_pattern_candidates(gray, center_x, top_y, bottom_y)
            pattern_capture.append((gray, center_x, top_y, bottom_y))

    # Try to lock reference pattern from best match among first 20 frames
    if pattern_capture:
        best_frame = pattern_capture[0]
        gray, center_x, ref_top, ref_bottom = best_frame
        x1 = max(center_x - pattern_width // 2 - search_margin_x, 0)
        x2 = min(center_x + pattern_width // 2 + search_margin_x, gray.shape[1])
        strip = gray[:, x1:x2]

        def find_best(candidates, region):
            return min(
                candidates,
                key=lambda p: np.linalg.norm(strip[region:region + max_pattern_height, :] - p),
                default=None
            )

        selected_patterns['top'] = find_best(pattern_candidates_top, ref_top)
        selected_patterns['bottom'] = find_best(pattern_candidates_bottom, ref_bottom)
        ref_distance = ref_bottom - ref_top
        print("✅ Reference patterns locked in.")
    else:
        print("❌ Could not lock reference patterns. Exiting.")
        return

    data = []
    # Main loop: from the first frame after skip_start_frames
    for i, frame in enumerate(original_frames):
        img_path = os.path.join(input_folder, frame)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center_x = detect_rebar_center_x(gray)
        x1 = max(center_x - pattern_width // 2 - search_margin_x, 0)
        x2 = min(center_x + pattern_width // 2 + search_margin_x, gray.shape[1])
        strip = gray[:, x1:x2]

        top_y = match_pattern(strip, selected_patterns['top'], (10, gray.shape[0] // 3))
        bottom_y = match_pattern(strip, selected_patterns['bottom'], (2 * gray.shape[0] // 3, gray.shape[0] - 10))

        if top_y is None or bottom_y is None:
            print(f"⚠ Skipping frame {frame}: no match found.")
            continue

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

        timestamp = frame.split('_')[-1].replace('s.jpg', '').replace('s.png', '')
        data.append({
            "frame": frame,
            "timestamp_s": float(timestamp),
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