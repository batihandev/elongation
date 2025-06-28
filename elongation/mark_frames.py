import os
import cv2
import numpy as np
import pandas as pd
from elongation.detect_markers import detect_markers
from tqdm import tqdm
from elongation.subpixel import subpixel_refine
from sklearn.cluster import DBSCAN

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
    color_correction=(255, 0, 255), # Purple
    pattern_height=15,
    pattern_width=15,
    fallback_search_radius=50,
    selection_weights={'statistical': 0, 'confidence': 0, 'temporal': 1},

    min_contrast=10,
    min_edge_strength=5,
    progress_callback=None,
    cancel_event=None,
    pattern_top_grid=7,
    pattern_bottom_grid=3,
    num_divisions=75,  # NEW: number of grid points (patterns) to extract
    min_gap=25,        # NEW: minimum gap (in grid indices) between patterns for measurement
    dbscan_eps=1     # DBSCAN clustering range for elongation grouping
):
    """
    pattern_top_grid: int (default 6) - grid line (0=bottom, 10=top) for top pattern extraction
    pattern_bottom_grid: int (default 3) - grid line (0=bottom, 10=top) for bottom pattern extraction
    num_divisions: int - number of grid points (patterns) to extract between top and bottom
    min_gap: int - minimum gap (in grid indices) between patterns for measurement
    dbscan_eps: float - DBSCAN clustering range for elongation grouping
    """
    # tqdm bar for main progress
    main_bar = None
    def notify_progress(p, msg):
        nonlocal main_bar
        if progress_callback:
            progress_callback(p, msg)
        elif main_bar is not None:
            main_bar.set_postfix_str(msg)
        else:
            print(f"{int(p*100)}% - {msg}")

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

    def extract_pattern(gray, center_x, y_position):
        h, w = gray.shape
        x1 = max(center_x - pattern_width // 2, 0)
        x2 = min(center_x + pattern_width // 2, w)
        strip = gray[:, x1:x2]

        if y_position + pattern_height <= h:
            pattern = strip[y_position:y_position + pattern_height, :].copy()
            return pattern
        return None

    def validate_pattern(pattern, min_contrast=10, min_edge_strength=5):
        """Validate pattern quality before using it"""
        if pattern is None:
            return False
        
        contrast = np.std(pattern)
        edge_strength = np.mean(np.abs(cv2.Sobel(pattern, cv2.CV_64F, 1, 0, ksize=3)))
        
        return contrast > min_contrast and edge_strength > min_edge_strength

    def calculate_pattern_quality(pattern):
        """Calculate pattern quality based on contrast and edge strength"""
        if pattern is None:
            return 0.0
        
        # Calculate local contrast
        contrast = np.std(pattern)
        # Calculate edge strength using Sobel
        sobel_x = cv2.Sobel(pattern, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(pattern, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
        return contrast * edge_strength

    def calculate_adaptive_search_radius(pattern, base_radius=20):
        """Calculate search radius based on pattern characteristics"""
        if pattern is None:
            return base_radius
        
        pattern_height, pattern_width = pattern.shape
        # Larger patterns need larger search radius
        size_factor = max(pattern_height, pattern_width) / 10.0
        return int(base_radius * size_factor)

    def find_pattern_near_reference(strip, pattern, reference_y, max_search_distance=20):
        h = strip.shape[0]
        
        # Use adaptive search radius
        adaptive_radius = calculate_adaptive_search_radius(pattern, max_search_distance)
        
        # Convert reference_y to int for slice indexing
        reference_y_int = int(round(reference_y))
        
        # Define search window around reference_y
        y1 = max(0, reference_y_int - adaptive_radius)
        y2 = min(h, reference_y_int + adaptive_radius + pattern.shape[0])
        search_strip = strip[y1:y2, :]
        
        if search_strip.shape[0] < pattern.shape[0]:
            return None, None

        # Use matchTemplate
        search_strip_f = search_strip.astype(np.float32)
        pattern_f = pattern.astype(np.float32)
        res = cv2.matchTemplate(search_strip_f, pattern_f, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        best_y_local = max_loc[1]  # y in the local search_strip
        best_y = y1 + best_y_local

        # Subpixel refinement
        if 0 < best_y_local < res.shape[0] - 1:
            y_vals = np.array([best_y_local - 1, best_y_local, best_y_local + 1])
            scores = res[y_vals, 0]
            denom = 2 * (scores[0] - 2 * scores[1] + scores[2])
            if denom != 0:
                delta = (scores[0] - scores[2]) / denom
                best_y_subpixel = best_y + delta
            else:
                best_y_subpixel = float(best_y)
        else:
            best_y_subpixel = float(best_y)

        best_score = max_val
        return best_y_subpixel, best_score

    def detect_outliers(elongations, threshold=2.0):
        """Detect statistical outliers using z-score"""
        if len(elongations) < 3:
            return [False] * len(elongations)
        
        mean_val = np.mean(elongations)
        std_val = np.std(elongations)
        if std_val == 0:
            return [False] * len(elongations)
        
        z_scores = [abs((e - mean_val) / std_val) for e in elongations]
        return [z > threshold for z in z_scores]

    def safe_elongation_calculation(current_pos, ref_pos, ref_gauge_length):
        """Safely calculate elongation with error handling"""
        try:
            if current_pos is None or ref_pos is None or ref_gauge_length == 0:
                return None
            return ((current_pos - ref_pos) / ref_gauge_length) * 100
        except (ZeroDivisionError, TypeError):
            return None

    def apply_temporal_smoothing(elongation_history, window_size=3):
        """Apply moving average smoothing"""
        if len(elongation_history) < window_size:
            return elongation_history[-1] if elongation_history else 0
        
        recent_values = elongation_history[-window_size:]
        return np.mean(recent_values)

    def select_best_elongation(elongations, scores, pattern_qualities, previous_elongation=None, weights=None):
        """
        Select the minimum elongation value that is higher than the previous frame's elongation (if any),
        otherwise select the minimum value. Label as 'min_increase' if a higher value is found, otherwise 'min'.
        """
        valid_indices = [i for i, e in enumerate(elongations) if e is not None]
        if not valid_indices:
            return 0.0, 0, 0.0, 'min'
        valid_elongations = [elongations[i] for i in valid_indices]
        if previous_elongation is not None:
            higher_indices = [i for i, e in zip(valid_indices, valid_elongations) if e > previous_elongation]
            if higher_indices:
                min_idx = min(higher_indices, key=lambda i: elongations[i])
                return elongations[min_idx], min_idx, 1.0, 'min_increase'
        # fallback: just select minimum
        min_idx = min(valid_indices, key=lambda i: elongations[i])
        return elongations[min_idx], min_idx, 1.0, 'min'

    def draw_reference_grid(img, ref_top, ref_bottom, ref_distance):
        h, w = img.shape[:2]
        # Draw 10 white grid lines evenly spaced across the entire image height (0=bottom, 9=near top)
        for i in range(10):  # Only 0-9, do not draw the 10th (topmost) line
            y = int((9 - i) * h / 9)  # 0 at bottom, 9 near top
            cv2.line(img, (0, y), (w, y), color_grid, 1)
            # Draw grid numbers (0=bottom, 9=near top) at left margin
            cv2.putText(img, str(i), (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if cancel_event and cancel_event.is_set():
        notify_progress(1.0, "Processing cancelled by user.")
        return

    os.makedirs(output_folder, exist_ok=True)
    frames = sorted(f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png')))
    frames = frames[skip_start_frames: len(frames) - skip_end_frames]
    main_bar = tqdm(total=len(frames), desc="Marking frames", unit="frame")
    notify_progress(0.1, f"🔎 Total usable frames after skip: {len(frames)}")

    # Precompute all images
    precomputed = {}
    for frame in tqdm(frames, desc="Precomputing grayscale", leave=False):
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

    # Extract patterns from first frame
    notify_progress(0.2, "Extracting patterns from first frame...")
    first_frame = frames[0]
    img, gray = precomputed[first_frame]
    center_x = detect_rebar_center_x(gray)
    h = gray.shape[0]

    # Generate grid positions (from top to bottom)
    grid_indices = np.linspace(pattern_top_grid, pattern_bottom_grid, num_divisions)
    y_grids = [int((10 - g) * h / 10) for g in grid_indices]

    # Extract patterns at each grid position
    patterns = [extract_pattern(gray, center_x, y) for y in y_grids]
    pattern_names = [f"{g:.2f}" for g in grid_indices]
    valid_patterns = []
    for i, (pattern, name) in enumerate(zip(patterns, pattern_names)):
        if pattern is not None and validate_pattern(pattern, min_contrast, min_edge_strength):
            valid_patterns.append((pattern, name, y_grids[i]))
    if len(valid_patterns) < 2:
        notify_progress(1.0, "❌ Could not extract enough valid patterns. Exiting.")
        return

    # Find reference positions from first frame
    x1 = max(center_x - pattern_width // 2, 0)
    x2 = min(center_x + pattern_width // 2, gray.shape[1])
    strip = gray[:, x1:x2]
    ref_ys = []
    for pattern, name, y_grid in valid_patterns:
        ref_y, _ = find_pattern_near_reference(strip, pattern, y_grid)
        ref_ys.append(ref_y)
    if any(y is None for y in ref_ys):
        notify_progress(1.0, "❌ Could not establish all reference positions. Exiting.")
        return

    # Calculate reference gauge lengths for all valid pairs (min_gap)
    ref_gauge_lengths = {}
    pattern_count = len(valid_patterns)
    for i in range(pattern_count):
        for j in range(i + min_gap, pattern_count):
            key = f"{i}_to_{j}"
            ref_gauge_lengths[key] = ref_ys[j] - ref_ys[i]

    data = []
    previous_elongation_percent = None
    elongation_history = []  # For temporal smoothing

    for frame_idx, frame in enumerate(tqdm(frames, desc="Processing frames", leave=False)):
        if cancel_event and cancel_event.is_set():
            notify_progress(1.0, "Processing cancelled by user.")
            break
        img, gray = precomputed[frame]
        center_x = detect_rebar_center_x(gray)
        x1 = max(center_x - pattern_width // 2, 0)
        x2 = min(center_x + pattern_width // 2, gray.shape[1])
        strip = gray[:, x1:x2]
        # Find current positions for all patterns
        curr_ys = []
        curr_scores = []
        for (pattern, name, y_grid) in valid_patterns:
            y, score = find_pattern_near_reference(strip, pattern, y_grid)
            curr_ys.append(y)
            curr_scores.append(score)
        # Calculate pattern qualities
        pattern_qualities = [calculate_pattern_quality(pattern) for (pattern, _, _) in valid_patterns]
        # Calculate all possible elongation measurements (min_gap)
        pair_indices = [(i, j) for i in range(pattern_count) for j in range(i + min_gap, pattern_count)]
        num_pairs = len(pair_indices)
        min_samples = max(2, int(num_pairs * 0.35))
        if frame_idx == 0:
            print(f"  Total pairs to calculate: {num_pairs}")
            print(f"  DBSCAN min_samples (35% of pairs): {min_samples}")
        elongations = []
        scores = []
        positions = []
        measurement_names = []
        combined_qualities = []
        for i, j in tqdm(pair_indices, desc='Pairwise elongation', unit='pair', leave=False):
            if curr_ys[i] is not None and curr_ys[j] is not None:
                elongation = curr_ys[j] - curr_ys[i]
                key = f"{i}_to_{j}"
                elongation_percent = safe_elongation_calculation(curr_ys[j], curr_ys[i], ref_gauge_lengths[key])
                elongations.append(elongation_percent)
                scores.append((curr_scores[i] or 0.0) + (curr_scores[j] or 0.0))
                positions.append((curr_ys[i], curr_ys[j]))
                measurement_names.append(key)
                combined_qualities.append((pattern_qualities[i] + pattern_qualities[j]) / 2)
            else:
                elongations.append(None)
                scores.append(0.0)
                positions.append((None, None))
                measurement_names.append(f"{i}_to_{j}")
                combined_qualities.append(0.0)
        # Select best elongation using hybrid approach
        valid_elongations = [e for e in elongations if e is not None]
        valid_scores = [s for idx, s in enumerate(scores) if elongations[idx] is not None]
        valid_qualities = [combined_qualities[idx] for idx, e in enumerate(elongations) if e is not None]
        valid_names = [measurement_names[idx] for idx, e in enumerate(elongations) if e is not None]
        if len(valid_elongations) > 0:
            # --- Adaptive DBSCAN clustering on elongation values ---
            elong_array = np.array([e for e in elongations if e is not None]).reshape(-1, 1)
            if len(elong_array) > 0:
                default_eps = dbscan_eps
                max_eps = 5.0
                eps = default_eps
                found_cluster = False
                while True:
                    db = DBSCAN(eps=eps, min_samples=min_samples).fit(elong_array)
                    labels, counts = np.unique(db.labels_, return_counts=True)
                    valid_labels = labels[labels != -1]
                    if len(valid_labels) > 0:
                        # Find the largest cluster (by member count)
                        main_label = valid_labels[np.argmax(counts[labels != -1])]
                        main_group = elong_array[db.labels_ == main_label].flatten()
                        median_elong = float(np.median(main_group))
                        best_elongation_percent = median_elong
                        confidence_score = 1.0
                        selection_type = 'dbscan_median'
                        found_cluster = True
                        break
                    else:
                        eps *= 2
                        if eps > max_eps:
                            median_elong = float(np.min(elong_array))
                            best_elongation_percent = median_elong
                            confidence_score = 1.0
                            selection_type = 'min_fallback'
                            break
            else:
                median_elong = 0.0
                best_elongation_percent = median_elong
                confidence_score = 1.0
                selection_type = 'empty'
            # Find the pair whose elongation is closest to the median
            if len(elong_array) > 0:
                closest_idx = np.argmin(np.abs(elong_array.flatten() - median_elong))
                selected_measurement_name = measurement_names[closest_idx]
                selected_top_y, selected_bottom_y = positions[closest_idx]
                # Get cluster member count for display
                if selection_type == 'dbscan_median':
                    cluster_members = len(main_group)
                    cluster_info = f"{cluster_members}/{num_pairs}"
                else:
                    cluster_info = f"1/{num_pairs}"
            else:
                selected_measurement_name = 'none'
                selected_top_y, selected_bottom_y = 0, 0
                cluster_info = f"0/{num_pairs}"
        else:
            # Fallback: try wider search radius for first and last pattern
            y_first, _ = find_pattern_near_reference(strip, valid_patterns[0][0], valid_patterns[0][2], fallback_search_radius)
            y_last, _ = find_pattern_near_reference(strip, valid_patterns[-1][0], valid_patterns[-1][2], fallback_search_radius)
            if y_first is None or y_last is None:
                continue
            selected_top_y, selected_bottom_y = y_first, y_last
            key = f"0_to_{pattern_count-1}"
            best_elongation_percent = safe_elongation_calculation(y_last, y_first, ref_gauge_lengths[key])
            confidence_score = 0.5
            best_idx = 0
            selected_measurement_name = key + " (fallback)"
            selection_type = "fallback"  # Add selection type for fallback case
            main_bar.set_postfix({
                "frame": frame,
                "meas": selected_measurement_name,
                "elong%": f"{best_elongation_percent:.2f}",
                "conf": f"{confidence_score:.2f}",
                "selection_type": selection_type
            })
        previous_elongation_percent = best_elongation_percent
        selected_gauge_length = ref_gauge_lengths[selected_measurement_name.replace(' (fallback)', '')]
        elongation_px = selected_bottom_y - selected_top_y - selected_gauge_length
        elongation_percent = best_elongation_percent
        cv2.line(img, (center_x, 0), (center_x, img.shape[0]), color_middle, 1)
        draw_reference_grid(img, int(round(ref_ys[0])), int(round(ref_ys[-1])), selected_gauge_length)
        cv2.line(img, (0, int(round(selected_top_y))), (img.shape[1], int(round(selected_top_y))), color_curr, 2)
        cv2.line(img, (0, int(round(selected_bottom_y))), (img.shape[1], int(round(selected_bottom_y))), color_curr, 2)
        
        # Display only the elongation percentage, no cluster_type text
        selection_text = f"{(selected_bottom_y - selected_top_y):.2f}px ({elongation_percent:.2f}%) {cluster_info}"
        # Center the text horizontally
        text_size = cv2.getTextSize(selection_text, font, font_scale, 2)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        cv2.putText(img, selection_text, (text_x, int(round(selected_top_y)) - 10), font, font_scale, color_curr, 2)
        output_path = os.path.join(output_folder, frame)
        cv2.imwrite(output_path, img)
        timestamp_str = os.path.splitext(frame)[0].split('_')[-1].replace('s', '')
        data.append({
            "frame": frame,
            "timestamp_s": float(timestamp_str),
            "ref_top_px": ref_ys[0],
            "ref_bottom_px": ref_ys[-1],
            "curr_top_px": selected_top_y,
            "curr_bottom_px": selected_bottom_y,
            "elongation_px": elongation_px,
            "elongation_percent": elongation_percent,
            "selected_measurement": selected_measurement_name,
            "confidence_score": confidence_score,
            "selection_type": selection_type,
            "measurements_available": len(valid_elongations)
        })
        main_bar.update(1)
    if cancel_event and cancel_event.is_set():
        notify_progress(1.0, "Processing cancelled by user.")
        return
    pd.DataFrame(data).to_csv(csv_output_path, index=False)
    notify_progress(1.0, f"✔ All done! Marked frames in '{output_folder}', data in '{csv_output_path}'")
    main_bar.close()

if __name__ == "__main__":
    process_images("output_frames", "elongation_marked_frames", "elongation_data.csv")
