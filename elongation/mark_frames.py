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
    color_correction=(255, 0, 255), # Purple
    pattern_height=10,
    pattern_width=10,
    fallback_search_radius=50,
    selection_weights={'statistical': 0.3, 'confidence': 0.3, 'temporal': 0.4},
    outlier_threshold=2.0,
    temporal_smoothing_window=3,
    enable_temporal_smoothing=True,
    min_contrast=10,
    min_edge_strength=5,
    progress_callback=None,
    cancel_event=None,
    pattern_top_grid=5,
    pattern_bottom_grid=4
):
    """
    pattern_top_grid: int (default 6) - grid line (0=bottom, 10=top) for top pattern extraction
    pattern_bottom_grid: int (default 3) - grid line (0=bottom, 10=top) for bottom pattern extraction
    """
    def notify_progress(p, msg):
        if progress_callback:
            progress_callback(p, msg)
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

    def select_best_elongation(elongations, scores, pattern_qualities, previous_elongation=None, 
                              weights={'statistical': 0.4, 'confidence': 0.3, 'temporal': 0.3}):
        """
        Select best elongation using hybrid approach combining multiple criteria.
        
        Args:
            elongations: List of 6 elongation percentages [all possible combinations]
            scores: List of 6 OpenCV correlation scores [all possible combinations]
            pattern_qualities: List of 6 pattern quality scores
            previous_elongation: Previous frame's elongation percentage (optional)
            weights: Dictionary of weights for each criterion
            
        Returns:
            best_elongation: Selected elongation percentage
            best_idx: Index of selected elongation
            confidence_score: Overall confidence in the selection (0-1)
        """
        if len(elongations) != 6 or len(scores) != 6 or len(pattern_qualities) != 6:
            return elongations[0] if elongations else 0.0, 0, 0.5  # Fallback to first value
        
        # Filter out values under 100%
        valid_indices = [i for i, e in enumerate(elongations) if e is not None and e >= 100.0]
        if not valid_indices:
            # If no values >= 100%, use all valid values
            valid_indices = [i for i, e in enumerate(elongations) if e is not None]
        
        if not valid_indices:
            return elongations[0] if elongations else 0.0, 0, 0.5
        
        valid_elongations = [elongations[i] for i in valid_indices]
        valid_scores = [scores[i] for i in valid_indices]
        valid_qualities = [pattern_qualities[i] for i in valid_indices]
        
        # Detect outliers
        outlier_flags = detect_outliers(valid_elongations, outlier_threshold)
        non_outlier_indices = [i for i, is_outlier in enumerate(outlier_flags) if not is_outlier]
        
        if not non_outlier_indices:
            # If all are outliers, use all values
            non_outlier_indices = list(range(len(valid_elongations)))
        
        final_elongations = [valid_elongations[i] for i in non_outlier_indices]
        final_scores = [valid_scores[i] for i in non_outlier_indices]
        final_qualities = [valid_qualities[i] for i in non_outlier_indices]
        
        if not final_elongations:
            return elongations[0] if elongations else 0.0, 0, 0.5
        
        # Normalize scores to 0-1 range
        max_score = max(final_scores)
        min_score = min(final_scores)
        if max_score == min_score:
            normalized_scores = [0.5] * len(final_scores)  # All equal
        else:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in final_scores]
        
        # Normalize pattern qualities
        max_quality = max(final_qualities)
        min_quality = min(final_qualities)
        if max_quality == min_quality:
            normalized_qualities = [0.5] * len(final_qualities)
        else:
            normalized_qualities = [(q - min_quality) / (max_quality - min_quality) for q in final_qualities]
        
        # 1. Statistical criterion: Distance from median
        median_elongation = np.median(final_elongations)
        max_elongation = max(final_elongations)
        statistical_scores = [1.0 - min(abs(e - median_elongation) / max_elongation, 1.0) for e in final_elongations]
        
        # 2. Enhanced confidence criterion: Combine OpenCV scores and pattern quality
        confidence_scores = [(ns + nq) / 2 for ns, nq in zip(normalized_scores, normalized_qualities)]
        
        # 3. Temporal criterion: Consistency with previous frame
        if previous_elongation is not None:
            # Calculate reasonable change threshold (e.g., 5% of previous value)
            change_threshold = max(previous_elongation * 0.05, 2.0)  # At least 2% or 5% of previous
            temporal_scores = [1.0 - min(abs(e - previous_elongation) / change_threshold, 1.0) for e in final_elongations]
        else:
            temporal_scores = [0.5] * len(final_elongations)  # Neutral if no previous frame
        
        # Combine all criteria with weights
        combined_scores = []
        for i in range(len(final_elongations)):
            combined_score = (
                weights['statistical'] * statistical_scores[i] +
                weights['confidence'] * confidence_scores[i] +
                weights['temporal'] * temporal_scores[i]
            )
            combined_scores.append(combined_score)
        
        # Select best elongation
        best_idx = np.argmax(combined_scores)
        best_elongation = final_elongations[best_idx]
        confidence_score = combined_scores[best_idx]
        
        # Map back to original indices
        original_best_idx = valid_indices[non_outlier_indices[best_idx]]
        
        return best_elongation, original_best_idx, confidence_score

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
    notify_progress(0.1,f"🔎 Total usable frames after skip: {len(frames)}")

    # Precompute all images
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

    # Extract patterns from first frame
    notify_progress(0.2,"🗕 Extracting patterns from first frame...")
    first_frame = frames[0]
    img, gray = precomputed[first_frame]
    center_x = detect_rebar_center_x(gray)
    h = gray.shape[0]
    
    # Calculate pattern positions using grid system
    y_grid_7 = int((10 - 7) * h / 10)
    y_grid_6 = int((10 - 6) * h / 10)
    y_grid_5 = int((10 - 5) * h / 10)
    y_grid_4 = int((10 - 4) * h / 10)
    y_grid_3 = int((10 - 3) * h / 10)
    
    print(f"[DEBUG] Pattern extraction positions: grid_7={y_grid_7}, grid_6={y_grid_6}, grid_5={y_grid_5}, grid_4={y_grid_4}, grid_3={y_grid_3}")
    
    # Extract patterns
    pattern_7 = extract_pattern(gray, center_x, y_grid_7)
    pattern_6 = extract_pattern(gray, center_x, y_grid_6)
    pattern_5 = extract_pattern(gray, center_x, y_grid_5)
    pattern_4 = extract_pattern(gray, center_x, y_grid_4)
    pattern_3 = extract_pattern(gray, center_x, y_grid_3)
    
    # Validate patterns
    patterns = [pattern_7, pattern_6, pattern_5, pattern_4, pattern_3]
    pattern_names = ['7', '6', '5', '4', '3']
    valid_patterns = []
    
    for i, (pattern, name) in enumerate(zip(patterns, pattern_names)):
        if pattern is not None and validate_pattern(pattern, min_contrast, min_edge_strength):
            valid_patterns.append((pattern, name))
            print(f"[DEBUG] Pattern {name} validated successfully")
        else:
            print(f"[DEBUG] Pattern {name} failed validation")
    
    if len(valid_patterns) < 3:
        notify_progress(1.0,"❌ Could not extract enough valid patterns. Exiting.")
        return

    # Find reference positions from first frame
    x1 = max(center_x - pattern_width // 2, 0)
    x2 = min(center_x + pattern_width // 2, gray.shape[1])
    strip = gray[:, x1:x2]
    
    ref_y_7, _ = find_pattern_near_reference(strip, pattern_7, y_grid_7)
    ref_y_6, _ = find_pattern_near_reference(strip, pattern_6, y_grid_6)
    ref_y_5, _ = find_pattern_near_reference(strip, pattern_5, y_grid_5)
    ref_y_4, _ = find_pattern_near_reference(strip, pattern_4, y_grid_4)
    ref_y_3, _ = find_pattern_near_reference(strip, pattern_3, y_grid_3)
    
    if ref_y_7 is None or ref_y_6 is None or ref_y_5 is None or ref_y_4 is None or ref_y_3 is None:
        notify_progress(1.0,"❌ Could not establish all reference positions. Exiting.")
        return

    # Calculate reference gauge lengths for all valid combinations (min gap = 2)
    ref_gauge_lengths = {
        '7_to_5': ref_y_5 - ref_y_7,  # gap = 2
        '7_to_4': ref_y_4 - ref_y_7,  # gap = 3
        '7_to_3': ref_y_3 - ref_y_7,  # gap = 4
        '6_to_4': ref_y_4 - ref_y_6,  # gap = 2
        '6_to_3': ref_y_3 - ref_y_6,  # gap = 3
        '5_to_3': ref_y_3 - ref_y_5   # gap = 2
    }
    
    print(f"[DEBUG] Reference positions - Grid 7: {ref_y_7}, Grid 6: {ref_y_6}, Grid 5: {ref_y_5}, Grid 4: {ref_y_4}, Grid 3: {ref_y_3}")
    print(f"[DEBUG] Reference gauge lengths: {ref_gauge_lengths}")

    data = []
    previous_elongation_percent = None
    elongation_history = []  # For temporal smoothing
    
    print("🖼 Drawing overlays and saving output frames...")
    for frame_idx, frame in enumerate(tqdm(frames, desc="Processing frames")):
        if cancel_event and cancel_event.is_set():
            notify_progress(1.0, "Processing cancelled by user.")
            break

        img, gray = precomputed[frame]
        center_x = detect_rebar_center_x(gray)
        
        x1 = max(center_x - pattern_width // 2, 0)
        x2 = min(center_x + pattern_width // 2, gray.shape[1])
        strip = gray[:, x1:x2]

        # Find current positions for all 5 patterns
        y_7, score_7 = find_pattern_near_reference(strip, pattern_7, ref_y_7)
        y_6, score_6 = find_pattern_near_reference(strip, pattern_6, ref_y_6)
        y_5, score_5 = find_pattern_near_reference(strip, pattern_5, ref_y_5)
        y_4, score_4 = find_pattern_near_reference(strip, pattern_4, ref_y_4)
        y_3, score_3 = find_pattern_near_reference(strip, pattern_3, ref_y_3)

        # Calculate pattern qualities
        pattern_qualities = []
        for pattern in [pattern_7, pattern_6, pattern_5, pattern_4, pattern_3]:
            quality = calculate_pattern_quality(pattern)
            pattern_qualities.append(quality)

        # Calculate all 6 possible elongation measurements (min gap = 2)
        elongations = []
        scores = []
        positions = []  # Store positions for each measurement
        measurement_names = []  # For debugging
        combined_qualities = []  # Combined pattern qualities for each measurement
        
        # 1. 7 → 5 (gap = 2)
        if all([y_7 is not None, y_5 is not None]):
            assert y_7 is not None and y_5 is not None
            elongation_7_to_5 = y_5 - y_7
            elongation_7_to_5_percent = safe_elongation_calculation(y_5, y_7, ref_gauge_lengths['7_to_5'])
            elongations.append(elongation_7_to_5_percent)
            scores.append((score_7 or 0.0) + (score_5 or 0.0))
            positions.append((y_7, y_5))
            measurement_names.append("7→5")
            combined_qualities.append((pattern_qualities[0] + pattern_qualities[2]) / 2)  # 7 and 5
        else:
            elongations.append(None)
            scores.append(0.0)
            positions.append((None, None))
            measurement_names.append("7→5")
            combined_qualities.append(0.0)
        
        # 2. 7 → 4 (gap = 3)
        if all([y_7 is not None, y_4 is not None]):
            assert y_7 is not None and y_4 is not None
            elongation_7_to_4 = y_4 - y_7
            elongation_7_to_4_percent = safe_elongation_calculation(y_4, y_7, ref_gauge_lengths['7_to_4'])
            elongations.append(elongation_7_to_4_percent)
            scores.append((score_7 or 0.0) + (score_4 or 0.0))
            positions.append((y_7, y_4))
            measurement_names.append("7→4")
            combined_qualities.append((pattern_qualities[0] + pattern_qualities[3]) / 2)  # 7 and 4
        else:
            elongations.append(None)
            scores.append(0.0)
            positions.append((None, None))
            measurement_names.append("7→4")
            combined_qualities.append(0.0)
        
        # 3. 7 → 3 (gap = 4)
        if all([y_7 is not None, y_3 is not None]):
            assert y_7 is not None and y_3 is not None
            elongation_7_to_3 = y_3 - y_7
            elongation_7_to_3_percent = safe_elongation_calculation(y_3, y_7, ref_gauge_lengths['7_to_3'])
            elongations.append(elongation_7_to_3_percent)
            scores.append((score_7 or 0.0) + (score_3 or 0.0))
            positions.append((y_7, y_3))
            measurement_names.append("7→3")
            combined_qualities.append((pattern_qualities[0] + pattern_qualities[4]) / 2)  # 7 and 3
        else:
            elongations.append(None)
            scores.append(0.0)
            positions.append((None, None))
            measurement_names.append("7→3")
            combined_qualities.append(0.0)
        
        # 4. 6 → 4 (gap = 2)
        if all([y_6 is not None, y_4 is not None]):
            assert y_6 is not None and y_4 is not None
            elongation_6_to_4 = y_4 - y_6
            elongation_6_to_4_percent = safe_elongation_calculation(y_4, y_6, ref_gauge_lengths['6_to_4'])
            elongations.append(elongation_6_to_4_percent)
            scores.append((score_6 or 0.0) + (score_4 or 0.0))
            positions.append((y_6, y_4))
            measurement_names.append("6→4")
            combined_qualities.append((pattern_qualities[1] + pattern_qualities[3]) / 2)  # 6 and 4
        else:
            elongations.append(None)
            scores.append(0.0)
            positions.append((None, None))
            measurement_names.append("6→4")
            combined_qualities.append(0.0)

        # 5. 6 → 3 (gap = 3)
        if all([y_6 is not None, y_3 is not None]):
            assert y_6 is not None and y_3 is not None
            elongation_6_to_3 = y_3 - y_6
            elongation_6_to_3_percent = safe_elongation_calculation(y_3, y_6, ref_gauge_lengths['6_to_3'])
            elongations.append(elongation_6_to_3_percent)
            scores.append((score_6 or 0.0) + (score_3 or 0.0))
            positions.append((y_6, y_3))
            measurement_names.append("6→3")
            combined_qualities.append((pattern_qualities[1] + pattern_qualities[4]) / 2)  # 6 and 3
        else:
            elongations.append(None)
            scores.append(0.0)
            positions.append((None, None))
            measurement_names.append("6→3")
            combined_qualities.append(0.0)
        
        # 6. 5 → 3 (gap = 2)
        if all([y_5 is not None, y_3 is not None]):
            assert y_5 is not None and y_3 is not None
            elongation_5_to_3 = y_3 - y_5
            elongation_5_to_3_percent = safe_elongation_calculation(y_3, y_5, ref_gauge_lengths['5_to_3'])
            elongations.append(elongation_5_to_3_percent)
            scores.append((score_5 or 0.0) + (score_3 or 0.0))
            positions.append((y_5, y_3))
            measurement_names.append("5→3")
            combined_qualities.append((pattern_qualities[2] + pattern_qualities[4]) / 2)  # 5 and 3
        else:
            elongations.append(None)
            scores.append(0.0)
            positions.append((None, None))
            measurement_names.append("5→3")
            combined_qualities.append(0.0)

        # Select best elongation using hybrid approach
        valid_elongations = [e for e in elongations if e is not None]
        valid_scores = [s for i, s in enumerate(scores) if elongations[i] is not None]
        valid_qualities = [combined_qualities[i] for i, e in enumerate(elongations) if e is not None]
        valid_names = [measurement_names[i] for i, e in enumerate(elongations) if e is not None]
        
        if len(valid_elongations) > 0:
            best_elongation_percent, best_idx, confidence_score = select_best_elongation(
                elongations, scores, combined_qualities, previous_elongation_percent, selection_weights
            )
            
            # Get the selected positions
            selected_top_y, selected_bottom_y = positions[best_idx]
            selected_measurement_name = measurement_names[best_idx]
            
            # Apply temporal smoothing if enabled
            if enable_temporal_smoothing:
                elongation_history.append(best_elongation_percent)
                if len(elongation_history) > temporal_smoothing_window:
                    elongation_history.pop(0)
                smoothed_elongation = apply_temporal_smoothing(elongation_history, temporal_smoothing_window)
                best_elongation_percent = smoothed_elongation
            
            print(f"[DEBUG] Frame {frame}: Selected measurement {best_idx} ({selected_measurement_name}) with confidence {confidence_score:.3f}")
            print(f"[DEBUG] Frame {frame}: All elongations: {[f'{m}={e:.2f}%' if e is not None else f'{m}=None' for m, e in zip(measurement_names, elongations)]}")
        else:
            # Fallback: try wider search radius
            if y_7 is None:
                y_7, _ = find_pattern_near_reference(strip, pattern_7, ref_y_7, fallback_search_radius)
            if y_5 is None:
                y_5, _ = find_pattern_near_reference(strip, pattern_5, ref_y_5, fallback_search_radius)
            
            if y_7 is None or y_5 is None:
                continue
                
            selected_top_y, selected_bottom_y = y_7, y_5
            best_elongation_percent = safe_elongation_calculation(y_5, y_7, ref_gauge_lengths['7_to_5'])
            confidence_score = 0.5  # Low confidence for fallback
            best_idx = 0
            selected_measurement_name = "7→5 (fallback)"

        # Update previous elongation for next frame
        previous_elongation_percent = best_elongation_percent

        # Calculate final elongation using dynamic gauge length
        selected_gauge_length = ref_gauge_lengths[selected_measurement_name.replace('→', '_to_').replace(' (fallback)', '')]
        elongation_px = selected_bottom_y - selected_top_y - selected_gauge_length
        elongation_percent = best_elongation_percent

        # Draw reference grid (no yellow lines)
        cv2.line(img, (center_x, 0), (center_x, img.shape[0]), color_middle, 1)
        draw_reference_grid(img, int(round(ref_y_7)), int(round(ref_y_5)), selected_gauge_length)

        # Draw blue lines at the selected positions
        cv2.line(img, (0, int(round(selected_top_y))), (img.shape[1], int(round(selected_top_y))), color_curr, 2)
        cv2.line(img, (0, int(round(selected_bottom_y))), (img.shape[1], int(round(selected_bottom_y))), color_curr, 2)
        cv2.putText(img, f"{(selected_bottom_y - selected_top_y):.2f}px ({elongation_percent:.2f}%)", (img.shape[1] - 250, int(round(selected_top_y)) - 10), font, font_scale, color_curr, 2)

        output_path = os.path.join(output_folder, frame)
        cv2.imwrite(output_path, img)

        timestamp_str = os.path.splitext(frame)[0].split('_')[-1].replace('s', '')
        data.append({
            "frame": frame,
            "timestamp_s": float(timestamp_str),
            "ref_top_px": ref_y_7,
            "ref_bottom_px": ref_y_5,
            "curr_top_px": selected_top_y,
            "curr_bottom_px": selected_bottom_y,
            "elongation_px": elongation_px,
            "elongation_percent": elongation_percent,
            "selected_measurement": selected_measurement_name,
            "confidence_score": confidence_score,
            "measurements_available": len(valid_elongations)
        })

    if cancel_event and cancel_event.is_set():
        notify_progress(1.0, "Processing cancelled by user.")
        return
        
    pd.DataFrame(data).to_csv(csv_output_path, index=False)
    notify_progress(1.0,f"✔ All done! Marked frames in '{output_folder}', data in '{csv_output_path}'")

if __name__ == "__main__":
    process_images("output_frames", "elongation_marked_frames", "elongation_data.csv")
