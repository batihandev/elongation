import os
import cv2
import numpy as np
from tqdm import tqdm

def detect_focus_quality(frame, threshold=50):
    """
    Detect if a frame is in focus using Laplacian variance.
    Higher variance indicates sharper (more focused) image.
    
    Args:
        frame: Input frame (BGR or grayscale)
        threshold: Minimum variance threshold for "in focus"
    
    Returns:
        tuple: (is_focused, focus_score)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Calculate Laplacian variance (measure of sharpness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    focus_score = laplacian.var()
    
    is_focused = focus_score > threshold
    return is_focused, focus_score

def detect_motion_quality(frame, prev_frame=None, motion_threshold=1000):
    """
    Detect if a frame has excessive camera movement/shake.
    Uses optical flow to measure motion between consecutive frames.
    
    Args:
        frame: Current frame (BGR or grayscale)
        prev_frame: Previous frame for comparison
        motion_threshold: Maximum allowed motion magnitude
    
    Returns:
        tuple: (is_stable, motion_score, motion_vectors)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    if prev_frame is None:
        # First frame - assume stable
        return True, 0.0, None
    
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
    
    # Calculate optical flow using Lucas-Kanade method
    # Create a grid of points to track
    h, w = gray.shape
    step = 20  # Check every 20th pixel
    points = np.array([[x, y] for x in range(step, w-step, step) 
                       for y in range(step, h-step, step)], dtype=np.float32)
    
    if len(points) == 0:
        return True, 0.0, None
    
    # Calculate optical flow
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, points, points.copy(),
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Filter valid points - status is a 1D array, not 2D
    valid_indices = status.ravel() == 1
    valid_points = points[valid_indices]
    valid_next_points = next_points[valid_indices]
    
    if len(valid_points) < 10:  # Need minimum points for reliable motion detection
        return True, 0.0, None
    
    # Calculate motion vectors
    motion_vectors = valid_next_points - valid_points
    motion_magnitudes = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)
    
    # Calculate motion statistics
    mean_motion = np.mean(motion_magnitudes)
    max_motion = np.max(motion_magnitudes)
    motion_variance = np.var(motion_magnitudes)
    
    # Combined motion score (weighted combination of mean, max, and variance)
    motion_score = 0.5 * mean_motion + 0.3 * max_motion + 0.2 * motion_variance
    
    is_stable = motion_score < motion_threshold
    
    return is_stable, motion_score, motion_vectors

def detect_edge_movement(frame, prev_frame, edge_threshold=50):
    """
    Detect if scene edges are moving (indicating camera movement).
    Focuses on the edges of the frame rather than the center.
    
    Args:
        frame: Current frame
        prev_frame: Previous frame
        edge_threshold: Threshold for edge movement detection
    
    Returns:
        tuple: (is_stable, edge_movement_score, edge_shift)
    """
    if prev_frame is None:
        return True, 0.0, (0, 0)
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        prev_gray = prev_frame
    
    h, w = gray.shape
    
    # Define edge regions (top, bottom, left, right edges)
    edge_regions = [
        (0, 0, w, h//8),           # Top edge
        (0, 7*h//8, w, h//8),      # Bottom edge
        (0, 0, w//8, h),           # Left edge
        (7*w//8, 0, w//8, h)       # Right edge
    ]
    
    total_edge_movement = 0
    edge_shifts = []
    
    for x, y, ew, eh in edge_regions:
        # Extract edge region
        edge_region = gray[y:y+eh, x:x+ew]
        prev_edge_region = prev_gray[y:y+eh, x:x+ew]
        
        # Calculate correlation to find shift
        if edge_region.shape[0] > 10 and edge_region.shape[1] > 10:
            # Use template matching to find the shift
            result = cv2.matchTemplate(edge_region, prev_edge_region, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            # Calculate shift magnitude
            shift_magnitude = np.sqrt(max_loc[0]**2 + max_loc[1]**2)
            edge_shifts.append(shift_magnitude)
            total_edge_movement += shift_magnitude
    
    avg_edge_movement = total_edge_movement / len(edge_regions) if edge_regions else 0
    is_stable = avg_edge_movement < edge_threshold
    
    return is_stable, avg_edge_movement, edge_shifts

def detect_background_movement(frame, prev_frame, rebar_center_region=None, bg_threshold=30):
    """
    Detect if background is moving while rebar stays relatively stable.
    Segments rebar from background and tracks background movement.
    
    Args:
        frame: Current frame
        prev_frame: Previous frame
        rebar_center_region: Region where rebar is located (x, y, w, h)
        bg_threshold: Threshold for background movement detection
    
    Returns:
        tuple: (is_stable, bg_movement_score, rebar_movement_score)
    """
    if prev_frame is None:
        return True, 0.0, 0.0
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        prev_gray = prev_frame
    
    h, w = gray.shape
    
    # If rebar region not provided, estimate it (center 40% of frame)
    if rebar_center_region is None:
        center_x = w // 2
        center_y = h // 2
        rebar_w = w // 3
        rebar_h = h // 2
        rebar_center_region = (center_x - rebar_w//2, center_y - rebar_h//2, rebar_w, rebar_h)
    
    rx, ry, rw, rh = rebar_center_region
    
    # Extract rebar region
    rebar_region = gray[ry:ry+rh, rx:rx+rw]
    prev_rebar_region = prev_gray[ry:ry+rh, rx:rx+rw]
    
    # Extract background regions (areas outside rebar)
    # Create background mask
    bg_mask = np.ones((h, w), dtype=np.uint8)
    bg_mask[ry:ry+rh, rx:rx+rw] = 0
    
    # Extract background regions
    bg_regions = []
    prev_bg_regions = []
    
    # Top background
    if ry > 0:
        bg_regions.append(gray[0:ry, :])
        prev_bg_regions.append(prev_gray[0:ry, :])
    
    # Bottom background
    if ry + rh < h:
        bg_regions.append(gray[ry+rh:h, :])
        prev_bg_regions.append(prev_gray[ry+rh:h, :])
    
    # Left background
    if rx > 0:
        bg_regions.append(gray[ry:ry+rh, 0:rx])
        prev_bg_regions.append(prev_gray[ry:ry+rh, 0:rx])
    
    # Right background
    if rx + rw < w:
        bg_regions.append(gray[ry:ry+rh, rx+rw:w])
        prev_bg_regions.append(prev_gray[ry:ry+rh, rx+rw:w])
    
    # Calculate background movement
    bg_movements = []
    for bg_region, prev_bg_region in zip(bg_regions, prev_bg_regions):
        if bg_region.size > 100 and prev_bg_region.size > 100:
            # Use template matching to find movement
            result = cv2.matchTemplate(bg_region, prev_bg_region, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            movement = np.sqrt(max_loc[0]**2 + max_loc[1]**2)
            bg_movements.append(movement)
    
    # Calculate rebar movement
    if rebar_region.size > 100:
        result = cv2.matchTemplate(rebar_region, prev_rebar_region, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        rebar_movement = np.sqrt(max_loc[0]**2 + max_loc[1]**2)
    else:
        rebar_movement = 0
    
    avg_bg_movement = np.mean(bg_movements) if bg_movements else 0
    
    # Frame is stable if background movement is low OR rebar movement is similar to background
    # (indicating it's rebar elongation, not camera movement)
    is_stable = avg_bg_movement < bg_threshold or abs(avg_bg_movement - rebar_movement) < bg_threshold/2
    
    return is_stable, avg_bg_movement, rebar_movement

def detect_cumulative_movement(video_path, fps, max_check_frames=300, check_interval=50):
    """
    Detect gradual movement over multiple frames by comparing positions at intervals.
    
    Args:
        video_path: Path to video file
        fps: Frames per second
        max_check_frames: Maximum frames to check
        check_interval: How many frames apart to compare
    
    Returns:
        tuple: (is_stable, cumulative_movement, movement_trend)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Track reference points over time
    reference_positions = []
    frame_indices = []
    
    for frame_idx in range(0, max_check_frames, check_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            break
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Find stable reference points (corners, edges)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=10, qualityLevel=0.01, minDistance=20)
        
        if corners is not None and len(corners) > 0:
            # Use the most stable corner as reference
            reference_pos = corners[0][0]
            reference_positions.append(reference_pos)
            frame_indices.append(frame_idx)
    
    cap.release()
    
    if len(reference_positions) < 3:
        return True, 0.0, "insufficient_data"
    
    # Calculate cumulative movement
    movements = []
    for i in range(1, len(reference_positions)):
        prev_pos = reference_positions[i-1]
        curr_pos = reference_positions[i]
        movement = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
        movements.append(movement)
    
    total_movement = sum(movements)
    avg_movement = np.mean(movements) if movements else 0
    
    # Detect movement trend (direction)
    if len(reference_positions) >= 3:
        # Calculate if movement is consistently in one direction
        x_movements = [reference_positions[i][0] - reference_positions[i-1][0] 
                      for i in range(1, len(reference_positions))]
        y_movements = [reference_positions[i][1] - reference_positions[i-1][1] 
                      for i in range(1, len(reference_positions))]
        
        x_trend = np.mean(x_movements)
        y_trend = np.mean(y_movements)
        
        if abs(y_trend) > abs(x_trend) and y_trend > 0:
            movement_trend = "upward"
        elif abs(y_trend) > abs(x_trend) and y_trend < 0:
            movement_trend = "downward"
        elif abs(x_trend) > abs(y_trend) and x_trend > 0:
            movement_trend = "rightward"
        elif abs(x_trend) > abs(y_trend) and x_trend < 0:
            movement_trend = "leftward"
        else:
            movement_trend = "random"
    else:
        movement_trend = "unknown"
    
    # Determine if movement is significant
    is_stable = total_movement < 100  # Threshold for cumulative movement
    
    return is_stable, total_movement, movement_trend

def detect_orb_rotation(frame, prev_frame, min_matches=10, angle_threshold=0.5, translation_threshold=0.2, frame_gap=1, mask_center=True):
    """
    Detect global rotation/translation between two frames using ORB feature matching.
    - Masks out the central 60% of the image (uses only left/right 20%) if mask_center is True.
    - Lower translation threshold for higher sensitivity.
    - Can compare frames farther apart using frame_gap (default 1 = consecutive).
    Returns (is_stable, angle, translation, num_matches).
    Requires OpenCV with contrib modules (cv2.ORB_create).
    """
    if not hasattr(cv2, 'ORB_create'):
        raise ImportError("cv2.ORB_create not found. Please install opencv-contrib-python.")
    if prev_frame is None:
        return True, 0.0, (0.0, 0.0), 0
    # Optionally mask out the center
    def mask_edges(img):
        h, w = img.shape
        mask = np.zeros_like(img, dtype=np.uint8)
        left = int(w * 0.2)
        right = int(w * 0.8)
        mask[:, :left] = 1
        mask[:, right:] = 1
        return img * mask
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    if mask_center:
        gray1 = mask_edges(gray1)
        gray2 = mask_edges(gray2)
    orb = cv2.ORB_create(500)  # type: ignore[attr-defined]
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        return True, 0.0, (0.0, 0.0), 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < min_matches:
        return True, 0.0, (0.0, 0.0), len(matches)
    src_pts = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1,2)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1,2)
    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is not None:
        dx, dy = M[0,2], M[1,2]
        angle = np.arctan2(M[1,0], M[0,0]) * 180 / np.pi
        is_stable = abs(angle) < angle_threshold and np.hypot(dx, dy) < translation_threshold
        return is_stable, angle, (dx, dy), len(matches)
    return True, 0.0, (0.0, 0.0), len(matches)

def find_enhanced_stable_start_frame(
    video_path, fps, max_check_frames=300, focus_threshold=50, 
    motion_threshold=200, edge_threshold=30, bg_threshold=20,
    min_consecutive_stable=10,
    orb_frame_gap=5, orb_translation_threshold=0.2, orb_required_consecutive_stable=5,
    tqdm_bar=None
):
    """
    Enhanced stability detection using multiple methods:
    1. Focus detection (Laplacian variance)
    2. Motion detection (Optical flow)
    3. Edge movement detection (Scene edges)
    4. Background movement detection (Background vs rebar)
    5. ORB-based edge-only global movement detection (checks every orb_frame_gap frames, requires orb_required_consecutive_stable stable results)
    
    Args:
        video_path: Path to video file
        fps: Frames per second
        max_check_frames: Maximum frames to check for stability
        focus_threshold: Focus detection threshold
        motion_threshold: Motion detection threshold
        edge_threshold: Edge movement threshold
        bg_threshold: Background movement threshold
        min_consecutive_stable: Minimum consecutive stable frames required (for focus/motion/edge/bg)
        orb_frame_gap: Frame gap for ORB-based detection (default 5)
        orb_translation_threshold: Translation threshold for ORB-based detection (default 0.2)
        orb_required_consecutive_stable: Number of consecutive stable ORB checks required (default 5)
    
    Returns:
        tuple: (start_frame, start_time_seconds, stability_info)
    """
    import warnings
    from tqdm import tqdm
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    # Remove all print/debug logs, use tqdm.write only for summary
    orb_checked_indices = []
    stable_count = 0
    focus_scores = []
    motion_scores = []
    edge_scores = []
    bg_scores = []
    rebar_scores = []
    frame_times = []
    orb_results = {}
    prev_frame = None
    orb_prev_frame = None
    orb_consecutive_stable = 0
    orb_first_unstable = None
    orb_last_unstable = None
    orb_first_stable_after_unstable = None
    orb_in_unstable = False
    orb_idx = 0
    # Only show a single summary at the end
    for frame_idx in range(max_check_frames):
        success, frame = cap.read()
        if not success:
            break
        timestamp = frame_idx / fps
        is_focused, focus_score = detect_focus_quality(frame, focus_threshold)
        is_motion_stable, motion_score, motion_vectors = detect_motion_quality(
            frame, prev_frame, motion_threshold
        )
        is_edge_stable, edge_score, edge_shifts = detect_edge_movement(
            frame, prev_frame, edge_threshold
        )
        is_bg_stable, bg_score, rebar_score = detect_background_movement(
            frame, prev_frame, bg_threshold=bg_threshold
        )
        focus_scores.append(focus_score)
        motion_scores.append(motion_score)
        edge_scores.append(edge_score)
        bg_scores.append(bg_score)
        rebar_scores.append(rebar_score)
        frame_times.append(timestamp)
        orb_stable = True
        if frame_idx >= orb_frame_gap and frame_idx % orb_frame_gap == 0:
            orb_checked_indices.append(frame_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - orb_frame_gap)
            _, orb_prev_frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # restore position
            orb_stable, orb_angle, orb_trans, orb_matches = detect_orb_rotation(
                frame, orb_prev_frame, translation_threshold=orb_translation_threshold, frame_gap=orb_frame_gap, mask_center=True
            )
            orb_results[frame_idx] = orb_stable
            if not orb_stable:
                if orb_first_unstable is None:
                    orb_first_unstable = frame_idx
                orb_last_unstable = frame_idx
                orb_in_unstable = True
                orb_consecutive_stable = 0
            elif orb_in_unstable:
                orb_consecutive_stable += 1
                if orb_consecutive_stable == orb_required_consecutive_stable and orb_first_stable_after_unstable is None:
                    orb_first_stable_after_unstable = frame_idx - (orb_required_consecutive_stable - 1) * orb_frame_gap
                    orb_in_unstable = False
        is_frame_stable = is_focused and is_motion_stable and is_edge_stable and is_bg_stable
        if frame_idx >= orb_frame_gap and frame_idx % orb_frame_gap == 0:
            is_frame_stable = is_frame_stable and orb_stable
        if frame_idx < orb_frame_gap:
            is_frame_stable = False
        elif orb_in_unstable and orb_first_stable_after_unstable is None:
            is_frame_stable = False
        elif orb_first_stable_after_unstable is not None and frame_idx < orb_first_stable_after_unstable:
            is_frame_stable = False
        if is_frame_stable:
            stable_count += 1
            if stable_count >= min_consecutive_stable:
                if orb_first_stable_after_unstable is not None and frame_idx >= orb_first_stable_after_unstable:
                    cap.release()
                    start_time = timestamp - (min_consecutive_stable / fps)
                    start_frame = max(0, frame_idx - min_consecutive_stable)
                    from tqdm import tqdm
                    tqdm.write(f"✅ Enhanced stable frame detected at frame {start_frame} (time: {start_time:.2f}s)")
                    return start_frame, start_time, {
                        'focus_scores': focus_scores,
                        'motion_scores': motion_scores,
                        'edge_scores': edge_scores,
                        'bg_scores': bg_scores,
                        'rebar_scores': rebar_scores,
                        'frame_times': frame_times,
                        'focus_threshold': focus_threshold,
                        'motion_threshold': motion_threshold,
                        'edge_threshold': edge_threshold,
                        'bg_threshold': bg_threshold,
                        'orb_first_unstable': orb_first_unstable,
                        'orb_last_unstable': orb_last_unstable,
                        'orb_first_stable_after_unstable': orb_first_stable_after_unstable,
                        'orb_frame_gap': orb_frame_gap,
                        'orb_translation_threshold': orb_translation_threshold,
                        'orb_required_consecutive_stable': orb_required_consecutive_stable
                    }
                elif orb_first_unstable is None and frame_idx >= 50:
                    cap.release()
                    start_time = timestamp - (min_consecutive_stable / fps)
                    start_frame = max(0, frame_idx - min_consecutive_stable)
                    from tqdm import tqdm
                    tqdm.write(f"✅ Enhanced stable frame detected at frame {start_frame} (time: {start_time:.2f}s)")
                    return start_frame, start_time, {
                        'focus_scores': focus_scores,
                        'motion_scores': motion_scores,
                        'edge_scores': edge_scores,
                        'bg_scores': bg_scores,
                        'rebar_scores': rebar_scores,
                        'frame_times': frame_times,
                        'focus_threshold': focus_threshold,
                        'motion_threshold': motion_threshold,
                        'edge_threshold': edge_threshold,
                        'bg_threshold': bg_threshold,
                        'orb_first_unstable': orb_first_unstable,
                        'orb_last_unstable': orb_last_unstable,
                        'orb_first_stable_after_unstable': orb_first_stable_after_unstable,
                        'orb_frame_gap': orb_frame_gap,
                        'orb_translation_threshold': orb_translation_threshold,
                        'orb_required_consecutive_stable': orb_required_consecutive_stable
                    }
        else:
            stable_count = 0
        prev_frame = frame.copy()
    cap.release()
    from tqdm import tqdm
    tqdm.write(f"⚠️ No enhanced stable frames detected in first {max_check_frames} frames")
    return 0, 0.0, {
        'focus_scores': focus_scores,
        'motion_scores': motion_scores,
        'edge_scores': edge_scores,
        'bg_scores': bg_scores,
        'rebar_scores': rebar_scores,
        'frame_times': frame_times,
        'focus_threshold': focus_threshold,
        'motion_threshold': motion_threshold,
        'edge_threshold': edge_threshold,
        'bg_threshold': bg_threshold,
        'orb_first_unstable': orb_first_unstable,
        'orb_last_unstable': orb_last_unstable,
        'orb_first_stable_after_unstable': orb_first_stable_after_unstable,
        'orb_frame_gap': orb_frame_gap,
        'orb_translation_threshold': orb_translation_threshold,
        'orb_required_consecutive_stable': orb_required_consecutive_stable
    }

def find_focus_start_frame(video_path, fps, max_check_frames=300, focus_threshold=50, min_consecutive_focused=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    focused_count = 0
    focus_scores = []
    frame_times = []
    # Remove all print/logs except final summary
    for frame_idx in range(max_check_frames):
        success, frame = cap.read()
        if not success:
            break
        timestamp = frame_idx / fps
        is_focused, focus_score = detect_focus_quality(frame, focus_threshold)
        focus_scores.append(focus_score)
        frame_times.append(timestamp)
        if is_focused:
            focused_count += 1
            if focused_count >= min_consecutive_focused:
                cap.release()
                start_time = timestamp - (min_consecutive_focused / fps)
                start_frame = max(0, frame_idx - min_consecutive_focused)
                from tqdm import tqdm
                tqdm.write(f"✅ Focus detected at frame {start_frame} (time: {start_time:.2f}s)")
                return start_frame, start_time, {
                    'focus_scores': focus_scores,
                    'frame_times': frame_times,
                    'threshold': focus_threshold
                }
        else:
            focused_count = 0
    cap.release()
    from tqdm import tqdm
    tqdm.write(f"⚠️ No stable focus detected in first {max_check_frames} frames")
    return 0, 0.0, {'focus_scores': focus_scores, 'frame_times': frame_times, 'threshold': focus_threshold}

def extract_frames(
    video_path, 
    output_folder, 
    every_n_frames=1, 
    skip_start_frames=0,
    skip_start_seconds=0.0,
    auto_detect_focus=True,
    auto_detect_motion=True,
    focus_threshold=50,
    motion_threshold=200,
    edge_threshold=30,
    bg_threshold=20,
    min_consecutive_stable=10,
    progress_callback=None,
    cancel_event=None
):
    """
    Extract frames from video with enhanced focus and motion detection and flexible skipping options.
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

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("FPS is 0, cannot proceed.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    main_bar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
    notify_progress(0.0, f"Opened video: {video_path}, FPS: {fps}, Total frames: {total_frames}")
    
    # Calculate skip frames from seconds
    skip_frames_from_seconds = int(skip_start_seconds * fps)
    
    # Determine final skip frames (max of frame-based and time-based)
    final_skip_frames = max(skip_start_frames, skip_frames_from_seconds)
    
    # Auto-detect stability if enabled
    stability_skip_frames = 0
    stability_skip_seconds = 0.0
    stability_info = None
    
    if auto_detect_focus and auto_detect_motion:
        notify_progress(0.01, "Detecting camera focus and motion stability...")
        stability_skip_frames, stability_skip_seconds, stability_info = find_enhanced_stable_start_frame(
            video_path, fps, focus_threshold=focus_threshold, 
            motion_threshold=motion_threshold, edge_threshold=edge_threshold,
            bg_threshold=bg_threshold, min_consecutive_stable=min_consecutive_stable,
            orb_frame_gap=5, orb_translation_threshold=0.2, orb_required_consecutive_stable=5
        )
    elif auto_detect_focus:
        notify_progress(0.01, "Detecting camera focus...")
        stability_skip_frames, stability_skip_seconds, stability_info = find_focus_start_frame(
            video_path, fps, focus_threshold=focus_threshold, 
            min_consecutive_focused=min_consecutive_stable
        )
    
    final_skip_frames = max(final_skip_frames, stability_skip_frames)
    
    # Report skipping strategy
    if final_skip_frames > 0:
        skip_time = final_skip_frames / fps
        notify_progress(0.02, f"⏭️ Skipping {final_skip_frames} frames ({skip_time:.2f}s)")
        if auto_detect_focus and auto_detect_motion and stability_skip_frames > 0:
            notify_progress(0.02, f"   - {stability_skip_frames} frames ({stability_skip_seconds:.2f}s) due to focus/motion detection")
        elif auto_detect_focus and stability_skip_frames > 0:
            notify_progress(0.02, f"   - {stability_skip_frames} frames ({stability_skip_seconds:.2f}s) due to focus detection")
        if skip_start_frames > 0:
            notify_progress(0.02, f"   - {skip_start_frames} frames due to manual frame skip")
        if skip_frames_from_seconds > 0:
            notify_progress(0.02, f"   - {skip_frames_from_seconds} frames due to {skip_start_seconds}s time skip")
    
    # Reset video to start
    cap.release()
    cap = cv2.VideoCapture(video_path)
    
    frame_index = 0
    saved_index = 0
    
    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        main_bar = pbar
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Skip frames based on calculated skip amount
            if frame_index < final_skip_frames:
                frame_index += 1
                pbar.update(1)
                continue

            if frame_index % every_n_frames == 0:
                timestamp = frame_index / fps
                filename = os.path.join(output_folder, f"frame_{saved_index:04d}_{timestamp:.2f}s.jpg")
                cv2.imwrite(filename, frame)
                saved_index += 1
                # Show per-frame status in tqdm
                main_bar.set_postfix({
                    "saved": saved_index,
                    "frame": frame_index,
                    "time": f"{timestamp:.2f}s"
                })
            
            if cancel_event and cancel_event.is_set():
                notify_progress(1.0, "Processing cancelled by user.")
                break

            frame_index += 1
            pbar.update(1)

    cap.release()
    
    # Save stability detection info if available
    if stability_info:
        stability_log_path = os.path.join(output_folder, "stability_detection_log.txt")
        with open(stability_log_path, 'w') as f:
            f.write(f"Stability Detection Results\n")
            f.write(f"==========================\n")
            f.write(f"Focus threshold: {stability_info.get('focus_threshold', focus_threshold)}\n")
            f.write(f"Motion threshold: {stability_info.get('motion_threshold', motion_threshold)}\n")
            f.write(f"Min consecutive stable frames: {min_consecutive_stable}\n")
            f.write(f"Frames skipped due to stability: {stability_skip_frames}\n")
            f.write(f"Time skipped due to stability: {stability_skip_seconds:.2f}s\n")
            
            if 'focus_scores' in stability_info:
                f.write(f"Focus scores range: {min(stability_info['focus_scores']):.1f} - {max(stability_info['focus_scores']):.1f}\n")
                f.write(f"Average focus score: {np.mean(stability_info['focus_scores']):.1f}\n")
            
            if 'motion_scores' in stability_info:
                f.write(f"Motion scores range: {min(stability_info['motion_scores']):.1f} - {max(stability_info['motion_scores']):.1f}\n")
                f.write(f"Average motion score: {np.mean(stability_info['motion_scores']):.1f}\n")
    
    notify_progress(0.05, f"Total frames saved: {saved_index}")
    main_bar.close()
    return {
        'frames_saved': saved_index,
        'frames_skipped': final_skip_frames,
        'time_skipped': final_skip_frames / fps,
        'stability_info': stability_info
    }

if __name__ == "__main__":
    # Example usage with focus and motion detection
    result = extract_frames(
        "40kn-2.mp4", 
        "output_frames", 
        every_n_frames=5,
        auto_detect_focus=True,
        auto_detect_motion=True,
        skip_start_seconds=2.0  # Skip first 2 seconds
    )
    print(f"Extraction result: {result}")