# Project Review: Elongation-Rebar

## Project Overview

**Elongation-Rebar** is a Python-based system for analyzing the elongation of rebar (reinforcing steel bars) during mechanical testing, using video footage. The project extracts frames from test videos, detects and tracks natural texture patterns on the rebar (rather than relying on paint marks), and computes elongation over time. The analysis includes:

- Frame extraction from video
- Pattern extraction and matching using the rebar's natural texture (focused on the 3rd and 7th grid lines, i.e., 30% and 70% of the image height)
- Visualization with fixed reference (yellow) and current (blue) markers
- Elongation calculation and smoothing
- Pixel-to-mm conversion and force estimation
- Batch processing and PDF report generation
- FastAPI backend for web-based interaction

## Project Structure

- `elongation/` — Core image processing, pattern extraction, analysis, and reporting modules
- `backend/` — FastAPI backend for file upload, processing, and API endpoints
- `frontend/` — (Nuxt-based) web frontend (not covered in this review)
- `sample/` — Sample test videos
- `results/` — Output data, plots, and reports
- `process_video.py` — CLI entry point for single video processing
- `PATTERN_EXTRACTION_CHECKLIST.md` — Stepwise checklist for pattern extraction and matching logic
- `.cursor` — Editor navigation hints

## Key Features

- **Pattern Extraction:** Uses the natural texture of the rebar, extracting patterns at fixed grid lines (30% and 70% of height) for robust, paint-free tracking.
- **Reference and Current Markers:** Yellow lines (reference) are fixed at the best pattern locations from the first frame; blue lines (current) are drawn at matched locations in each frame.
- **Batch and Single Processing:** Supports both batch automation and single video analysis.
- **Pixel-to-mm Conversion:** Allows calibration for real-world measurements.
- **Force Curve Estimation:** Computes estimated force using material properties.
- **PDF Reporting:** Generates summary reports with plots.

## Parameter Documentation

### Frame Extraction Parameters (`extract_frames`)

- **`every_n_frames`** (default: 1): Extract every Nth frame from the video
  - **1**: Extract all frames (highest temporal resolution, slowest processing)
  - **5**: Extract every 5th frame (balanced resolution/speed)
  - **10+**: Extract every 10th+ frame (lower resolution, faster processing)
  - **Trade-off**: Higher values = faster processing but lower temporal resolution

#### Focus Detection and Skipping Parameters

- **`skip_start_frames`** (default: 0): Skip N frames from the start (manual frame-based skipping)
- **`skip_start_seconds`** (default: 0.0): Skip N seconds from the start (manual time-based skipping)
- **`auto_detect_focus`** (default: True): Automatically detect camera focus and skip unfocused frames
- **`auto_detect_motion`** (default: True): Automatically detect camera movement/shake and skip unstable frames
- **`focus_threshold`** (default: 50): Laplacian variance threshold for focus detection
  - **Lower values (30-40)**: More sensitive, may include slightly blurry frames
  - **Higher values (60-80)**: More strict, only very sharp frames
- **`motion_threshold`** (default: 1000): Optical flow threshold for motion detection
  - **Lower values (500-800)**: More sensitive, detects smaller movements
  - **Higher values (1500-2000)**: More strict, only detects significant camera shake
- **`min_consecutive_stable`** (default: 15): Minimum consecutive stable frames required
  - **Smaller values (10-12)**: Faster detection, may start with brief stability
  - **Larger values (20-25)**: More stable period required, slower detection

#### Focus and Motion Detection Algorithms

The system uses two complementary detection methods:

**Focus Detection (Laplacian Variance):**

- **Sharp images**: High variance (sharp edges, good focus)
- **Blurry images**: Low variance (smooth gradients, poor focus)

**Motion Detection (Optical Flow):**

- **Stable camera**: Low motion scores (minimal movement between frames)
- **Camera shake/movement**: High motion scores (significant displacement)
- **Grid-based tracking**: Samples points across the image for comprehensive motion analysis
- **Combined scoring**: Weighted combination of mean, max, and variance of motion vectors

**Stability Requirements:**

- **Consecutive detection**: Requires multiple stable frames to ensure consistent quality
- **Dual validation**: Frame must be both focused AND motion-free
- **Automatic logging**: Saves stability detection results to `stability_detection_log.txt`

### Pattern Detection Parameters (`process_images`)

#### Pattern Size and Search Parameters

- **`max_pattern_height`** (default: 12): Height of pattern windows to extract in pixels
  - **Smaller values (8-10)**: More precise but may miss larger texture features
  - **Larger values (15-20)**: Capture more texture but may include noise
- **`pattern_width`** (default: 8): Width of pattern windows to extract in pixels
  - **Smaller values (6-8)**: Focus on vertical texture patterns
  - **Larger values (10-12)**: Include more horizontal texture information

#### Search Margins

- **`search_margin_y`** (default: 5): Vertical search margin around grid lines in pixels
  - **Smaller values (3-5)**: More precise but may miss patterns if rebar moves slightly
  - **Larger values (8-12)**: More robust to vertical movement but slower processing
- **`search_margin_x`** (default: 5): Horizontal search margin around rebar center in pixels
  - **Smaller values (3-5)**: Focus on center of rebar
  - **Larger values (8-12)**: Account for horizontal rebar movement

#### Pattern Capture Parameters

- **`pattern_capture_frames`** (default: 10): Number of initial frames to extract patterns from
  - **Smaller values (5-8)**: Faster but may miss optimal patterns
  - **Larger values (15-20)**: More thorough pattern selection but slower
- **`pattern_capture_step`** (default: 5): Step between frames for pattern extraction
  - **Smaller values (2-3)**: More frames analyzed, better pattern selection
  - **Larger values (8-10)**: Faster processing, fewer frames analyzed

#### Quality Control Parameters

- **`top_n_to_keep`** (default: 5): Number of best pattern pairs to keep
  - **Smaller values (3-5)**: Focus on best patterns, faster processing
  - **Larger values (8-10)**: More options, potentially better results
- **`prune_threshold`** (default: 1e7): Score threshold for pruning poor matches
  - **Lower values (1e6)**: More strict filtering, fewer false positives
  - **Higher values (1e8)**: More lenient filtering, may include noise

#### Detection Parameters

- **`scan_width`** (default: 10): Width of strip for rebar center detection in pixels
  - **Smaller values (6-8)**: More precise center detection
  - **Larger values (12-15)**: More robust to rebar positioning
- **`threshold_ratio`** (default: 0.3): Ratio for gradient threshold in marker detection
  - **Lower values (0.2)**: More sensitive detection, may include noise
  - **Higher values (0.5)**: Less sensitive, may miss subtle patterns
- **`min_valid_distance`** (default: 50): Minimum distance between top/bottom patterns in pixels
  - **Smaller values (30-40)**: Allow closer patterns, may be less reliable
  - **Larger values (60-80)**: Require more separation, more reliable gauge length
- **`max_band_thickness`** (default: 10): Maximum thickness of detected bands in pixels
  - **Smaller values (6-8)**: Focus on thin, distinct features
  - **Larger values (12-15)**: Include thicker texture bands

### Heavy Processing Recommendations

For better marking quality, consider these parameter adjustments:

**High-Quality Settings:**

- `every_n_frames=1` (extract all frames)
- `max_pattern_height=15`, `pattern_width=10` (larger patterns)
- `search_margin_y=8`, `search_margin_x=8` (larger search areas)
- `pattern_capture_frames=15`, `pattern_capture_step=3` (more thorough pattern selection)
- `top_n_to_keep=8` (more pattern options)
- `prune_threshold=5e6` (stricter filtering)

**Performance vs Quality Trade-offs:**

- **Speed**: Increase `

## ORB-Based Edge-Only, Consecutive-Stable Global Movement Detection (UPDATED)

- **Purpose:** Detects subtle global camera movements, including rotation and translation, using ORB feature matching between frames, focusing only on the left/right 20% of the image (ignoring the rebar in the center).
- **How it works:**
  - Extracts keypoints and descriptors from the edges of both frames using ORB (Oriented FAST and Rotated BRIEF).
  - Matches keypoints and estimates the affine transform (rotation, translation).
  - Flags frames as unstable if the estimated rotation or translation exceeds a threshold.
  - Checks every `orb_frame_gap` frames (default: 5), and requires `orb_required_consecutive_stable` (default: 5) consecutive stable results after instability before considering the video stable.
- **Why:**
  - More robust than optical flow or edge-based methods for detecting global scene changes, especially in environments with subtle features (e.g., cracks, machine edges).
  - Ignores rebar movement by masking out the center, focusing only on the background.
  - Works across different backgrounds and lighting conditions, as long as some texture is present.
- **Integration:**
  - Available as `detect_orb_rotation(frame, prev_frame, ...)` in `elongation/extract_frames.py`.
  - Fully integrated into the enhanced stability detection pipeline (`find_enhanced_stable_start_frame`) and main frame extraction logic.
  - All stability checks (focus, motion, edge, background, ORB) must pass for a frame to be considered stable.
- **Parameters:**
  - `orb_frame_gap`: Frame gap for ORB-based edge-only detection (default: 5)
  - `orb_translation_threshold`: Translation threshold for ORB-based detection (default: 0.2)
  - `orb_required_consecutive_stable`: Number of consecutive stable ORB checks required (default: 5)
  - `min_matches`: Minimum number of feature matches required for reliable detection (default: 10)
  - `angle_threshold`: Maximum allowed rotation (degrees) for a frame to be considered stable (default: 0.5)

**Recommendation:** Use ORB-based edge-only, consecutive-stable detection for robust, environment-independent camera stability analysis, especially when subtle global movement or rotation is a concern. All stability checks must pass for a frame to be considered stable and used for elongation analysis.
